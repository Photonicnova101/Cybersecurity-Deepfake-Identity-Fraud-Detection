"""
Audio deepfake detection module
"""

import numpy as np
import librosa
import tensorflow as tf
from typing import Dict, Tuple
import logging

from src.audio.voice_analysis import VoiceAnalyzer

logger = logging.getLogger(__name__)


class AudioDeepfakeDetector:
    """Detector for synthetic audio and voice cloning"""
    
    def __init__(self,
                 model_path: str = 'models/audio_detector.h5',
                 sample_rate: int = 16000,
                 n_mels: int = 128):
        """
        Initialize audio deepfake detector
        
        Args:
            model_path: Path to trained audio model
            sample_rate: Audio sample rate
            n_mels: Number of mel-frequency bands
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.voice_analyzer = VoiceAnalyzer(sample_rate=sample_rate)
        
        # Load model if exists
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded audio model from {model_path}")
        except:
            logger.warning(f"Could not load model from {model_path}, using feature-based detection")
            self.model = None
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features
        
        Args:
            audio: Audio array
            
        Returns:
            Mel-spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Audio array
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc
        )
        
        # Delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        
        return features
    
    def predict_audio(self, audio_path: str) -> Dict:
        """
        Predict if audio is deepfake
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)
            
            if len(audio) == 0:
                return {
                    'prediction': 'error',
                    'message': 'Empty audio file'
                }
            
            # Extract features
            mel_spec = self.extract_mel_spectrogram(audio)
            
            if self.model is not None:
                # Use neural network model
                # Reshape for model input
                mel_spec_input = np.expand_dims(mel_spec, axis=0)
                mel_spec_input = np.expand_dims(mel_spec_input, axis=-1)
                
                prediction = self.model.predict(mel_spec_input, verbose=0)[0][0]
                
                is_fake = prediction >= 0.5
                confidence = prediction if is_fake else (1 - prediction)
                
                result = {
                    'prediction': 'fake' if is_fake else 'real',
                    'confidence': float(confidence),
                    'raw_score': float(prediction),
                    'audio_path': audio_path
                }
            else:
                # Use feature-based detection
                result = self._feature_based_detection(audio)
                result['audio_path'] = audio_path
            
            # Add voice analysis
            voice_features = self.voice_analyzer.analyze_voice(audio)
            result['voice_features'] = voice_features
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            return {
                'prediction': 'error',
                'message': str(e)
            }
    
    def _feature_based_detection(self, audio: np.ndarray) -> Dict:
        """
        Feature-based detection when no model is available
        
        Args:
            audio: Audio array
            
        Returns:
            Detection result
        """
        # Extract features
        mfcc = self.extract_mfcc(audio)
        
        # Calculate statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Simple heuristic (should be replaced with trained model)
        # This is a placeholder - actual detection would need a trained model
        score = np.mean(np.abs(mfcc_mean))
        
        # Normalize to 0-1 range (approximate)
        normalized_score = min(max(score / 10.0, 0), 1)
        
        is_fake = normalized_score > 0.5
        confidence = normalized_score if is_fake else (1 - normalized_score)
        
        return {
            'prediction': 'fake' if is_fake else 'real',
            'confidence': float(confidence),
            'raw_score': float(normalized_score),
            'method': 'feature_based',
            'note': 'Using feature-based detection. Load a trained model for better accuracy.'
        }
    
    def analyze_audio_quality(self, audio: np.ndarray) -> Dict:
        """
        Analyze audio quality metrics
        
        Args:
            audio: Audio array
            
        Returns:
            Quality metrics
        """
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        # Spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        return {
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'spectral_centroid_mean': float(np.mean(spec_centroid)),
            'spectral_rolloff_mean': float(np.mean(spec_rolloff)),
            'rms_energy_mean': float(np.mean(rms)),
            'duration_seconds': len(audio) / self.sample_rate
        }
