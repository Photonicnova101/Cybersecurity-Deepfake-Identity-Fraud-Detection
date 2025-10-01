"""
Voice feature analysis for deepfake detection
"""

import numpy as np
import librosa
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class VoiceAnalyzer:
    """Analyze voice characteristics for authenticity"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize voice analyzer
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def analyze_voice(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive voice analysis
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with voice features
        """
        features = {}
        
        # Pitch analysis
        features.update(self._analyze_pitch(audio))
        
        # Formant analysis
        features.update(self._analyze_formants(audio))
        
        # Jitter and shimmer (voice quality)
        features.update(self._analyze_voice_quality(audio))
        
        # Spectral features
        features.update(self._analyze_spectral(audio))
        
        return features
    
    def _analyze_pitch(self, audio: np.ndarray) -> Dict:
        """
        Analyze pitch characteristics
        
        Args:
            audio: Audio array
            
        Returns:
            Pitch features
        """
        # Extract pitch using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_range': 0.0
            }
        
        return {
            'pitch_mean': float(np.mean(f0_clean)),
            'pitch_std': float(np.std(f0_clean)),
            'pitch_range': float(np.max(f0_clean) - np.min(f0_clean)),
            'pitch_median': float(np.median(f0_clean))
        }
    
    def _analyze_formants(self, audio: np.ndarray) -> Dict:
        """
        Analyze formant frequencies
        
        Args:
            audio: Audio array
            
        Returns:
            Formant features
        """
        # Get spectrum
        spectrum = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        
        # Find peaks (simplified formant detection)
        mean_spectrum = np.mean(spectrum, axis=1)
        
        # Find first 3 formants (peaks in spectrum)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.1)
        
        formants = freqs[peaks[:3]] if len(peaks) >= 3 else []
        
        if len(formants) < 3:
            formants = list(formants) + [0.0] * (3 - len(formants))
        
        return {
            'formant_f1': float(formants[0]),
            'formant_f2': float(formants[1]),
            'formant_f3': float(formants[2])
        }
    
    def _analyze_voice_quality(self, audio: np.ndarray) -> Dict:
        """
        Analyze voice quality metrics (jitter, shimmer)
        
        Args:
            audio: Audio array
            
        Returns:
            Voice quality features
        """
        # Simplified jitter calculation
        # (Real implementation would use pitch periods)
        f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) < 2:
            return {
                'jitter': 0.0,
                'shimmer': 0.0
            }
        
        # Jitter (pitch variability)
        pitch_diffs = np.diff(f0_clean)
        jitter = np.mean(np.abs(pitch_diffs)) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
        
        # Shimmer (amplitude variability)
        rms = librosa.feature.rms(y=audio)[0]
        rms_diffs = np.diff(rms)
        shimmer = np.mean(np.abs(rms_diffs)) / np.mean(rms) if np.mean(rms) > 0 else 0
        
        return {
            'jitter': float(jitter),
            'shimmer': float(shimmer)
        }
    
    def _analyze_spectral(self, audio: np.ndarray) -> Dict:
        """
        Analyze spectral characteristics
        
        Args:
            audio: Audio array
            
        Returns:
            Spectral features
        """
        # Spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        # Spectral bandwidth
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        # Spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        
        # Spectral flatness
        spec_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spec_centroid)),
            'spectral_bandwidth_mean': float(np.mean(spec_bandwidth)),
            'spectral_contrast_mean': float(np.mean(spec_contrast)),
            'spectral_flatness_mean': float(np.mean(spec_flatness))
        }
    
    def detect_anomalies(self, features: Dict) -> Dict:
        """
        Detect anomalies in voice features that might indicate synthetic speech
        
        Args:
            features: Voice features dictionary
            
        Returns:
            Anomaly detection results
        """
        anomalies = []
        
        # Check for unusual pitch
        if features.get('pitch_std', 0) < 5 or features.get('pitch_std', 0) > 100:
            anomalies.append('unusual_pitch_variation')
        
        # Check for unusual jitter/shimmer
        if features.get('jitter', 0) < 0.001 or features.get('jitter', 0) > 0.1:
            anomalies.append('unusual_jitter')
        
        if features.get('shimmer', 0) < 0.001 or features.get('shimmer', 0) > 0.2:
            anomalies.append('unusual_shimmer')
        
        # Check spectral flatness (synthetic voices often have flatter spectrum)
        if features.get('spectral_flatness_mean', 0) > 0.5:
            anomalies.append('high_spectral_flatness')
        
        return {
            'has_anomalies': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies
        }
