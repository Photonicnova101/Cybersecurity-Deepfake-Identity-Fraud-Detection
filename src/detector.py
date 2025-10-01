"""
Main deepfake detection engine
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Union, Tuple
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import logging

from src.preprocessing import PreprocessingPipeline
from src.utils.visualization import generate_heatmap
from src.utils.video_utils import extract_frames

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """Main class for deepfake detection"""
    
    def __init__(self, 
                 model_path: str = 'models/efficientnet_b0.h5',
                 confidence_threshold: float = 0.7,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the deepfake detector
        
        Args:
            model_path: Path to the trained model
            confidence_threshold: Minimum confidence for positive detection
            input_size: Input size expected by the model
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.preprocessor = PreprocessingPipeline(target_size=input_size)
        
        # Load model
        self.model = self._load_model()
        logger.info(f"Loaded model from {model_path}")
    
    def _load_model(self) -> tf.keras.Model:
        """Load the pre-trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            model = load_model(self.model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_image(self, 
                     image_path: str, 
                     return_heatmap: bool = False) -> Dict:
        """
        Predict if an image contains a deepfake
        
        Args:
            image_path: Path to the image file
            return_heatmap: Whether to generate attention heatmap
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image, face_detected = self.preprocessor.preprocess_image(image)
            
            if not face_detected:
                return {
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'message': 'No face detected in image',
                    'face_detected': False
                }
            
            # Make prediction
            processed_batch = np.expand_dims(processed_image, axis=0)
            prediction = self.model.predict(processed_batch, verbose=0)[0][0]
            
            # Interpret results
            is_fake = prediction >= self.confidence_threshold
            confidence = prediction if is_fake else (1 - prediction)
            
            result = {
                'prediction': 'fake' if is_fake else 'real',
                'confidence': float(confidence),
                'raw_score': float(prediction),
                'face_detected': True,
                'image_path': image_path
            }
            
            # Generate heatmap if requested
            if return_heatmap:
                heatmap = generate_heatmap(self.model, processed_image)
                result['heatmap'] = heatmap
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'message': str(e),
                'face_detected': False
            }
    
    def predict_video(self, 
                     video_path: str,
                     frame_interval: int = 30,
                     return_detailed: bool = False) -> Dict:
        """
        Predict if a video contains deepfakes
        
        Args:
            video_path: Path to the video file
            frame_interval: Analyze every Nth frame
            return_detailed: Return per-frame results
            
        Returns:
            Dictionary with video analysis results
        """
        try:
            # Extract frames
            frames = extract_frames(video_path, interval=frame_interval)
            logger.info(f"Extracted {len(frames)} frames from video")
            
            if len(frames) == 0:
                return {
                    'prediction': 'error',
                    'message': 'Could not extract frames from video'
                }
            
            # Analyze each frame
            frame_results = []
            fake_count = 0
            total_confidence = 0.0
            
            for idx, frame in enumerate(frames):
                # Preprocess frame
                processed_frame, face_detected = self.preprocessor.preprocess_image(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
                
                if not face_detected:
                    continue
                
                # Predict
                processed_batch = np.expand_dims(processed_frame, axis=0)
                prediction = self.model.predict(processed_batch, verbose=0)[0][0]
                
                is_fake = prediction >= self.confidence_threshold
                confidence = prediction if is_fake else (1 - prediction)
                
                if is_fake:
                    fake_count += 1
                
                total_confidence += confidence
                
                frame_result = {
                    'frame_number': idx * frame_interval,
                    'prediction': 'fake' if is_fake else 'real',
                    'confidence': float(confidence),
                    'raw_score': float(prediction)
                }
                
                frame_results.append(frame_result)
            
            # Aggregate results
            if len(frame_results) == 0:
                return {
                    'prediction': 'unknown',
                    'message': 'No faces detected in video frames'
                }
            
            fake_percentage = (fake_count / len(frame_results)) * 100
            avg_confidence = total_confidence / len(frame_results)
            
            video_result = {
                'prediction': 'fake' if fake_percentage > 50 else 'real',
                'confidence': float(avg_confidence),
                'fake_frame_percentage': float(fake_percentage),
                'total_frames_analyzed': len(frame_results),
                'fake_frames': fake_count,
                'real_frames': len(frame_results) - fake_count,
                'video_path': video_path
            }
            
            if return_detailed:
                video_result['frame_results'] = frame_results
            
            return video_result
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return {
                'prediction': 'error',
                'message': str(e)
            }
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        logger.info(f"Processing batch of {len(image_paths)} images")
        
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'input_size': self.input_size,
            'confidence_threshold': self.confidence_threshold,
            'total_parameters': self.model.count_params(),
            'layers': len(self.model.layers)
        }
