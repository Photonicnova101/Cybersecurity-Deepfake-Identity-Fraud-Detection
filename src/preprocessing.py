"""
Preprocessing pipeline for images and videos
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from mtcnn import MTCNN
import logging

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Handles preprocessing of images for deepfake detection"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 face_margin: float = 0.2,
                 normalize: bool = True):
        """
        Initialize preprocessing pipeline
        
        Args:
            target_size: Target image size for model input
            face_margin: Margin around detected face (0.2 = 20%)
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.face_margin = face_margin
        self.normalize = normalize
        
        # Initialize face detector
        self.face_detector = MTCNN()
        logger.info("Initialized preprocessing pipeline")
    
    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect face in image using MTCNN
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Face detection result or None
        """
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if len(faces) == 0:
                return None
            
            # Return the face with highest confidence
            faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
            return faces[0]
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None
    
    def crop_face(self, 
                  image: np.ndarray, 
                  face_box: dict) -> np.ndarray:
        """
        Crop face from image with margin
        
        Args:
            image: Input image
            face_box: Face bounding box from detector
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face_box['box']
        
        # Add margin
        margin_x = int(w * self.face_margin)
        margin_y = int(h * self.face_margin)
        
        # Calculate crop coordinates
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        return face
    
    def preprocess_image(self, 
                        image: Image.Image) -> Tuple[np.ndarray, bool]:
        """
        Complete preprocessing pipeline for an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (preprocessed image, face_detected flag)
        """
        try:
            # Convert to numpy array
            image_array = np.array(image)
            
            # Convert to RGB if needed
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Detect face
            face_result = self.detect_face(image_array)
            
            if face_result is None:
                # No face detected, resize whole image
                resized = cv2.resize(image_array, self.target_size)
                if self.normalize:
                    resized = resized.astype('float32') / 255.0
                return resized, False
            
            # Crop face
            face = self.crop_face(image_array, face_result)
            
            # Resize to target size
            face_resized = cv2.resize(face, self.target_size)
            
            # Normalize if requested
            if self.normalize:
                face_resized = face_resized.astype('float32') / 255.0
            
            return face_resized, True
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            # Return default preprocessed image
            default_image = np.zeros((*self.target_size, 3))
            return default_image, False
    
    def preprocess_batch(self, 
                        images: list) -> Tuple[np.ndarray, list]:
        """
        Preprocess multiple images
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Tuple of (batch array, list of face detection flags)
        """
        processed_images = []
        face_flags = []
        
        for image in images:
            processed, face_detected = self.preprocess_image(image)
            processed_images.append(processed)
            face_flags.append(face_detected)
        
        batch_array = np.array(processed_images)
        return batch_array, face_flags
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation (for training)
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, 
                                  (image.shape[1], image.shape[0]))
        
        return image
    
    def extract_facial_landmarks(self, 
                                 image: np.ndarray) -> Optional[dict]:
        """
        Extract facial landmarks for advanced analysis
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with landmark positions
        """
        face_result = self.detect_face(image)
        
        if face_result is None or 'keypoints' not in face_result:
            return None
        
        keypoints = face_result['keypoints']
        
        return {
            'left_eye': keypoints.get('left_eye'),
            'right_eye': keypoints.get('right_eye'),
            'nose': keypoints.get('nose'),
            'mouth_left': keypoints.get('mouth_left'),
            'mouth_right': keypoints.get('mouth_right')
        }
