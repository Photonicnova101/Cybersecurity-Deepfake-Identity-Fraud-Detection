"""
Ensemble model combining multiple deepfake detectors
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict
import logging

from src.models.efficientnet import EfficientNetModel
from src.models.xception import XceptionModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of multiple models for improved detection"""
    
    def __init__(self, model_configs: List[Dict] = None):
        """
        Initialize ensemble model
        
        Args:
            model_configs: List of model configuration dictionaries
        """
        self.models = []
        self.model_names = []
        self.weights = []
        
        if model_configs is None:
            # Default ensemble
            self.model_configs = [
                {'type': 'efficientnet', 'variant': 'b0', 'weight': 1.0},
                {'type': 'xception', 'weight': 1.0}
            ]
        else:
            self.model_configs = model_configs
    
    def load_models(self, model_paths: List[str]):
        """
        Load pre-trained models
        
        Args:
            model_paths: List of paths to saved models
        """
        if len(model_paths) != len(self.model_configs):
            raise ValueError("Number of model paths must match number of configurations")
        
        for path, config in zip(model_paths, self.model_configs):
            try:
                model = tf.keras.models.load_model(path)
                self.models.append(model)
                self.model_names.append(config['type'])
                self.weights.append(config.get('weight', 1.0))
                logger.info(f"Loaded {config['type']} model from {path}")
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
                raise
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, image: np.ndarray, method: str = 'average') -> Dict:
        """
        Make prediction using ensemble
        
        Args:
            image: Input image (or batch of images)
            method: Ensemble method ('average', 'weighted', 'voting', 'max')
            
        Returns:
            Dictionary with prediction results
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded. Call load_models() first.")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(image, verbose=0)
            predictions.append(pred)
        
        # Combine predictions
        if method == 'average':
            combined = np.mean(predictions, axis=0)
        elif method == 'weighted':
            combined = np.average(predictions, axis=0, weights=self.weights)
        elif method == 'voting':
            # Majority voting
            binary_preds = [pred >= 0.5 for pred in predictions]
            combined = np.mean(binary_preds, axis=0)
        elif method == 'max':
            combined = np.max(predictions, axis=0)
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")
        
        # Get individual model results
        individual_results = []
        for i, (pred, name) in enumerate(zip(predictions, self.model_names)):
            individual_results.append({
                'model': name,
                'prediction': 'fake' if pred[0][0] >= 0.5 else 'real',
                'confidence': float(pred[0][0] if pred[0][0] >= 0.5 else 1 - pred[0][0]),
                'raw_score': float(pred[0][0])
            })
        
        # Final ensemble result
        is_fake = combined[0][0] >= 0.5
        confidence = combined[0][0] if is_fake else (1 - combined[0][0])
        
        return {
            'prediction': 'fake' if is_fake else 'real',
            'confidence': float(confidence),
            'raw_score': float(combined[0][0]),
            'ensemble_method': method,
            'individual_models': individual_results
        }
    
    def predict_batch(self, images: np.ndarray, method: str = 'average') -> List[Dict]:
        """
        Make predictions on batch of images
        
        Args:
            images: Batch of images
            method: Ensemble method
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(np.expand_dims(image, axis=0), method=method)
            results.append(result)
        return results
    
    def evaluate_ensemble(self, 
                         test_data: tf.data.Dataset,
                         method: str = 'average') -> Dict:
        """
        Evaluate ensemble on test data
        
        Args:
            test_data: Test dataset
            method: Ensemble method
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_labels = []
        
        for images, labels in test_data:
            for image, label in zip(images, labels):
                result = self.predict(np.expand_dims(image, axis=0), method=method)
                all_predictions.append(result['raw_score'])
                all_labels.append(label.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        binary_preds = (all_predictions >= 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': float(accuracy_score(all_labels, binary_preds)),
            'precision': float(precision_score(all_labels, binary_preds)),
            'recall': float(recall_score(all_labels, binary_preds)),
            'f1_score': float(f1_score(all_labels, binary_preds)),
            'auc_roc': float(roc_auc_score(all_labels, all_predictions))
        }
        
        return metrics
    
    def get_model_info(self) -> Dict:
        """Get information about ensemble models"""
        return {
            'num_models': len(self.models),
            'models': [
                {
                    'name': name,
                    'weight': weight,
                    'parameters': model.count_params()
                }
                for name, weight, model in zip(self.model_names, self.weights, self.models)
            ]
        }


def create_ensemble_from_paths(model_paths: List[str],
                               model_configs: List[Dict],
                               ensemble_method: str = 'weighted') -> EnsembleModel:
    """
    Helper function to create ensemble from saved models
    
    Args:
        model_paths: Paths to saved models
        model_configs: Model configurations
        ensemble_method: Default ensemble method
        
    Returns:
        EnsembleModel instance
    """
    ensemble = EnsembleModel(model_configs)
    ensemble.load_models(model_paths)
    
    logger.info(f"Created ensemble with {len(model_paths)} models")
    
    return ensemble
