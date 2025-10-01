"""
Visualization utilities for deepfake detection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def generate_heatmap(model: tf.keras.Model, 
                     image: np.ndarray,
                     last_conv_layer_name: str = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap showing attention regions
    
    Args:
        model: Trained model
        image: Preprocessed image
        last_conv_layer_name: Name of last convolutional layer
        
    Returns:
        Heatmap as numpy array
    """
    try:
        # Find last convolutional layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        
        # Get gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            loss = predictions[0]
        
        # Get gradients of loss with respect to conv layer
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        # Resize to match image size
        heatmap = cv2.resize(heatmap.numpy(), (image.shape[1], image.shape[0]))
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return np.zeros((image.shape[0], image.shape[1]))


def overlay_heatmap(image: np.ndarray, 
                   heatmap: np.ndarray,
                   alpha: float = 0.4) -> np.ndarray:
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image
        heatmap: Heatmap to overlay
        alpha: Transparency factor
        
    Returns:
        Combined image
    """
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), 
        cv2.COLORMAP_JET
    )
    
    # Convert image to uint8 if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def plot_results(results: dict, save_path: Optional[str] = None):
    """
    Plot detection results
    
    Args:
        results: Dictionary with detection results
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot confidence score
    prediction = results['prediction']
    confidence = results['confidence']
    
    colors = ['green' if prediction == 'real' else 'red']
    axes[0].bar(['Confidence'], [confidence], color=colors)
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Confidence Score')
    axes[0].set_title(f'Prediction: {prediction.upper()}')
    axes[0].axhline(y=0.7, color='black', linestyle='--', label='Threshold')
    axes[0].legend()
    
    # Plot distribution if available
    if 'frame_results' in results:
        frame_scores = [fr['confidence'] for fr in results['frame_results']]
        axes[1].hist(frame_scores, bins=20, color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Frame Scores')
    else:
        axes[1].text(0.5, 0.5, 'No frame data available',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
