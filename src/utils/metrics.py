"""
Performance metrics for deepfake detection
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve
)
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     y_prob: np.ndarray = None) -> Dict:
    """
    Calculate comprehensive performance metrics
    
    Args:
        y_true: True labels (0 for real, 1 for fake)
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def compute_roc_curve(y_true: np.ndarray, 
                     y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_prob)


def calculate_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Equal Error Rate
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        
    Returns:
        Equal Error Rate
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    
    # Find index where FPR and FNR are closest
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return float(eer)
