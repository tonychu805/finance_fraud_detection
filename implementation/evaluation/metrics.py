"""
Metrics module for model evaluation.
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve, average_precision_score
)

class MetricsCalculator:
    """Calculator for various model evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive set of metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metric names and their values
        """
        return {
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
            "average_precision": float(average_precision_score(y_true, y_pred_proba)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        import matplotlib.pyplot as plt
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        import matplotlib.pyplot as plt
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 