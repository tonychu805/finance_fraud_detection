"""
Cross-validation functionality for fraud detection models.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from fraud_detection.models.core.ensemble import FraudEnsemble
from fraud_detection.models.core.lightgbm import LightGBMModel
from fraud_detection.models.core.random_forest import RandomForestModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrossValidator:
    """Handles cross-validation for fraud detection models."""
    
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        model_type: str = 'ensemble'
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            model_type: Type of model to use ('ensemble', 'lightgbm', or 'random_forest')
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_type = model_type
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
    
    def _initialize_model(self, feature_names: List[str], fold: int) -> Any:
        """
        Initialize model for a specific fold.
        
        Args:
            feature_names: List of feature names
            fold: Current fold number
            
        Returns:
            Initialized model instance
        """
        if self.model_type == 'ensemble':
            model = FraudEnsemble(
                name=f"fraud_ensemble_fold_{fold}",
                feature_names=feature_names
            )
        elif self.model_type == 'lightgbm':
            model = LightGBMModel(feature_names=feature_names)
        elif self.model_type == 'random_forest':
            model = RandomForestModel(feature_names=feature_names)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Disable model saving during cross-validation
        model.save_model_and_metrics = lambda *args, **kwargs: None
        
        return model
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary containing cross-validation results
        """
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Store metrics for each fold
        fold_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y), 1):
            logger.info(f"\nTraining fold {fold}/{self.n_splits}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Initialize and train model
            model = self._initialize_model(feature_names, fold)
            model.fit(X_train_fold.values, y_train_fold.values)
            
            # Make predictions
            y_pred_fold = model.predict(X_val_fold.values)
            y_pred_proba_fold = model.predict_proba(X_val_fold.values)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val_fold.values, y_pred_fold, y_pred_proba_fold)
            
            # Store metrics
            for metric_name, value in metrics.items():
                fold_metrics[metric_name].append(value)
            
            # Log fold results
            logger.info(f"Fold {fold} metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
        
        # Calculate mean and std of metrics
        cv_results = {}
        for metric_name, values in fold_metrics.items():
            cv_results[f'mean_{metric_name}'] = np.mean(values)
            cv_results[f'std_{metric_name}'] = np.std(values)
        
        # Log final results
        logger.info("\nCross-validation results:")
        logger.info(f"Mean Precision: {cv_results['mean_precision']:.4f} (±{cv_results['std_precision']:.4f})")
        logger.info(f"Mean Recall: {cv_results['mean_recall']:.4f} (±{cv_results['std_recall']:.4f})")
        logger.info(f"Mean F1 Score: {cv_results['mean_f1']:.4f} (±{cv_results['std_f1']:.4f})")
        logger.info(f"Mean ROC AUC: {cv_results['mean_roc_auc']:.4f} (±{cv_results['std_roc_auc']:.4f})")
        
        return cv_results 