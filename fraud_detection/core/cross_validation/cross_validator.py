"""
Cross-validation functionality for fraud detection models.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from fraud_detection.models.core.ensemble import FraudEnsemble, LightGBMModel, RandomForestModel

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
            return FraudEnsemble(
                name=f"fraud_ensemble_fold_{fold}",
                feature_names=feature_names
            )
        elif self.model_type == 'lightgbm':
            return LightGBMModel(feature_names=feature_names)
        elif self.model_type == 'random_forest':
            return RandomForestModel(feature_names=feature_names)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def _calculate_cost_savings(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transaction_amounts: pd.DataFrame,
        investigation_cost: float = 50.0
    ) -> Dict[str, float]:
        """
        Calculate cost savings metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            transaction_amounts: DataFrame with 'amount' and 'is_synthetic' columns
            investigation_cost: Cost to investigate each flagged transaction
            
        Returns:
            Dictionary of cost metrics
        """
        # Calculate confusion matrix components
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        # Get real transaction amounts (exclude synthetic)
        real_mask = ~transaction_amounts['is_synthetic'].values
        real_amounts = transaction_amounts['amount'].values[real_mask]
        real_tp_mask = tp_mask[real_mask]
        real_fn_mask = fn_mask[real_mask]
        
        # Calculate costs and savings
        money_saved = real_amounts[real_tp_mask].sum()
        investigation_cost_total = len(fp_mask) * investigation_cost
        money_lost = real_amounts[real_fn_mask].sum()
        net_savings = money_saved - investigation_cost_total - money_lost
        
        # Calculate ROI
        total_cost = investigation_cost_total + money_lost
        roi = (net_savings / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'money_saved': money_saved,
            'investigation_cost': investigation_cost_total,
            'money_lost': money_lost,
            'net_savings': net_savings,
            'roi': roi
        }
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        transaction_amounts: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            transaction_amounts: DataFrame with transaction amounts
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
            'roc_auc': [],
            'money_saved': [],
            'investigation_cost': [],
            'money_lost': [],
            'net_savings': [],
            'roi': []
        }
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y), 1):
            logger.info(f"\nTraining fold {fold}/{self.n_splits}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            val_amounts_fold = transaction_amounts.iloc[val_idx]
            
            # Initialize and train model
            model = self._initialize_model(feature_names, fold)
            model.fit(X_train_fold.values, y_train_fold.values)
            
            # Make predictions
            y_pred_fold = model.predict(X_val_fold.values)
            y_pred_proba_fold = model.predict_proba(X_val_fold.values)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val_fold, y_pred_fold, y_pred_proba_fold)
            cost_metrics = self._calculate_cost_savings(
                y_val_fold,
                y_pred_fold,
                val_amounts_fold
            )
            
            # Store metrics
            for metric_name, value in metrics.items():
                fold_metrics[metric_name].append(value)
            for metric_name, value in cost_metrics.items():
                fold_metrics[metric_name].append(value)
            
            # Log fold results
            logger.info(f"Fold {fold} metrics:")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Net Savings: ${cost_metrics['net_savings']:,.2f}")
            logger.info(f"ROI: {cost_metrics['roi']:.2f}%")
        
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
        logger.info(f"Mean Net Savings: ${cv_results['mean_net_savings']:,.2f} (±${cv_results['std_net_savings']:,.2f})")
        logger.info(f"Mean ROI: {cv_results['mean_roi']:.2f}% (±{cv_results['std_roi']:.2f}%)")
        
        return cv_results 