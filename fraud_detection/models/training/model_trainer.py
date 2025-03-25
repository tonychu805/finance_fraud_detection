"""
Model training module for fraud detection.

This module handles model training, evaluation, and logging of results to Supabase.
"""

import json
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from fraud_detection.database.supabase_client import SupabaseClient
from fraud_detection.models.ml_tracker import MLTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and evaluation for fraud detection."""
    
    def __init__(
        self,
        config: Dict,
        supabase_client: Optional[SupabaseClient] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            config: Model configuration dictionary
            supabase_client: Optional Supabase client for logging results
        """
        self.config = config
        self.supabase_client = supabase_client
        self.ml_tracker = MLTracker(supabase_client) if supabase_client else None
        
    async def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[lgb.LGBMClassifier, Dict]:
        """
        Train a LightGBM model for fraud detection.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        logger.info("Starting model training")
        
        # Initialize LightGBM model with configuration
        model = lgb.LGBMClassifier(
            **self.config.get("model_params", {})
        )
        
        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Train the model
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(model, X_train, y_train)
        logger.info(f"Training metrics: {train_metrics}")
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            val_metrics = self._calculate_metrics(model, X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Log results if ML tracking is enabled
            if self.ml_tracker:
                version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Log training results
                await self.ml_tracker.log_training_result(
                    version=version,
                    algorithm="lightgbm",
                    metrics=train_metrics,
                    params=self.config["model_params"],
                    importance=self._get_feature_importance(model)
                )
                
                # Log validation results
                await self.ml_tracker.log_evaluation_result(
                    version=version,
                    dataset_type="validation",
                    metrics=val_metrics
                )
        
        return model, train_metrics
    
    def _calculate_metrics(
        self,
        model: lgb.LGBMClassifier,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred)),
            "auc": float(roc_auc_score(y, y_pred_proba))
        }
    
    def _get_feature_importance(
        self,
        model: lgb.LGBMClassifier
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = model.feature_importances_
        feature_names = (
            model.feature_name_ 
            if hasattr(model, "feature_name_") 
            else [f"feature_{i}" for i in range(len(importance_scores))]
        )
        
        return dict(zip(feature_names, importance_scores.astype(float))) 