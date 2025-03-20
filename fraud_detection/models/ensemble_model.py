"""
Ensemble model for fraud detection combining LightGBM and Random Forest models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

class EnsembleFraudModel:
    """Ensemble model combining LightGBM and Random Forest for fraud detection."""
    
    def __init__(
        self, 
        version: str = "v1.0.0",
        use_lightgbm: bool = True,
        use_randomforest: bool = True
    ):
        """Initialize the ensemble model.
        
        Args:
            version: Model version identifier
            use_lightgbm: Whether to use LightGBM in the ensemble
            use_randomforest: Whether to use Random Forest in the ensemble
        """
        self.version = version
        self.use_lightgbm = use_lightgbm
        self.use_randomforest = use_randomforest
        self.lightgbm_model = None
        self.random_forest_model = None
        self.ensemble_model = None
        self.metadata = {
            "version": version,
            "algorithm": self._get_algorithm_name(),
            "performance_metrics": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
            },
            "last_trained": datetime.now().isoformat(),
            "feature_importance": {},
        }
    
    def _get_algorithm_name(self) -> str:
        """Get the name of the algorithm based on which models are used."""
        if self.use_lightgbm and self.use_randomforest:
            return "Ensemble (LightGBM + RandomForest)"
        elif self.use_lightgbm:
            return "LightGBM"
        elif self.use_randomforest:
            return "RandomForest"
        else:
            raise ValueError("At least one model must be enabled")
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        lightgbm_params: Optional[Dict] = None,
        randomforest_params: Optional[Dict] = None
    ) -> None:
        """Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            lightgbm_params: Optional parameters for LightGBM
            randomforest_params: Optional parameters for Random Forest
        """
        if not (self.use_lightgbm or self.use_randomforest):
            raise ValueError("At least one model must be enabled")
            
        # Train LightGBM if enabled
        if self.use_lightgbm:
            lgb_params = lightgbm_params or {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 7,
                'num_leaves': 31,
                'random_state': 42
            }
            self.lightgbm_model = LGBMClassifier(**lgb_params)
            
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            self.lightgbm_model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                eval_metric="auc",
                early_stopping_rounds=50 if eval_set else None
            )
            logger.info("LightGBM model trained")
        
        # Train Random Forest if enabled
        if self.use_randomforest:
            rf_params = randomforest_params or {
                'n_estimators': 100,
                'max_depth': 7,
                'random_state': 42
            }
            self.random_forest_model = RandomForestClassifier(**rf_params)
            self.random_forest_model.fit(X_train, y_train)
            logger.info("Random Forest model trained")
        
        # For backward compatibility, use LightGBM as the main model if available
        self.ensemble_model = self.lightgbm_model if self.use_lightgbm else self.random_forest_model
        
        # Update metadata with performance metrics
        if X_val is not None and y_val is not None:
            metrics = self._calculate_metrics(X_val, y_val)
        else:
            metrics = self._calculate_metrics(X_train, y_train)
            
        self.metadata["performance_metrics"] = metrics
        self.metadata["last_trained"] = datetime.now().isoformat()
        self.metadata["feature_importance"] = self._get_feature_importance()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions (0 or 1)
        """
        if not (self.use_lightgbm or self.use_randomforest):
            raise ValueError("At least one model must be enabled")
            
        predictions = []
        weights = []
        
        # Get LightGBM predictions if available
        if self.use_lightgbm and self.lightgbm_model is not None:
            lgb_pred = self.lightgbm_model.predict_proba(X)[:, 1]
            predictions.append(lgb_pred)
            weights.append(1.0)
            
        # Get Random Forest predictions if available
        if self.use_randomforest and self.random_forest_model is not None:
            rf_pred = self.random_forest_model.predict_proba(X)[:, 1]
            predictions.append(rf_pred)
            weights.append(1.0)
            
        # Average predictions (weighted)
        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(p * w for p, w in zip(predictions, weights))
        return (ensemble_pred > 0.5).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from the ensemble model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of fraud probabilities
        """
        if not (self.use_lightgbm or self.use_randomforest):
            raise ValueError("At least one model must be enabled")
            
        predictions = []
        weights = []
        
        # Get LightGBM predictions if available
        if self.use_lightgbm and self.lightgbm_model is not None:
            lgb_pred = self.lightgbm_model.predict_proba(X)[:, 1]
            predictions.append(lgb_pred)
            weights.append(1.0)
            
        # Get Random Forest predictions if available
        if self.use_randomforest and self.random_forest_model is not None:
            rf_pred = self.random_forest_model.predict_proba(X)[:, 1]
            predictions.append(rf_pred)
            weights.append(1.0)
            
        # Average predictions (weighted)
        weights = np.array(weights) / sum(weights)
        return sum(p * w for p, w in zip(predictions, weights))
        
    def save(self, path: Optional[Path] = None) -> None:
        """Save the ensemble model to disk.
        
        Args:
            path: Optional path to save the model to
        """
        if path is None:
            path = Path("models/fraud_ensemble_model.joblib")
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model data
        model_data = {
            "lightgbm_model": self.lightgbm_model,
            "random_forest_model": self.random_forest_model,
            "ensemble_model": self.ensemble_model,
            "metadata": self.metadata,
            "version": self.version,
            "use_lightgbm": self.use_lightgbm,
            "use_randomforest": self.use_randomforest
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
        # Save metadata separately
        metadata_path = path.parent / "ensemble_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
            
    def load(self, path: Optional[Path] = None) -> bool:
        """Load the ensemble model from disk.
        
        Args:
            path: Optional path to load the model from
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        if path is None:
            path = Path("models/fraud_ensemble_model.joblib")
            
        if not path.exists():
            logger.warning(f"Model file not found at {path}")
            return False
            
        try:
            model_data = joblib.load(path)
            
            self.lightgbm_model = model_data.get("lightgbm_model")
            self.random_forest_model = model_data.get("random_forest_model")
            self.ensemble_model = model_data.get("ensemble_model")
            self.metadata = model_data.get("metadata", self.metadata)
            self.version = model_data.get("version", self.version)
            self.use_lightgbm = model_data.get("use_lightgbm", True)
            self.use_randomforest = model_data.get("use_randomforest", True)
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dict of performance metrics
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred)),
            "auc": float(roc_auc_score(y, y_pred_proba))
        }
        
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from both models.
        
        Returns:
            Dict of feature importance scores
        """
        importance_dict = {}
        
        if self.use_lightgbm and self.lightgbm_model is not None:
            lgb_importance = self.lightgbm_model.feature_importances_
            importance_dict["lightgbm"] = {
                f"feature_{i}": float(imp) 
                for i, imp in enumerate(lgb_importance)
            }
            
        if self.use_randomforest and self.random_forest_model is not None:
            rf_importance = self.random_forest_model.feature_importances_
            importance_dict["randomforest"] = {
                f"feature_{i}": float(imp) 
                for i, imp in enumerate(rf_importance)
            }
            
        return importance_dict 