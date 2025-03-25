"""
Ensemble model implementation for fraud detection.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fraud_detection.models.base import BaseEnsemble, BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, name: str = "lightgbm", **params):
        super().__init__(name)
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.01,
            'max_depth': 7,
            'n_estimators': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'early_stopping_rounds': 50,
            'verbose': -1,
            'colsample_bytree': 0.8,
            'subsample': 0.8
        }
        default_params.update(params)
        self.model = LGBMClassifier(**default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        self.model.fit(X, y)
        self.metadata['feature_importance'] = dict(zip(
            X.columns,
            self.model.feature_importances_
        ))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest model wrapper."""
    
    def __init__(self, name: str = "random_forest", **params):
        super().__init__(name)
        default_params = {
            'n_estimators': 100,
            'max_depth': 7,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = RandomForestClassifier(**default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        self.model.fit(X, y)
        self.metadata['feature_importance'] = dict(zip(
            X.columns,
            self.model.feature_importances_
        ))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)


class FraudEnsemble(BaseEnsemble):
    """Ensemble model for fraud detection."""
    
    def __init__(
        self,
        name: str = "fraud_ensemble",
        models: Optional[Dict[str, BaseModel]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(name, models)
        self.weights = weights or {}
        
        # Initialize default models if none provided
        if not models:
            self.models = {
                'lightgbm': LightGBMModel(),
                'random_forest': RandomForestModel()
            }
            self.weights = {
                'lightgbm': 0.6,
                'random_forest': 0.4
            }
    
    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        self._normalize_weights()
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all models in the ensemble."""
        for name, model in self.models.items():
            model.fit(X, y)
        
        # Update metadata
        self.metadata.update({
            'feature_importance': self._get_feature_importance(X.columns),
            'model_weights': self.weights.copy()
        })
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions."""
        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            predictions += self.weights[name] * model.predict(X)
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict weighted probabilities."""
        probas = np.zeros((len(X), 2))
        for name, model in self.models.items():
            probas += self.weights[name] * model.predict_proba(X)
        return probas
    
    def _get_feature_importance(self, features: List[str]) -> Dict[str, float]:
        """Calculate weighted feature importance."""
        importance = {feature: 0.0 for feature in features}
        for name, model in self.models.items():
            if 'feature_importance' in model.metadata:
                for feature, imp in model.metadata['feature_importance'].items():
                    importance[feature] += self.weights[name] * imp
        return importance
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the ensemble model."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_prob)
        }
        
        # Update metadata
        self.metadata['performance_metrics'] = metrics
        
        return metrics 