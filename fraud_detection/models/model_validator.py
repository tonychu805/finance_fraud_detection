"""
Model validation module for fraud detection.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)

class ModelValidator:
    """Handles model validation and hyperparameter tuning for fraud detection."""
    
    def __init__(
        self,
        n_splits: int = 3,  # Reduced from 5 to 3 splits
        test_size_fraction: float = 0.15,  # 15% test size
        gap_fraction: float = 0.001  # 0.1% of data as gap
    ):
        """Initialize the model validator.
        
        Args:
            n_splits: Number of splits for time-series cross-validation
            test_size_fraction: Fraction of data to use for testing
            gap_fraction: Fraction of data to use as gap between train and test
        """
        self.n_splits = n_splits
        self.test_size_fraction = test_size_fraction
        self.gap_fraction = gap_fraction
        
        # Custom scoring metrics
        self.scoring = {
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'average_precision': make_scorer(average_precision_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
    
    def get_time_series_splits(self, X: pd.DataFrame) -> TimeSeriesSplit:
        """Create time-series cross-validation splits.
        
        Args:
            X: Feature matrix
            
        Returns:
            TimeSeriesSplit object
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size_fraction)
        gap = int(n_samples * self.gap_fraction)
        
        logger.info(f"Creating TimeSeriesSplit with:")
        logger.info(f"- Total samples: {n_samples}")
        logger.info(f"- Test size: {test_size}")
        logger.info(f"- Gap size: {gap}")
        logger.info(f"- Number of splits: {self.n_splits}")
        
        return TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=gap,
            test_size=test_size
        )
    
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target vector
            
        Returns:
            Average validation score (weighted F1)
        """
        # Define hyperparameter search space
        param_space = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "bootstrap": True,
            "random_state": 42
        }
        
        # Initialize model
        model = RandomForestClassifier(**param_space)
        
        # Perform time-series cross-validation
        cv = self.get_time_series_splits(X)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            logger.info(f"Fold {fold}:")
            logger.info(f"- Train size: {len(X_train)}")
            logger.info(f"- Validation size: {len(X_val)}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and calculate weighted F1 score
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred)
            scores.append(score)
            
            logger.info(f"- Fold {fold} F1 score: {score:.4f}")
        
        mean_score = np.mean(scores)
        logger.info(f"Mean F1 score: {mean_score:.4f}")
        
        return mean_score
    
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        timeout: Optional[int] = 3600  # 1 hour
    ) -> Dict:
        """Tune model hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            
        Returns:
            Dictionary of best parameters
        """
        logger.info("Starting hyperparameter optimization")
        logger.info(f"Dataset size: {len(X)} samples")
        
        # Create study object
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Best trial score: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        return study.best_trial.params
    
    def validate_model(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, List[float]]:
        """Validate model using time-series cross-validation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of validation metrics
        """
        cv = self.get_time_series_splits(X)
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'average_precision': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            logger.info(f"\nFold {fold}:")
            logger.info(f"Train size: {len(X_train)}")
            logger.info(f"Validation size: {len(X_val)}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            metrics['average_precision'].append(average_precision_score(y_val, y_pred_proba))
            metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            
            logger.info("\nFold metrics:")
            for metric, values in metrics.items():
                logger.info(f"{metric}: {values[-1]:.4f}")
        
        return metrics
    
    def get_feature_importance(
        self,
        model: RandomForestClassifier,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Get feature importance analysis.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'std': std
        })
        
        return feature_importance.sort_values('importance', ascending=False) 