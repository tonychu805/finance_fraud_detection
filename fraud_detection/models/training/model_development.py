"""
Model development module for fraud detection.

This module implements the model development phase as outlined in the documentation,
including model selection, training, and evaluation.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from fraud_detection.models.config import ModelConfig
from fraud_detection.models.registry import ModelRegistry
from fraud_detection.models.evaluation.metrics import MetricsCalculator
from fraud_detection.utils.result_storage import ResultStorage
from fraud_detection.database.supabase_client import SupabaseClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudModelDevelopment:
    """
    Handles model development for fraud detection.
    """
    
    def __init__(self, config: ModelConfig, supabase_client: Optional[SupabaseClient] = None):
        """
        Initialize the model development module.
        
        Args:
            config: Configuration object containing all settings
            supabase_client: Optional Supabase client for database storage
        """
        self.config = config
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.result_storage = ResultStorage(config.output_dir, supabase_client)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load processed data splits."""
        logger.info("Loading processed data splits")
        
        # Load train data
        train_data = pd.read_csv(self.config.data_dir / "train_data.csv")
        X_train = train_data.drop(columns=["is_fraud"])
        y_train = train_data["is_fraud"]
        
        # Load validation data
        val_data = pd.read_csv(self.config.data_dir / "val_data.csv")
        X_val = val_data.drop(columns=["is_fraud"])
        y_val = val_data["is_fraud"]
        
        # Load test data
        test_data = pd.read_csv(self.config.data_dir / "test_data.csv")
        X_test = test_data.drop(columns=["is_fraud"])
        y_test = test_data["is_fraud"]
        
        logger.info(f"Loaded {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE and RandomUnderSampler.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Balanced feature matrix and target vector
        """
        logger.info("Handling class imbalance")
        
        # Define the resampling strategy
        over = SMOTE(sampling_strategy=self.config.smote_strategy, random_state=self.config.random_state)
        under = RandomUnderSampler(sampling_strategy=self.config.under_sampler_strategy, random_state=self.config.random_state)
        
        # Create the pipeline
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        
        # Fit and transform the data
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        
        logger.info(f"Original distribution: {y.value_counts(normalize=True).to_dict()}")
        logger.info(f"Resampled distribution: {pd.Series(y_resampled).value_counts(normalize=True).to_dict()}")
        
        return X_resampled, y_resampled
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train baseline models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training baseline models")
        
        # Get baseline models from registry
        baseline_models = ModelRegistry.get_baseline_models(self.config.random_state)
        
        # Train each model
        for name, model in baseline_models.items():
            logger.info(f"Training {name}")
            # Set class weights if needed
            model = ModelRegistry.set_class_weights(model, y_train)
            model.fit(X_train, y_train)
            self.models[name] = model
            
        return self.models
    
    def evaluate_models(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Evaluate all trained models on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating models")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            # Get predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(y_val, y_pred, y_pred_proba)
            evaluation_results[name] = metrics
            
            # Save metrics plots
            self.result_storage.save_metrics_plots(y_val, y_pred_proba, name, model_version=name)
            
            # Log results
            logger.info(f"\nResults for {name}:")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.3f}")
            logger.info(f"Average Precision: {metrics['average_precision']:.3f}")
            
        return evaluation_results
    
    def tune_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Perform hyperparameter tuning for the best performing model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Tuning best model")
        
        # Get base model for tuning
        base_model = ModelRegistry.get_tuning_model(self.config.random_state)
        base_model = ModelRegistry.set_class_weights(base_model, y_train)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.config.param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best ROC AUC: {float(grid_search.best_score_):.3f}")
        
        # Store feature importance
        self.feature_importance = {
            feature: float(importance)
            for feature, importance in zip(X_train.columns, self.best_model.feature_importances_)
        }
        
        # Save feature importance plot
        self.result_storage.save_feature_importance(
            self.feature_importance,
            filename=self.config.feature_importance_plot_name,
            model_version="best_model"
        )
    
    def run_full_pipeline(self) -> Dict:
        """
        Run the complete model development pipeline.
        
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting model development pipeline")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Train baseline models
        self.train_baseline_models(X_train_balanced, y_train_balanced)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_val, y_val)
        
        # Tune best model
        self.tune_best_model(X_train_balanced, y_train_balanced)
        
        # Final evaluation on test set
        final_results = self.evaluate_models(X_test, y_test)
        
        # Save results
        results = {
            "baseline_evaluation": evaluation_results,
            "final_evaluation": final_results,
            "feature_importance": self.feature_importance,
            "algorithm": "Ensemble",
            "performance_metrics": final_results.get("best_model", {})
        }
        
        self.result_storage.save_results(
            results, 
            filename=self.config.results_file_name,
            model_version="best_model"
        )
        
        logger.info("Model development pipeline completed successfully")
        return results

if __name__ == "__main__":
    # Example usage
    config = ModelConfig()
    model_dev = FraudModelDevelopment(config)
    results = model_dev.run_full_pipeline()
    
    # Print key findings
    print("\nKey Findings:")
    print("1. Best performing model:")
    best_model = max(results["baseline_evaluation"].items(), key=lambda x: x[1]["roc_auc"])
    print(f"   - {best_model[0]}: ROC AUC = {best_model[1]['roc_auc']:.3f}")
    
    print("\n2. Top 5 most important features:")
    importance_df = pd.DataFrame({
        'feature': list(results["feature_importance"].keys()),
        'importance': list(results["feature_importance"].values())
    }).sort_values('importance', ascending=False)
    for _, row in importance_df.head().iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}") 