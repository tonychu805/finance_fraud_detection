"""
Main script to run the complete fraud detection training pipeline.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from fraud_detection.core.data_processor import DataProcessor
from fraud_detection.models.ensemble_model import EnsembleFraudModel
from fraud_detection.models.model_validator import ModelValidator
from fraud_detection.database.supabase_client import SupabaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--model", 
        choices=["lightgbm", "randomforest", "both"],
        default="both",
        help="Which model to train (default: both)"
    )
    parser.add_argument(
        "--model-params",
        type=str,
        help="Path to JSON file containing model parameters"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of hyperparameter optimization trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for hyperparameter optimization in seconds"
    )
    return parser.parse_args()

def load_config(config_path: str = "config/model_config.json") -> dict:
    """Load model configuration."""
    with open(config_path) as f:
        return json.load(f)

def load_params(file_path: str) -> dict:
    """Load model parameters from JSON file."""
    if not file_path:
        return {}
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load parameters from {file_path}: {e}")
        return {}

async def main():
    """Run the complete training pipeline."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load configuration
        config = load_config()
        logger.info("Loaded model configuration")
        
        # Initialize Supabase client
        supabase_client = SupabaseClient()
        logger.info("Initialized Supabase client")
        
        # Initialize data processor
        processor = DataProcessor()
        logger.info("Initialized data processor")
        
        # Process data
        logger.info("Starting data processing pipeline...")
        X_train, X_val, X_test, y_train, y_val, y_test = processor.process_data_pipeline(
            save_processed=True
        )
        logger.info(f"Data processed - Training set shape: {X_train.shape}")
        
        # Initialize model validator
        validator = ModelValidator(
            n_splits=3,
            test_size_fraction=0.15,
            gap_fraction=0.001
        )
        
        # Initialize ensemble model with specified configuration
        use_lightgbm = args.model in ["lightgbm", "both"]
        use_randomforest = args.model in ["randomforest", "both"]
        
        if args.tune and args.model == "randomforest":
            logger.info("Starting hyperparameter tuning for Random Forest")
            # Combine train and validation sets for tuning
            X_tune = pd.concat([X_train, X_val])
            y_tune = pd.concat([y_train, y_val])
            
            # Tune hyperparameters
            best_params = validator.tune_hyperparameters(
                X_tune,
                y_tune,
                n_trials=args.n_trials,
                timeout=args.timeout
            )
            
            # Save best parameters
            best_params_path = Path("config/random_forest_best_params.json")
            with open(best_params_path, "w") as f:
                json.dump(best_params, f, indent=4)
            logger.info(f"Saved best parameters to {best_params_path}")
            
            # Use best parameters for training
            model_params = best_params
        else:
            # Load parameters from file
            model_params = load_params(args.model_params)
        
        # Initialize and train model
        model = EnsembleFraudModel(
            version="v1.0",
            use_lightgbm=use_lightgbm,
            use_randomforest=use_randomforest
        )
        
        logger.info(f"Training model(s): {args.model}")
        model.fit(
            X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            y_train=y_train.values if isinstance(y_train, pd.Series) else y_train,
            X_val=X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
            y_val=y_val.values if isinstance(y_val, pd.Series) else y_val
        )
        
        # Validate model
        if args.model == "randomforest":
            logger.info("Performing time-series cross-validation")
            validation_metrics = validator.validate_model(
                model.random_forest_model,
                pd.concat([X_train, X_val]),
                pd.concat([y_train, y_val])
            )
            
            # Log validation results
            logger.info("\nValidation Results:")
            for metric, values in validation_metrics.items():
                mean_score = np.mean(values)
                std_score = np.std(values)
                logger.info(f"{metric}: {mean_score:.4f} (+/- {std_score:.4f})")
            
            # Get feature importance
            feature_importance = validator.get_feature_importance(
                model.random_forest_model,
                processor.feature_stats["feature_names"]
            )
            logger.info("\nTop 10 Important Features:")
            logger.info(feature_importance.head(10))
        
        # Save model
        model_path = Path("models/fraud_ensemble_model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Final evaluation on test set
        test_predictions = model.predict(
            X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        )
        test_probas = model.predict_proba(
            X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        )
        
        # Calculate and log final metrics
        from sklearn.metrics import classification_report, average_precision_score
        logger.info("\nTest Set Performance:")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, test_predictions))
        
        average_precision = average_precision_score(y_test, test_probas)
        logger.info(f"\nAverage Precision-Recall Score: {average_precision:.4f}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 