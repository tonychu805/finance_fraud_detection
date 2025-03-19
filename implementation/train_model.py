#!/usr/bin/env python
"""
Script to train the fraud detection model.

This script processes the raw transaction data, trains the ensemble model,
and saves the results.
"""
import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from data_processor import DataProcessor
from models.ensemble_model import EnsembleFraudModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/model_training.log"),
    ],
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    
    parser.add_argument(
        "--sample", 
        type=int, 
        default=None,
        help="Number of samples to use for training (for testing/development)"
    )
    
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing"
    )
    
    parser.add_argument(
        "--val-size", 
        type=float, 
        default=0.1,
        help="Proportion of data to use for validation"
    )
    
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--use-neural-network", 
        action="store_true",
        help="Include neural network in the ensemble model"
    )
    
    parser.add_argument(
        "--lightgbm-params", 
        type=str,
        default=None,
        help="JSON file with LightGBM parameters"
    )
    
    parser.add_argument(
        "--random-forest-params", 
        type=str,
        default=None,
        help="JSON file with Random Forest parameters"
    )
    
    parser.add_argument(
        "--neural-network-params", 
        type=str,
        default=None,
        help="JSON file with Neural Network parameters"
    )
    
    parser.add_argument(
        "--model-version", 
        type=str,
        default=None,
        help="Model version to use (default: auto-generate based on date)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="implementation/models",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--no-save-intermediates", 
        action="store_true",
        help="Do not save intermediate data files"
    )
    
    return parser.parse_args()


def load_params(file_path: Optional[str]) -> Optional[Dict]:
    """Load parameters from a JSON file."""
    if file_path is None:
        return None
    
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading parameters from {file_path}: {e}")
        return None


def generate_model_version() -> str:
    """Generate a model version based on the current date."""
    now = datetime.datetime.now()
    return f"v{now.year}.{now.month}.{now.day}"


def save_training_report(
    training_history: Dict,
    feature_stats: Dict,
    output_dir: str,
    version: str,
):
    """Save a report of the training process."""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "version": version,
        "training_date": datetime.datetime.now().isoformat(),
        "training_history": training_history,
        "feature_stats": feature_stats,
    }
    
    report_path = Path(output_dir) / f"training_report_{version}.json"
    
    with open(report_path, "w") as f:
        json.dump(
            report, 
            f, 
            indent=2, 
            default=lambda x: float(x) if isinstance(x, np.number) else str(x)
        )
    
    logger.info(f"Training report saved to {report_path}")


def main():
    """Run the model training process."""
    args = parse_args()
    
    # Start timing
    start_time = datetime.datetime.now()
    logger.info(f"Starting model training at {start_time}")
    
    # Process data
    logger.info("Initializing data processor")
    data_processor = DataProcessor()
    
    logger.info("Processing data")
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.process_data_pipeline(
        sample_size=args.sample,
        test_size=args.test_size,
        val_size=args.val_size,
        save_processed=not args.no_save_intermediates,
        random_state=args.random_state,
    )
    
    # Generate or use provided model version
    model_version = args.model_version or generate_model_version()
    
    # Load model parameters if provided
    lightgbm_params = load_params(args.lightgbm_params)
    random_forest_params = load_params(args.random_forest_params)
    neural_network_params = load_params(args.neural_network_params)
    
    # Initialize and train model
    logger.info(f"Initializing ensemble model (version: {model_version})")
    model = EnsembleFraudModel(
        model_dir=Path(args.output_dir),
        version=model_version,
        use_neural_network=args.use_neural_network,
    )
    
    logger.info("Training ensemble model")
    training_history = model.fit(
        X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
        y_train=y_train.values if isinstance(y_train, pd.Series) else y_train,
        X_val=X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
        y_val=y_val.values if isinstance(y_val, pd.Series) else y_val,
        lightgbm_params=lightgbm_params,
        random_forest_params=random_forest_params,
        neural_network_params=neural_network_params,
    )
    
    # Save model
    logger.info("Saving trained model")
    saved_paths = model.save()
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set")
    test_data = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    test_labels = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    test_predictions = model.predict(test_data)
    test_probas = model.predict_proba(test_data)
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )
    
    metrics = {
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "precision": float(precision_score(test_labels, test_predictions)),
        "recall": float(recall_score(test_labels, test_predictions)),
        "f1": float(f1_score(test_labels, test_predictions)),
        "auc": float(roc_auc_score(test_labels, test_probas)),
    }
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    
    # Log metrics
    logger.info(f"Test metrics: {metrics}")
    logger.info("\nClassification Report:\n" + classification_report(test_labels, test_predictions))
    
    # Save feature names if available
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        
        # Save feature names to model output directory
        feature_names_path = Path(args.output_dir) / "feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"Feature names saved to {feature_names_path}")
        
        # Update feature importance with actual feature names if available
        if "feature_importance" in model.metadata:
            for model_type, importance in model.metadata["feature_importance"].items():
                named_importance = {}
                for idx, value in importance.items():
                    try:
                        idx_int = int(idx)
                        if idx_int < len(feature_names):
                            named_importance[feature_names[idx_int]] = value
                        else:
                            named_importance[f"feature_{idx}"] = value
                    except (ValueError, IndexError):
                        named_importance[f"feature_{idx}"] = value
                
                model.metadata["feature_importance"][model_type] = named_importance
    
    # Calculate execution time
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    logger.info(f"Model training completed in {execution_time}")
    
    # Save training report
    save_training_report(
        training_history=training_history,
        feature_stats=data_processor.feature_stats,
        output_dir=args.output_dir,
        version=model_version,
    )
    
    # Save test metrics to a separate file
    metrics_path = Path(args.output_dir) / f"test_metrics_{model_version}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "version": model_version,
            "test_metrics": metrics,
            "training_time": str(execution_time),
            "timestamp": end_time.isoformat(),
        }, f, indent=2)
    
    logger.info(f"Test metrics saved to {metrics_path}")
    
    logger.info(f"Model training process completed successfully!")
    
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during model training: {e}")
        sys.exit(1) 