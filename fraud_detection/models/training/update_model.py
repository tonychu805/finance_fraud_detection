#!/usr/bin/env python
"""
Script to retrain the model on the dataset and update the model used by the API.

This script is intended to be run periodically to update the fraud detection model
with the latest data and ensure the API is using the most recent model.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import joblib

from fraud_detection.data.processor import DataProcessor
from fraud_detection.models.ensemble import EnsembleFraudModel
from fraud_detection.models.manager import ModelManager
from fraud_detection.database.supabase_client import SupabaseClient
from fraud_detection.models.ml_tracker import MLTracker
from fraud_detection.models.training.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/model_update.log"),
    ],
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Update fraud detection model")
    
    parser.add_argument(
        "--sample", 
        type=int, 
        default=None,
        help="Number of samples to use for training (for testing/development)"
    )
    
    parser.add_argument(
        "--use-neural-network", 
        action="store_true",
        help="Include neural network in the ensemble model"
    )
    
    parser.add_argument(
        "--bypass-db", 
        action="store_true",
        help="Bypass database and use raw data directly"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force model retraining even if no new data"
    )
    
    return parser.parse_args()


async def update_model(config_path: str, bypass_db: bool = False):
    """
    Update the fraud detection model with new data and track the results.
    
    Args:
        config_path: Path to the model configuration file
        bypass_db: Whether to bypass database operations
    """
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
            
        # Initialize components
        data_processor = DataProcessor(config['data_config'])
        supabase_client = None if bypass_db else SupabaseClient()
        ml_tracker = None if bypass_db else MLTracker(supabase_client)
        model_trainer = ModelTrainer(config['model_config'], supabase_client)
        
        # Load and process data
        logger.info("Loading and processing data...")
        X_train, X_val, y_train, y_val = data_processor.prepare_data()
        
        # Train model
        logger.info("Training model...")
        model = await model_trainer.train(X_train, y_train, X_val, y_val)
        
        # If tracking is enabled, get the best model version for comparison
        if ml_tracker:
            best_model = await ml_tracker.get_best_model()
            if best_model:
                logger.info(f"Comparing with best model version: {best_model['version']}")
                current_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                comparison = await ml_tracker.compare_models(current_version, best_model['version'])
                logger.info("Model comparison results:")
                logger.info(json.dumps(comparison, indent=2))
                
                # Analyze feature importance trends
                trends = await ml_tracker.get_feature_trends()
                logger.info("Feature importance trends:")
                logger.info(json.dumps(trends, indent=2))
        
        # Save model
        model_path = Path(config['model_config']['model_path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise


def main():
    """Run the model update process."""
    args = parse_args()
    
    logger.info("Starting model update process")
    
    # Initialize database client
    db_client = None
    if not args.bypass_db:
        logger.info("Initializing database client")
        db_client = SupabaseClient()
    
    # Initialize model manager
    logger.info("Initializing model manager")
    model_manager = ModelManager()
    
    # Check if model needs to be updated
    # In a real implementation, this would check if there is enough new data
    # or if the model performance has degraded
    should_update = args.force
    
    if not args.force and not args.bypass_db:
        # In a production system, we would implement logic to check:
        # 1. If there is a significant amount of new data
        # 2. If the model performance has degraded
        # 3. If a certain time period has elapsed since last training
        
        # For simplicity, we'll update if there are new transactions
        if db_client:
            last_trained = model_manager.metadata.get("last_trained")
            
            # Count new transactions since last training (mock implementation)
            transactions = db_client.get_transactions(limit=5)
            
            # In a real system, we'd filter by timestamp > last_trained
            new_transaction_count = len(transactions)
            
            logger.info(f"Found {new_transaction_count} new transactions since last training")
            should_update = new_transaction_count > 0
    
    if args.bypass_db or should_update:
        logger.info("Model update needed. Starting the training process...")
        
        if args.bypass_db:
            # Use data processor directly with raw dataset
            logger.info("Bypassing database and using raw data directly")
            data_processor = DataProcessor()
            
            X_train, X_val, X_test, y_train, y_val, y_test = data_processor.process_data_pipeline(
                sample_size=args.sample,
                save_processed=True,
            )
            
            # Initialize and train model
            model = EnsembleFraudModel(use_neural_network=args.use_neural_network)
            
            logger.info("Training ensemble model with processed data")
            history = model.fit(
                X_train=X_train.values if hasattr(X_train, 'values') else X_train,
                y_train=y_train.values if hasattr(y_train, 'values') else y_train,
                X_val=X_val.values if hasattr(X_val, 'values') else X_val,
                y_val=y_val.values if hasattr(y_val, 'values') else y_val,
            )
            
            # Save model
            saved_paths = model.save()
            logger.info(f"Model saved to {saved_paths.get('combined')}")
            
            # Test model on test data
            logger.info("Evaluating model on test data")
            test_predictions = model.predict(
                X_test.values if hasattr(X_test, 'values') else X_test
            )
            
            # Calculate and log metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
            accuracy = accuracy_score(y_test_values, test_predictions)
            precision = precision_score(y_test_values, test_predictions)
            recall = recall_score(y_test_values, test_predictions)
            f1 = f1_score(y_test_values, test_predictions)
            
            logger.info(f"Test metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                       f"Recall: {recall:.4f}, F1: {f1:.4f}")
        else:
            # Use model manager to train with database
            logger.info("Training model using database client")
            result = model_manager.train_model(db_client)
            logger.info(f"Model training completed with result: {result}")
        
        # Reload model manager to use the new model
        logger.info("Reloading model manager to use the new model")
        new_model_manager = ModelManager()
        
        # Check if reload was successful
        if new_model_manager.is_model_loaded():
            logger.info("Model reload successful")
        else:
            logger.error("Model reload failed")
    
    else:
        logger.info("No model update needed at this time")
    
    logger.info("Model update process completed")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.exception(f"Error during model update: {e}")
        sys.exit(1) 