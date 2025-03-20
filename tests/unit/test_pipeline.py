"""
Test script to verify the complete fraud detection pipeline.
"""

import asyncio
import logging
import os
from pathlib import Path

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .ml_tracker import MLTracker
from .database.supabase_client import SupabaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pipeline():
    """Test the complete fraud detection pipeline."""
    try:
        # Step 1: Initialize components
        logger.info("Initializing components...")
        data_processor = DataProcessor()
        supabase_client = SupabaseClient()
        ml_tracker = MLTracker(supabase_client)
        
        # Step 2: Process a small sample of data
        logger.info("Processing sample data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.process_data_pipeline(
            sample_size=10000,  # Use a small sample for testing
            save_processed=True
        )
        logger.info(f"Processed data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        # Step 3: Initialize and train model
        logger.info("Training model...")
        config = {
            "model_params": {
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.01,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_estimators": 100,  # Reduced for testing
                "random_state": 42
            }
        }
        
        trainer = ModelTrainer(config, supabase_client)
        model, metrics = await trainer.train_model(
            X_train, y_train,
            X_val, y_val
        )
        
        logger.info(f"Training metrics: {metrics}")
        
        # Step 4: Test ML tracking
        logger.info("Testing ML tracking...")
        best_model = await ml_tracker.get_best_model()
        if best_model:
            logger.info(f"Retrieved best model version: {best_model['version']}")
            
        # Step 5: Get feature importance trends
        trends = await ml_tracker.get_feature_trends(top_n=5)
        logger.info(f"Top 5 feature importance trends: {trends}")
        
        logger.info("✅ Pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in pipeline test: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_pipeline()) 