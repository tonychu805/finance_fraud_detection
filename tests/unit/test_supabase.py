"""
Test script to verify Supabase connection and create ML tracking tables.
"""

import os
import asyncio
from database.supabase_client import SupabaseClient
from ml_tracker import MLTracker

async def test_connection():
    """Test Supabase connection and create tables."""
    try:
        # Initialize Supabase client
        supabase_client = SupabaseClient()
        ml_tracker = MLTracker(supabase_client)
        
        # Test version logging
        test_version = "test_v1"
        test_metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1": 0.90,
            "auc": 0.97
        }
        test_params = {
            "learning_rate": 0.01,
            "num_leaves": 31,
            "feature_fraction": 0.8
        }
        test_importance = {
            "amount": 0.25,
            "oldbalanceOrg": 0.20,
            "newbalanceOrig": 0.18
        }
        
        # Log test model version
        await ml_tracker.log_training_result(
            version=test_version,
            algorithm="test_lightgbm",
            metrics=test_metrics,
            params=test_params,
            importance=test_importance
        )
        
        print("✅ Successfully logged test model version")
        
        # Retrieve and verify the logged model
        best_model = await ml_tracker.get_best_model()
        if best_model:
            print("\n📊 Retrieved best model:")
            print(f"Version: {best_model['version']}")
            print(f"Algorithm: {best_model['algorithm_type']}")
            print(f"Metrics: {best_model['metrics']}")
        
        print("\n✨ Supabase connection and ML tracking system are working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_connection()) 