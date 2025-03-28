"""
Script to run cross-validation for fraud detection models.
"""
import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np

from fraud_detection.core.cross_validation import CrossValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_cv_results_dir():
    """Create and return the directory for cross-validation results."""
    # Get current version from timestamp
    version = datetime.now().strftime("%Y.%m.%d")
    
    # Create directory structure
    base_dir = Path("models/cross_validation")
    version_dir = base_dir / f"v{version}"
    results_dir = version_dir / "results"
    
    # Create directories if they don't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir

def load_processed_data():
    """Load processed data from the processed directory."""
    processed_dir = Path("data/processed")
    
    # Load training data
    train_data = pd.read_csv(processed_dir / "train_data.csv")
    
    # Separate features and target
    X_train = train_data.drop("isFraud", axis=1)
    y_train = train_data["isFraud"]
    
    return X_train, y_train

async def main():
    """Run cross-validation for different model types."""
    try:
        # Load processed data
        logger.info("Loading processed data")
        X_train, y_train = load_processed_data()
        
        # Model types to evaluate
        model_types = ['ensemble', 'lightgbm', 'random_forest']
        
        # Store results for all models
        all_results = {}
        
        # Run cross-validation for each model type
        for model_type in model_types:
            try:
                logger.info(f"\nRunning cross-validation for {model_type}")
                cv = CrossValidator(model_type=model_type)
                results = cv.cross_validate(X_train, y_train)
                all_results[model_type] = results
                
                # Log results for this model
                logger.info(f"\n{model_type.upper()} Cross-validation Results:")
                logger.info(f"Mean Precision: {results['mean_precision']:.4f} (±{results['std_precision']:.4f})")
                logger.info(f"Mean Recall: {results['mean_recall']:.4f} (±{results['std_recall']:.4f})")
                logger.info(f"Mean F1 Score: {results['mean_f1']:.4f} (±{results['std_f1']:.4f})")
                logger.info(f"Mean ROC AUC: {results['mean_roc_auc']:.4f} (±{results['std_roc_auc']:.4f})")
                
            except Exception as e:
                logger.error(f"Error during cross-validation for {model_type}: {str(e)}")
                continue
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = get_cv_results_dir()
        results_file = results_dir / f"cv_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"\nCross-validation results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 