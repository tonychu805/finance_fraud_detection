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

def load_processed_data():
    """Load processed data from the processed directory."""
    processed_dir = Path("data/processed")
    
    # Load training data
    train_data = pd.read_csv(processed_dir / "train_data.csv")
    train_amounts = pd.read_csv(processed_dir / "train_amounts.csv")
    
    # Separate features and target
    X_train = train_data.drop("isFraud", axis=1)
    y_train = train_data["isFraud"]
    
    return X_train, y_train, train_amounts

async def main():
    """Run cross-validation for different model types."""
    try:
        # Load processed data
        logger.info("Loading processed data")
        X_train, y_train, train_amounts = load_processed_data()
        
        # Model types to evaluate
        model_types = ['ensemble', 'lightgbm', 'random_forest']
        
        # Store results for all models
        all_results = {}
        
        # Run cross-validation for each model type
        for model_type in model_types:
            logger.info(f"\nRunning cross-validation for {model_type} model")
            
            # Initialize cross-validator
            cv = CrossValidator(
                n_splits=5,
                random_state=42,
                model_type=model_type
            )
            
            # Perform cross-validation
            cv_results = cv.cross_validate(
                X_train,
                y_train,
                train_amounts,
                feature_names=X_train.columns.tolist()
            )
            
            # Store results
            all_results[model_type] = cv_results
        
        # Save results
        version = datetime.now().strftime("%Y.%m.%d")
        results_dir = Path(f"models/v{version}_cv_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "cross_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"\nCross-validation results saved to {results_file}")
        
        # Print summary
        logger.info("\nCross-validation Summary:")
        for model_type, results in all_results.items():
            logger.info(f"\n{model_type.upper()} Model:")
            logger.info(f"Mean F1 Score: {results['mean_f1']:.4f} (±{results['std_f1']:.4f})")
            logger.info(f"Mean ROC AUC: {results['mean_roc_auc']:.4f} (±{results['std_roc_auc']:.4f})")
            logger.info(f"Mean Net Savings: ${results['mean_net_savings']:,.2f} (±${results['std_net_savings']:,.2f})")
            logger.info(f"Mean ROI: {results['mean_roi']:.2f}% (±{results['std_roi']:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 