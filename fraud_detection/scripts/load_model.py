"""
Script to load and examine saved fraud detection models.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Load and examine the saved model."""
    # Try to load both models
    model_paths = [
        Path("models/fraud_ensemble_model.joblib"),
        Path("models/lightgbm_calibrated_model.joblib")
    ]
    
    for model_path in model_paths:
        try:
            logger.info(f"Attempting to load model from {model_path}")
            model_data = joblib.load(model_path)
            
            logger.info("\nModel Information:")
            logger.info("=" * 50)
            
            # Print model metadata
            if isinstance(model_data, dict):
                for key, value in model_data.items():
                    if key == 'model':
                        logger.info(f"\nModel Type: {type(value).__name__}")
                        
                        # Try to get feature importances if available
                        if hasattr(value, 'feature_importances_'):
                            importances = value.feature_importances_
                            if 'feature_names' in model_data:
                                features = model_data['feature_names']
                            else:
                                features = [f"feature_{i}" for i in range(len(importances))]
                                
                            importance_df = pd.DataFrame({
                                'feature': features,
                                'importance': importances
                            }).sort_values('importance', ascending=False)
                            
                            logger.info("\nTop 10 Feature Importances:")
                            logger.info(importance_df.head(10))
                        
                        # Try to get model parameters
                        if hasattr(value, 'get_params'):
                            logger.info("\nModel Parameters:")
                            params = value.get_params()
                            for param, val in params.items():
                                logger.info(f"  {param}: {val}")
                    else:
                        logger.info(f"\n{key}: {value}")
            else:
                logger.info(f"Model Type: {type(model_data).__name__}")
                
                # Try to get feature importances if available
                if hasattr(model_data, 'feature_importances_'):
                    importances = model_data.feature_importances_
                    features = [f"feature_{i}" for i in range(len(importances))]
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    logger.info("\nTop 10 Feature Importances:")
                    logger.info(importance_df.head(10))
                
                # Try to get model parameters
                if hasattr(model_data, 'get_params'):
                    logger.info("\nModel Parameters:")
                    params = model_data.get_params()
                    for param, val in params.items():
                        logger.info(f"  {param}: {val}")
            
            logger.info("\n" + "=" * 50)
            
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")

if __name__ == "__main__":
    main() 