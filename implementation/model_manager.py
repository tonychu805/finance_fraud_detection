"""
Model manager for training, loading, and using fraud detection models.
"""
import datetime
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processor import DataProcessor
from models.ensemble_model import EnsembleFraudModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_PATH = Path("implementation/models/fraud_ensemble_model.joblib")
MODEL_METADATA_PATH = Path("implementation/models/ensemble_metadata.json")
FEATURE_COLUMNS = [
    "amount",
    "merchant_category_encoded",
    "transaction_type_encoded",
    "card_present",
    "country_encoded",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

class ModelManager:
    """Manager for fraud detection models."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the model manager.
        
        Args:
            model_path: Path to the model file. If None, uses the default path.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.ensemble_model = None
        self.data_processor = DataProcessor()
        self.metadata = self._load_metadata()
        self._load_model()
    
    def _load_metadata(self) -> Dict:
        """Load model metadata from file.
        
        Returns:
            Dict: Model metadata
        """
        if MODEL_METADATA_PATH.exists():
            try:
                with open(MODEL_METADATA_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model metadata: {e}")
        
        # Default metadata if file doesn't exist
        return {
            "version": "0.1.0",
            "algorithm": "Ensemble",
            "performance_metrics": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
            },
            "last_trained": datetime.datetime.now().isoformat(),
            "feature_importance": {},
        }
    
    def _load_model(self) -> None:
        """Load the model from file."""
        # Try to load the ensemble model first
        try:
            self.ensemble_model = EnsembleFraudModel()
            if self.ensemble_model.load():
                logger.info("Ensemble model loaded successfully")
                self.metadata = self.ensemble_model.metadata
                return
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            self.ensemble_model = None
        
        # Fall back to the old model loading if ensemble not available
        if self.model_path.exists():
            try:
                model_data = joblib.load(self.model_path)
                
                # Check if this is the new ensemble format
                if isinstance(model_data, dict) and "lightgbm_model" in model_data:
                    logger.info("Found ensemble model data, initializing ensemble")
                    self.ensemble_model = EnsembleFraudModel()
                    self.ensemble_model.lightgbm_model = model_data.get("lightgbm_model")
                    self.ensemble_model.random_forest_model = model_data.get("random_forest_model")
                    self.ensemble_model.ensemble_model = model_data.get("ensemble_model")
                    self.ensemble_model.metadata = model_data.get("metadata", self.metadata)
                    self.ensemble_model.version = model_data.get("version", "v1.0.0")
                    self.metadata = self.ensemble_model.metadata
                    logger.info(f"Ensemble model loaded from {self.model_path}")
                # Otherwise, it's the old format
                else:
                    # Legacy model loading (keeping for backward compatibility)
                    model = model_data.get("model")
                    encoders = model_data.get("encoders", {})
                    scaler = model_data.get("scaler")
                    
                    # Create an ensemble model wrapper around the old model
                    self.ensemble_model = EnsembleFraudModel()
                    # Use LightGBM for backward compatibility
                    self.ensemble_model.lightgbm_model = model
                    self.ensemble_model.ensemble_model = model
                    
                    # Store old encoders and scaler in the data processor
                    self.data_processor.feature_encoders = encoders
                    self.data_processor.scaler = scaler
                    
                    logger.info(f"Legacy model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Create a dummy model for demo
                self._create_dummy_model()
        else:
            logger.warning(f"Model file not found at {self.model_path}")
            # Create a dummy model for demo
            self._create_dummy_model()
    
    def _create_dummy_model(self) -> None:
        """Create a dummy model for demo purposes."""
        # Create a dummy ensemble model with just LightGBM
        self.ensemble_model = EnsembleFraudModel()
        
        # Generate some random dummy data
        X = np.random.random((100, len(FEATURE_COLUMNS)))
        y = (X[:, 0] > 0.5).astype(int)
        
        # Fit the model on dummy data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.ensemble_model.fit(X_train, y_train, X_test, y_test)
        
        logger.info("Dummy ensemble model created for demo purposes")
    
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.ensemble_model is not None and self.ensemble_model.ensemble_model is not None
    
    def predict(self, transaction: Dict) -> Dict:
        """Predict fraud for a single transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dict: Prediction results
        """
        if not self.is_model_loaded():
            logger.error("Model not loaded")
            # Return a default prediction
            return {
                "transaction_id": transaction.get("transaction_id", None),
                "fraud_probability": 0.5,
                "is_fraud": False,
                "model_version": self.metadata.get("version", "unknown"),
            }
        
        # Preprocess transaction
        try:
            preprocessed_features = self.data_processor.preprocess_transaction(transaction)
            
            # Make prediction
            fraud_probability = float(self.ensemble_model.predict_proba(preprocessed_features)[0])
            is_fraud = self.ensemble_model.predict(preprocessed_features)[0] == 1
            
            return {
                "transaction_id": transaction.get("transaction_id", None),
                "fraud_probability": fraud_probability,
                "is_fraud": bool(is_fraud),
                "model_version": self.metadata.get("version", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return a default prediction on error
            return {
                "transaction_id": transaction.get("transaction_id", None),
                "fraud_probability": 0.5,
                "is_fraud": False,
                "model_version": self.metadata.get("version", "unknown"),
            }
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for a batch of transactions.
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List[Dict]: Prediction results for each transaction
        """
        return [self.predict(transaction) for transaction in transactions]
    
    def get_model_info(self) -> Dict:
        """Get information about the current model.
        
        Returns:
            Dict: Model information
        """
        # Get metadata from the ensemble model if available
        if self.ensemble_model and hasattr(self.ensemble_model, 'metadata'):
            return self.ensemble_model.metadata
        return self.metadata
    
    def train_model(self, db, test_size: float = 0.2, random_state: int = 42) -> None:
        """Train a new model on data from the database.
        
        Args:
            db: Database client for retrieving training data
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
        """
        logger.info("Starting model training")
        
        try:
            # Fetch transactions from database
            transactions = db.get_transactions()
            
            # Convert to DataFrame if not already
            if not isinstance(transactions, pd.DataFrame):
                transactions = pd.DataFrame(transactions)
            
            # Process data
            data_processor = DataProcessor()
            
            # Generate training/validation splits
            X_train, X_val, X_test, y_train, y_val, y_test = data_processor.process_data_pipeline(
                sample_size=None,  # Use all available data
                test_size=test_size,
                val_size=0.1,  # 10% validation data
                save_processed=True,
                random_state=random_state,
            )
            
            # Initialize ensemble model
            model_version = self._generate_model_version()
            model = EnsembleFraudModel(version=model_version)
            
            # Train model
            model.fit(
                X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                y_train=y_train.values if isinstance(y_train, pd.Series) else y_train,
                X_val=X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
                y_val=y_val.values if isinstance(y_val, pd.Series) else y_val,
            )
            
            # Save model
            model.save()
            
            # Update instance variables
            self.ensemble_model = model
            self.metadata = model.metadata
            self.data_processor = data_processor
            
            logger.info(f"Model training completed successfully (version {model_version})")
            
            # Return success status
            return {
                "status": "success",
                "version": model_version,
                "metrics": model.metadata.get("performance_metrics", {}),
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def _generate_model_version(self) -> str:
        """Generate a new model version based on the current date.
        
        Returns:
            str: New model version
        """
        now = datetime.datetime.now()
        return f"v{now.year}.{now.month}.{now.day}" 