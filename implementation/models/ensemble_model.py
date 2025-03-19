"""
Ensemble model for fraud detection.

This module implements an ensemble approach combining multiple models
for improved fraud detection performance.
"""
import datetime
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# Try to import TensorFlow for neural network model
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Input
    )
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network model will not be used.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = Path("implementation/models")


class EnsembleFraudModel:
    """
    Ensemble model for fraud detection combining LightGBM, Random Forest,
    and a neural network (if TensorFlow is available).
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        version: str = "v1.0.0",
        use_neural_network: bool = True,
    ):
        """
        Initialize the ensemble model.
        
        Args:
            model_dir: Directory to save/load models. If None, uses the default.
            version: Model version string.
            use_neural_network: Whether to include a neural network in the ensemble.
        """
        self.model_dir = model_dir or MODELS_DIR
        self.version = version
        self.use_neural_network = use_neural_network and TENSORFLOW_AVAILABLE
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.lightgbm_model = None
        self.random_forest_model = None
        self.neural_network_model = None
        self.ensemble_model = None
        
        # Metadata
        self.metadata = {
            "version": version,
            "algorithm": "Ensemble (LightGBM + RandomForest" + 
                        (" + NeuralNetwork" if self.use_neural_network else "") + ")",
            "performance_metrics": {},
            "feature_importance": {},
            "last_trained": None,
            "model_params": {},
        }
        
    def build_lightgbm_model(self, params: Optional[Dict] = None) -> lgb.LGBMClassifier:
        """
        Build a LightGBM model with the specified parameters.
        
        Args:
            params: Parameters for the LightGBM model. If None, uses defaults.
            
        Returns:
            lgb.LGBMClassifier: The initialized model
        """
        default_params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 10,
            "num_leaves": 31,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        # Update default params with provided params
        if params:
            default_params.update(params)
        
        # Save params in metadata
        self.metadata["model_params"]["lightgbm"] = default_params
        
        # Initialize and return model
        return lgb.LGBMClassifier(**default_params)
    
    def build_random_forest_model(self, params: Optional[Dict] = None) -> RandomForestClassifier:
        """
        Build a Random Forest model with the specified parameters.
        
        Args:
            params: Parameters for the Random Forest model. If None, uses defaults.
            
        Returns:
            RandomForestClassifier: The initialized model
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0,
        }
        
        # Update default params with provided params
        if params:
            default_params.update(params)
        
        # Save params in metadata
        self.metadata["model_params"]["random_forest"] = default_params
        
        # Initialize and return model
        return RandomForestClassifier(**default_params)
    
    def build_neural_network_model(
        self, 
        input_dim: int,
        params: Optional[Dict] = None
    ) -> Optional[Model]:
        """
        Build a neural network model with the specified parameters.
        
        Args:
            input_dim: Number of input features.
            params: Parameters for the neural network. If None, uses defaults.
            
        Returns:
            Model: The initialized Keras model or None if TensorFlow is not available
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping neural network model.")
            return None
        
        default_params = {
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "activation": "relu",
            "output_activation": "sigmoid",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy", "AUC"],
        }
        
        # Update default params with provided params
        if params:
            default_params.update(params)
        
        # Save params in metadata
        self.metadata["model_params"]["neural_network"] = default_params
        
        # Create model
        inputs = Input(shape=(input_dim,))
        
        # First hidden layer
        x = Dense(
            default_params["hidden_layers"][0],
            activation=default_params["activation"]
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(default_params["dropout_rate"])(x)
        
        # Additional hidden layers
        for units in default_params["hidden_layers"][1:]:
            x = Dense(units, activation=default_params["activation"])(x)
            x = BatchNormalization()(x)
            x = Dropout(default_params["dropout_rate"])(x)
        
        # Output layer
        outputs = Dense(1, activation=default_params["output_activation"])(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=default_params["learning_rate"]),
            loss=default_params["loss"],
            metrics=default_params["metrics"],
        )
        
        return model
    
    def build_ensemble_model(
        self,
        weights: Optional[List[float]] = None
    ) -> VotingClassifier:
        """
        Build an ensemble model combining the individual models.
        
        Args:
            weights: Weights for each model in the ensemble. If None, uses equal weights.
            
        Returns:
            VotingClassifier: The ensemble model
        """
        # Initialize estimators list
        estimators = []
        
        # Add available models
        if self.lightgbm_model is not None:
            estimators.append(("lightgbm", self.lightgbm_model))
        
        if self.random_forest_model is not None:
            estimators.append(("random_forest", self.random_forest_model))
        
        # Note: Neural network is handled separately since it doesn't fit the sklearn API
        
        # Create voting classifier
        if len(estimators) > 1:
            return VotingClassifier(
                estimators=estimators,
                voting="soft",
                weights=weights[:len(estimators)] if weights else None,
            )
        elif len(estimators) == 1:
            # If only one model is available, return it directly
            return estimators[0][1]
        else:
            raise ValueError("No models available for the ensemble")
    
    def fit(
        self, 
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        lightgbm_params: Optional[Dict] = None,
        random_forest_params: Optional[Dict] = None,
        neural_network_params: Optional[Dict] = None,
        ensemble_weights: Optional[List[float]] = None,
    ) -> Dict:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            lightgbm_params: Parameters for LightGBM model
            random_forest_params: Parameters for Random Forest model
            neural_network_params: Parameters for neural network model
            ensemble_weights: Weights for ensemble models
            
        Returns:
            Dict: Training history and metrics
        """
        logger.info("Training ensemble fraud detection model")
        
        # Record training start time
        training_start = datetime.datetime.now()
        
        # Dictionary to store training history
        history = {}
        
        # Check for validation data
        has_validation = X_val is not None and y_val is not None
        
        # Train LightGBM model
        logger.info("Training LightGBM model")
        self.lightgbm_model = self.build_lightgbm_model(lightgbm_params)
        
        eval_set = [(X_train, y_train)]
        if has_validation:
            eval_set.append((X_val, y_val))
            
        self.lightgbm_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=50)] if has_validation else None
        )
        
        lgb_val_auc = None
        if has_validation:
            lgb_preds = self.lightgbm_model.predict_proba(X_val)[:, 1]
            lgb_val_auc = roc_auc_score(y_val, lgb_preds)
            logger.info(f"LightGBM validation AUC: {lgb_val_auc:.4f}")
        
        history["lightgbm"] = {
            "feature_importance": dict(zip(
                range(X_train.shape[1]),
                self.lightgbm_model.feature_importances_.tolist()
            )),
            "best_iteration": getattr(self.lightgbm_model, "best_iteration_", None),
            "val_auc": lgb_val_auc,
        }
        
        # Train Random Forest model
        logger.info("Training Random Forest model")
        self.random_forest_model = self.build_random_forest_model(random_forest_params)
        self.random_forest_model.fit(X_train, y_train)
        
        rf_val_auc = None
        if has_validation:
            rf_preds = self.random_forest_model.predict_proba(X_val)[:, 1]
            rf_val_auc = roc_auc_score(y_val, rf_preds)
            logger.info(f"Random Forest validation AUC: {rf_val_auc:.4f}")
        
        history["random_forest"] = {
            "feature_importance": dict(zip(
                range(X_train.shape[1]),
                self.random_forest_model.feature_importances_.tolist()
            )),
            "val_auc": rf_val_auc,
        }
        
        # Train neural network model if TensorFlow is available
        nn_val_auc = None
        if self.use_neural_network:
            logger.info("Training neural network model")
            self.neural_network_model = self.build_neural_network_model(
                X_train.shape[1], neural_network_params
            )
            
            if self.neural_network_model is not None:
                # Prepare callbacks
                callbacks = []
                
                # Add early stopping if validation data is available
                if has_validation:
                    callbacks.append(
                        EarlyStopping(
                            monitor="val_auc",
                            patience=10,
                            mode="max",
                            restore_best_weights=True,
                        )
                    )
                
                # Add model checkpoint
                model_checkpoint_path = self.model_dir / "nn_model_checkpoint.h5"
                callbacks.append(
                    ModelCheckpoint(
                        str(model_checkpoint_path),
                        monitor="val_auc" if has_validation else "auc",
                        mode="max",
                        save_best_only=True,
                        verbose=1,
                    )
                )
                
                # Train model
                nn_history = self.neural_network_model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val) if has_validation else None,
                    callbacks=callbacks,
                    verbose=2,
                )
                
                if has_validation:
                    nn_preds = self.neural_network_model.predict(X_val)
                    nn_val_auc = roc_auc_score(y_val, nn_preds)
                    logger.info(f"Neural network validation AUC: {nn_val_auc:.4f}")
                
                # Store training history
                history["neural_network"] = {
                    "training_history": {
                        k: [float(v) for v in vs] 
                        for k, vs in nn_history.history.items()
                    },
                    "val_auc": nn_val_auc,
                }
                
                # Load best model if checkpoint was saved
                if model_checkpoint_path.exists():
                    self.neural_network_model = load_model(str(model_checkpoint_path))
                    logger.info("Loaded best neural network model from checkpoint")
        
        # Build ensemble model
        if self.lightgbm_model is not None and self.random_forest_model is not None:
            logger.info("Building ensemble model")
            self.ensemble_model = self.build_ensemble_model(ensemble_weights)
            
            # If ensemble is more than a single model, fit it on the training data
            if isinstance(self.ensemble_model, VotingClassifier):
                self.ensemble_model.fit(X_train, y_train)
        else:
            # Use LightGBM as the ensemble model if only it is available
            self.ensemble_model = self.lightgbm_model if self.lightgbm_model else self.random_forest_model
            logger.warning("Using single model as ensemble because other models are not available")
        
        # Evaluate ensemble model on validation data
        if has_validation and self.ensemble_model is not None:
            # Get ensemble predictions
            ensemble_preds = self.predict_proba(X_val)
            
            # Compute metrics
            ensemble_val_auc = roc_auc_score(y_val, ensemble_preds)
            ensemble_y_pred = (ensemble_preds > 0.5).astype(int)
            
            metrics = {
                "accuracy": accuracy_score(y_val, ensemble_y_pred),
                "precision": precision_score(y_val, ensemble_y_pred),
                "recall": recall_score(y_val, ensemble_y_pred),
                "f1": f1_score(y_val, ensemble_y_pred),
                "auc": ensemble_val_auc,
            }
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_val, ensemble_y_pred).ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
            
            # Find best threshold using precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_val, ensemble_preds)
            # Find threshold with F1 score = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx]
            metrics["best_threshold"] = float(best_threshold)
            
            # Log metrics
            logger.info(f"Ensemble validation metrics: {metrics}")
            
            # Store metrics in history
            history["ensemble"] = {
                "metrics": metrics,
                "val_auc": ensemble_val_auc,
            }
            
            # Update metadata with performance metrics
            self.metadata["performance_metrics"] = metrics
        
        # Update metadata
        feature_importance = {}
        if self.lightgbm_model is not None:
            feature_importance["lightgbm"] = dict(zip(
                range(X_train.shape[1]),
                self.lightgbm_model.feature_importances_.tolist()
            ))
        
        if self.random_forest_model is not None:
            feature_importance["random_forest"] = dict(zip(
                range(X_train.shape[1]),
                self.random_forest_model.feature_importances_.tolist()
            ))
        
        # Combine feature importances using a weighted average if both models are available
        if "lightgbm" in feature_importance and "random_forest" in feature_importance:
            combined_importance = np.zeros(X_train.shape[1])
            for model_name, importance in feature_importance.items():
                # Weight by validation AUC if available
                weight = 1.0
                if model_name == "lightgbm" and lgb_val_auc is not None:
                    weight = lgb_val_auc
                elif model_name == "random_forest" and rf_val_auc is not None:
                    weight = rf_val_auc
                
                for feature, value in importance.items():
                    combined_importance[int(feature)] += value * weight
            
            # Normalize
            if np.sum(combined_importance) > 0:
                combined_importance = combined_importance / np.sum(combined_importance)
            
            feature_importance["combined"] = dict(zip(
                range(X_train.shape[1]),
                combined_importance.tolist()
            ))
        
        self.metadata["feature_importance"] = feature_importance
        self.metadata["last_trained"] = datetime.datetime.now().isoformat()
        
        # Record training duration
        training_duration = datetime.datetime.now() - training_start
        logger.info(f"Model training completed in {training_duration}")
        history["training_duration"] = str(training_duration)
        
        return history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions from the ensemble model.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Probability predictions
        """
        if self.ensemble_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # If ensemble is a VotingClassifier, use its predict_proba method
        if isinstance(self.ensemble_model, VotingClassifier):
            return self.ensemble_model.predict_proba(X)[:, 1]
        
        # If it's a single model from scikit-learn (e.g., LightGBM or RandomForest)
        elif hasattr(self.ensemble_model, "predict_proba"):
            return self.ensemble_model.predict_proba(X)[:, 1]
        
        # If it's the neural network
        elif self.neural_network_model is not None and self.ensemble_model is None:
            return self.neural_network_model.predict(X).flatten()
        
        # If we have component models but no ensemble, combine them manually
        probas = []
        weights = []
        
        # Get predictions from each model
        if self.lightgbm_model is not None:
            lgb_proba = self.lightgbm_model.predict_proba(X)[:, 1]
            probas.append(lgb_proba)
            weights.append(1.0)
        
        if self.random_forest_model is not None:
            rf_proba = self.random_forest_model.predict_proba(X)[:, 1]
            probas.append(rf_proba)
            weights.append(1.0)
        
        if self.neural_network_model is not None:
            nn_proba = self.neural_network_model.predict(X).flatten()
            probas.append(nn_proba)
            weights.append(1.0)
        
        # Combine predictions with weights
        if probas:
            weights = np.array(weights) / sum(weights)  # Normalize weights
            combined_proba = np.zeros_like(probas[0])
            for i, proba in enumerate(probas):
                combined_proba += proba * weights[i]
            return combined_proba
        
        # If no models are available, return zeros
        logger.error("No models available for prediction")
        return np.zeros(X.shape[0])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions from the ensemble model.
        
        Args:
            X: Features to predict
            threshold: Probability threshold for positive class
            
        Returns:
            np.ndarray: Binary predictions
        """
        # Use best threshold from metadata if available and not overridden
        if "performance_metrics" in self.metadata and \
           "best_threshold" in self.metadata["performance_metrics"] and \
           threshold == 0.5:
            threshold = self.metadata["performance_metrics"]["best_threshold"]
        
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def save(self, save_nn_model: bool = True) -> Dict[str, Path]:
        """
        Save the ensemble model components to disk.
        
        Args:
            save_nn_model: Whether to save the neural network model separately
            
        Returns:
            Dict[str, Path]: Paths to saved model files
        """
        logger.info(f"Saving model to {self.model_dir}")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        saved_paths = {}
        
        # Save sklearn models (LightGBM, RandomForest, Ensemble)
        sklearn_models = {
            "lightgbm": self.lightgbm_model,
            "random_forest": self.random_forest_model,
        }
        
        # Only add ensemble if it's not one of the individual models
        if isinstance(self.ensemble_model, VotingClassifier):
            sklearn_models["ensemble"] = self.ensemble_model
        
        # Save each sklearn model
        for name, model in sklearn_models.items():
            if model is not None:
                model_path = self.model_dir / f"{name}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                saved_paths[name] = model_path
                logger.info(f"Saved {name} model to {model_path}")
        
        # Save neural network model separately if requested
        if save_nn_model and self.neural_network_model is not None:
            nn_model_path = self.model_dir / "neural_network_model.h5"
            self.neural_network_model.save(nn_model_path)
            saved_paths["neural_network"] = nn_model_path
            logger.info(f"Saved neural network model to {nn_model_path}")
        
        # Save metadata
        metadata_path = self.model_dir / "ensemble_metadata.json"
        with open(metadata_path, "w") as f:
            # Convert NumPy types to Python native types for JSON serialization
            json_compatible_metadata = json.loads(
                json.dumps(self.metadata, default=lambda x: float(x) if isinstance(x, np.number) else x)
            )
            json.dump(json_compatible_metadata, f, indent=2)
        
        saved_paths["metadata"] = metadata_path
        logger.info(f"Saved model metadata to {metadata_path}")
        
        # Save combined model object for convenience
        combined_path = self.model_dir / "fraud_ensemble_model.joblib"
        model_data = {
            "lightgbm_model": self.lightgbm_model,
            "random_forest_model": self.random_forest_model,
            "ensemble_model": self.ensemble_model,
            "metadata": self.metadata,
            "version": self.version,
        }
        
        joblib.dump(model_data, combined_path)
        saved_paths["combined"] = combined_path
        logger.info(f"Saved combined model to {combined_path}")
        
        return saved_paths
    
    def load(self, version: Optional[str] = None) -> bool:
        """
        Load the ensemble model components from disk.
        
        Args:
            version: Model version to load. If None, loads the latest.
            
        Returns:
            bool: Whether the model was loaded successfully
        """
        load_version = version or self.version
        logger.info(f"Loading model version {load_version} from {self.model_dir}")
        
        # Try to load the combined model file first (fastest approach)
        combined_path = self.model_dir / "fraud_ensemble_model.joblib"
        if combined_path.exists():
            try:
                model_data = joblib.load(combined_path)
                self.lightgbm_model = model_data.get("lightgbm_model")
                self.random_forest_model = model_data.get("random_forest_model")
                self.ensemble_model = model_data.get("ensemble_model")
                self.metadata = model_data.get("metadata", self.metadata)
                self.version = model_data.get("version", self.version)
                
                logger.info(f"Loaded combined model from {combined_path}")
                
                # Load neural network model if available
                nn_model_path = self.model_dir / "neural_network_model.h5"
                if self.use_neural_network and nn_model_path.exists():
                    try:
                        self.neural_network_model = load_model(str(nn_model_path))
                        logger.info(f"Loaded neural network model from {nn_model_path}")
                    except Exception as e:
                        logger.warning(f"Error loading neural network model: {e}")
                
                return True
            
            except Exception as e:
                logger.warning(f"Error loading combined model: {e}. Will try individual models.")
        
        # Load individual models if combined model couldn't be loaded
        success = False
        
        # Load LightGBM model
        lgb_path = self.model_dir / "lightgbm_model.pkl"
        if lgb_path.exists():
            try:
                with open(lgb_path, "rb") as f:
                    self.lightgbm_model = pickle.load(f)
                logger.info(f"Loaded LightGBM model from {lgb_path}")
                success = True
            except Exception as e:
                logger.warning(f"Error loading LightGBM model: {e}")
        
        # Load Random Forest model
        rf_path = self.model_dir / "random_forest_model.pkl"
        if rf_path.exists():
            try:
                with open(rf_path, "rb") as f:
                    self.random_forest_model = pickle.load(f)
                logger.info(f"Loaded Random Forest model from {rf_path}")
                success = True
            except Exception as e:
                logger.warning(f"Error loading Random Forest model: {e}")
        
        # Load neural network model
        nn_path = self.model_dir / "neural_network_model.h5"
        if self.use_neural_network and nn_path.exists():
            try:
                self.neural_network_model = load_model(str(nn_path))
                logger.info(f"Loaded neural network model from {nn_path}")
                success = True
            except Exception as e:
                logger.warning(f"Error loading neural network model: {e}")
        
        # Load ensemble model
        ensemble_path = self.model_dir / "ensemble_model.pkl"
        if ensemble_path.exists():
            try:
                with open(ensemble_path, "rb") as f:
                    self.ensemble_model = pickle.load(f)
                logger.info(f"Loaded ensemble model from {ensemble_path}")
                success = True
            except Exception as e:
                logger.warning(f"Error loading ensemble model: {e}")
        
        # Load metadata
        metadata_path = self.model_dir / "ensemble_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded model metadata from {metadata_path}")
            except Exception as e:
                logger.warning(f"Error loading model metadata: {e}")
        
        # If no ensemble model but individual models are loaded, create an ensemble
        if self.ensemble_model is None and (self.lightgbm_model is not None or self.random_forest_model is not None):
            try:
                self.ensemble_model = self.build_ensemble_model()
                logger.info("Created ensemble model from individual models")
            except Exception as e:
                logger.warning(f"Error creating ensemble model: {e}")
                # Use one of the individual models if ensemble creation fails
                self.ensemble_model = self.lightgbm_model or self.random_forest_model
                logger.warning("Using individual model as fallback")
        
        return success

    
# Example usage
if __name__ == "__main__":
    # Create dummy data for testing
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = (X[:, 0] > 0.7).astype(int)  # Simple rule for binary target
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = EnsembleFraudModel(version="v0.1.0", use_neural_network=TENSORFLOW_AVAILABLE)
    history = model.fit(X_train, y_train, X_test, y_test)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Evaluate
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    
    # Save model
    model.save()
    
    # Load model and test
    new_model = EnsembleFraudModel()
    new_model.load()
    new_preds = new_model.predict(X_test)
    
    # Check predictions are the same
    print(f"Predictions match: {np.array_equal(y_pred, new_preds)}") 