"""
Model registry module for managing different models.
"""
from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class ModelRegistry:
    """Registry for managing different models."""
    
    @staticmethod
    def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
        """
        Get dictionary of baseline models.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of model names and their instances
        """
        return {
            "logistic_regression": LogisticRegression(
                class_weight="balanced",
                random_state=random_state
            ),
            "random_forest": RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state
            ),
            "xgboost": XGBClassifier(
                random_state=random_state
            )
        }
    
    @staticmethod
    def get_tuning_model(random_state: int = 42) -> XGBClassifier:
        """
        Get base model for hyperparameter tuning.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            Base model instance for tuning
        """
        return XGBClassifier(
            random_state=random_state
        )
    
    @staticmethod
    def set_class_weights(model: Any, y_train: np.ndarray) -> Any:
        """
        Set class weights for a model based on training data.
        
        Args:
            model: Model instance
            y_train: Training labels
            
        Returns:
            Model instance with updated class weights
        """
        if isinstance(model, XGBClassifier):
            model.set_params(scale_pos_weight=sum(y_train == 0) / sum(y_train == 1))
        return model 