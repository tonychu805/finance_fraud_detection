"""
Base classes for fraud detection models.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all fraud detection models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.metadata = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probability predictions
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        import joblib
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'metadata': self.metadata,
            'name': self.name
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        import joblib
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.metadata = model_data['metadata']
        self.name = model_data['name']


class BaseEnsemble(BaseModel):
    """Abstract base class for ensemble models."""
    
    def __init__(self, name: str, models: Optional[Dict[str, BaseModel]] = None):
        super().__init__(name)
        self.models = models or {}
        
    @abstractmethod
    def add_model(self, name: str, model: BaseModel) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Name of the model
            model: Model instance
        """
        pass
    
    @abstractmethod
    def remove_model(self, name: str) -> None:
        """
        Remove a model from the ensemble.
        
        Args:
            name: Name of the model to remove
        """
        pass 