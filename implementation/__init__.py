"""
Fraud Detection Model Development Package.

This package provides a modular framework for developing and evaluating fraud detection models.
It includes data processing, model development, evaluation, and database integration.
"""

from .config.model_config import ModelConfig
from .model_development import FraudModelDevelopment
from .models.model_registry import ModelRegistry
from .models.ensemble_model import EnsembleFraudModel
from .evaluation.metrics import MetricsCalculator
from .utils.result_storage import ResultStorage
from .supabase_client import SupabaseClient, MockSupabaseClient
from .data_processor import DataProcessor
from .model_manager import ModelManager
from .eda import FraudEDA

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "FraudModelDevelopment",
    "ModelRegistry",
    "EnsembleFraudModel",
    "MetricsCalculator",
    "ResultStorage",
    "SupabaseClient",
    "MockSupabaseClient",
    "DataProcessor",
    "ModelManager",
    "FraudEDA"
] 