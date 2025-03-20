"""
Core module for fraud detection system.

This module contains the fundamental components for data processing,
feature engineering, and exploratory data analysis.
"""

from fraud_detection.core.data_processor import DataProcessor
from fraud_detection.core.eda import FraudEDA

__all__ = ["DataProcessor", "FraudEDA"]
