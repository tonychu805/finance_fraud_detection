"""
API module for fraud detection system.

This module provides the REST API interface for the fraud detection
service, including endpoints for prediction and model management.
"""

from fraud_detection.api.app import app

__all__ = ["app"]
