"""
Pytest configuration and fixtures.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables for testing
load_dotenv()

@pytest.fixture
def test_data_path():
    """Path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_transaction():
    """Sample transaction data for testing."""
    return {
        "amount": 1000.00,
        "type": "TRANSFER",
        "oldbalanceOrg": 2000.00,
        "newbalanceOrig": 1000.00,
        "oldbalanceDest": 500.00,
        "newbalanceDest": 1500.00,
    }

@pytest.fixture
def model_config():
    """Test model configuration."""
    return {
        "model_type": "xgboost",
        "parameters": {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 100
        }
    }

@pytest.fixture
def api_client():
    """Test client for API endpoints."""
    from fraud_detection.api.app import app
    from fastapi.testclient import TestClient
    return TestClient(app) 