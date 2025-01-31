from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API Settings
API_V1_STR = "/api/v1"
PROJECT_NAME = "Fraud Detection System"

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")

# BigQuery settings
BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "fraud_detection")

# Model settings
DEFAULT_MODEL_PATH = MODELS_DIR / "fraud_detection_model.pkl"
MODEL_VERSION = "0.1.0"

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-development")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Feature names used in the model
FEATURE_COLUMNS = [
    "amount",
    "time_since_first_transaction",
    "time_since_last_transaction",
    "merchant_category",
    "merchant_country",
    "card_type",
    "transaction_type",
]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 