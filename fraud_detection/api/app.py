"""
Main FastAPI application for the Fraud Detection API.
"""
import os
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model_manager import ModelManager
from supabase_client import SupabaseClient, get_supabase_client

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent financial transactions",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Define API key authentication
async def verify_api_key(request: Request):
    """Verify the API key in the request header."""
    api_key = request.headers.get("X-API-Key")
    default_key = os.getenv("API_KEY", "demo_api_key_portfolio_project")
    
    if not api_key or api_key != default_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key

# Define data models
class Transaction(BaseModel):
    """Model for a financial transaction."""
    transaction_id: Optional[str] = None
    amount: float
    merchant_category: str
    merchant_name: str
    transaction_type: str
    card_present: bool
    country: str
    timestamp: str
    # Add other fields as needed

class TransactionBatch(BaseModel):
    """Model for a batch of financial transactions."""
    transactions: List[Transaction]

class PredictionResponse(BaseModel):
    """Model for a fraud prediction response."""
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    model_version: str

class BatchPredictionResponse(BaseModel):
    """Model for a batch fraud prediction response."""
    predictions: List[PredictionResponse]

class ModelInfo(BaseModel):
    """Model for information about the current model."""
    version: str
    algorithm: str
    performance_metrics: Dict
    last_trained: str
    feature_importance: Optional[Dict] = None

# Define routes
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Fraud Detection API",
        "version": "0.1.0",
        "description": "API for detecting fraudulent financial transactions",
        "documentation": "/docs",
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model_manager.is_model_loaded()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: Transaction,
    api_key: str = Depends(verify_api_key),
):
    """Predict fraud for a single transaction."""
    result = model_manager.predict(transaction.dict())
    return result

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: TransactionBatch,
    api_key: str = Depends(verify_api_key),
):
    """Predict fraud for a batch of transactions."""
    transactions = [t.dict() for t in batch.transactions]
    results = model_manager.predict_batch(transactions)
    return {"predictions": results}

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(
    api_key: str = Depends(verify_api_key),
):
    """Get information about the current model."""
    return model_manager.get_model_info()

@app.post("/model/train")
async def train_model(
    api_key: str = Depends(verify_api_key),
    db: SupabaseClient = Depends(get_supabase_client),
):
    """Train a new model."""
    # This would typically be a background task
    try:
        model_manager.train_model(db)
        return {"status": "success", "message": "Model training started"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting model training: {str(e)}",
        )

@app.get("/demo/reset")
async def reset_demo(
    api_key: str = Depends(verify_api_key),
    db: SupabaseClient = Depends(get_supabase_client),
):
    """Reset the demo environment with sample data."""
    try:
        # Reset would be implemented in the database client
        db.reset_demo()
        return {"status": "success", "message": "Demo environment reset"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting demo environment: {str(e)}",
        )

# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected error occurred: {str(exc)}"},
    ) 