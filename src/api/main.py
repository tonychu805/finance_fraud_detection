from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROJECT_NAME, API_V1_STR, LOG_FORMAT, LOG_LEVEL

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=PROJECT_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

@app.get(f"{API_V1_STR}/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Fraud Detection API",
        "docs_url": "/docs"
    }

# Import and include routers
# from .routers import predictions, models
# app.include_router(predictions.router, prefix=f"{API_V1_STR}/predictions", tags=["predictions"])
# app.include_router(models.router, prefix=f"{API_V1_STR}/models", tags=["models"]) 