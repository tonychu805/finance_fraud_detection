"""
Supabase client for database operations.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from supabase import create_client, Client
from dotenv import load_dotenv
import os

class SupabaseClient:
    """Client for interacting with Supabase database."""
    
    def __init__(self):
        """Initialize Supabase client."""
        load_dotenv()
        
        # Get Supabase credentials from environment variables
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Supabase credentials not found in environment variables")
        
        # Initialize Supabase client
        self.client: Client = create_client(self.url, self.key)
        
    def save_model_results(self, results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Save model results to Supabase.
        
        Args:
            results: Dictionary containing model results
            model_name: Name of the model
            
        Returns:
            Dictionary containing the saved record
        """
        # Prepare data for storage
        data = {
            "model_name": model_name,
            "results": json.dumps(results),
            "created_at": datetime.utcnow().isoformat(),
            "metrics": {
                "roc_auc": results.get("baseline_evaluation", {}).get("xgboost", {}).get("roc_auc", 0),
                "average_precision": results.get("baseline_evaluation", {}).get("xgboost", {}).get("average_precision", 0)
            }
        }
        
        # Insert data into the model_results table
        response = self.client.table("model_results").insert(data).execute()
        
        return response.data[0]
    
    def get_model_results(self, model_name: Optional[str] = None) -> list:
        """
        Retrieve model results from Supabase.
        
        Args:
            model_name: Optional name of the model to filter results
            
        Returns:
            List of model results
        """
        query = self.client.table("model_results")
        
        if model_name:
            query = query.eq("model_name", model_name)
        
        response = query.order("created_at", desc=True).execute()
        
        # Parse JSON results
        for record in response.data:
            record["results"] = json.loads(record["results"])
        
        return response.data
    
    def save_feature_importance(self, feature_importance: Dict[str, float], model_name: str) -> Dict[str, Any]:
        """
        Save feature importance to Supabase.
        
        Args:
            feature_importance: Dictionary of feature names and their importance scores
            model_name: Name of the model
            
        Returns:
            Dictionary containing the saved record
        """
        data = {
            "model_name": model_name,
            "feature_importance": json.dumps(feature_importance),
            "created_at": datetime.utcnow().isoformat()
        }
        
        response = self.client.table("feature_importance").insert(data).execute()
        
        return response.data[0]
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> list:
        """
        Retrieve feature importance from Supabase.
        
        Args:
            model_name: Optional name of the model to filter results
            
        Returns:
            List of feature importance records
        """
        query = self.client.table("feature_importance")
        
        if model_name:
            query = query.eq("model_name", model_name)
        
        response = query.order("created_at", desc=True).execute()
        
        # Parse JSON feature importance
        for record in response.data:
            record["feature_importance"] = json.loads(record["feature_importance"])
        
        return response.data
    
    def save_model_metrics(self, metrics: Dict[str, float], model_name: str) -> Dict[str, Any]:
        """
        Save model metrics to Supabase.
        
        Args:
            metrics: Dictionary of metric names and their values
            model_name: Name of the model
            
        Returns:
            Dictionary containing the saved record
        """
        data = {
            "model_name": model_name,
            "metrics": json.dumps(metrics),
            "created_at": datetime.utcnow().isoformat()
        }
        
        response = self.client.table("model_metrics").insert(data).execute()
        
        return response.data[0]
    
    def get_model_metrics(self, model_name: Optional[str] = None) -> list:
        """
        Retrieve model metrics from Supabase.
        
        Args:
            model_name: Optional name of the model to filter results
            
        Returns:
            List of model metrics records
        """
        query = self.client.table("model_metrics")
        
        if model_name:
            query = query.eq("model_name", model_name)
        
        response = query.order("created_at", desc=True).execute()
        
        # Parse JSON metrics
        for record in response.data:
            record["metrics"] = json.loads(record["metrics"])
        
        return response.data 