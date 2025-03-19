"""
Supabase client for database interactions.
"""
import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import Depends

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Try to import Supabase
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    logger.warning("Supabase client not available. Using mock mode.")
    SUPABASE_AVAILABLE = False

# Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")


class MockSupabaseClient:
    """Mock Supabase client for development."""
    
    def __init__(self):
        """Initialize the mock client with in-memory storage."""
        self.transactions = []
        self.users = []
        self.model_versions = []
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with mock data."""
        # Add some mock transactions
        self.transactions = [
            {
                "id": f"tx_{i}",
                "transaction_id": f"tx_{i}",
                "amount": 100 * (i % 10 + 1),
                "merchant_category": f"category_{i % 5}",
                "merchant_name": f"merchant_{i % 10}",
                "transaction_type": "purchase" if i % 3 != 0 else "withdrawal",
                "card_present": i % 2 == 0,
                "country": "US" if i % 4 != 0 else "CA",
                "timestamp": (
                    datetime.datetime.now() - datetime.timedelta(days=i % 30)
                ).isoformat(),
                "is_fraud": i % 20 == 0,  # 5% fraud rate
                "created_at": datetime.datetime.now().isoformat(),
            }
            for i in range(100)  # 100 mock transactions
        ]
        
        # Add some mock users
        self.users = [
            {
                "id": "demo_user",
                "email": "demo@example.com",
                "name": "Demo User",
                "created_at": datetime.datetime.now().isoformat(),
            }
        ]
        
        # Add a mock model version
        self.model_versions = [
            {
                "id": "v2023.1.1",
                "version": "v2023.1.1",
                "algorithm": "LightGBM",
                "performance_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.85,
                    "f1": 0.88,
                    "auc": 0.97,
                },
                "last_trained": datetime.datetime.now().isoformat(),
                "created_at": datetime.datetime.now().isoformat(),
            }
        ]
        
        logger.info("Mock data initialized")
    
    def get_transactions(self, limit: int = 1000) -> List[Dict]:
        """Get transactions from mock storage.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List[Dict]: List of transactions
        """
        return self.transactions[:limit]
    
    def add_transaction(self, transaction: Dict) -> Dict:
        """Add a transaction to mock storage.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dict: Added transaction
        """
        # Generate an ID if not provided
        if "id" not in transaction:
            transaction["id"] = f"tx_{len(self.transactions)}"
        
        # Add created_at timestamp
        if "created_at" not in transaction:
            transaction["created_at"] = datetime.datetime.now().isoformat()
        
        self.transactions.append(transaction)
        return transaction
    
    def add_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Add multiple transactions to mock storage.
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List[Dict]: Added transactions
        """
        return [self.add_transaction(tx) for tx in transactions]
    
    def get_model_versions(self) -> List[Dict]:
        """Get model versions from mock storage.
        
        Returns:
            List[Dict]: List of model versions
        """
        return self.model_versions
    
    def add_model_version(self, model_version: Dict) -> Dict:
        """Add a model version to mock storage.
        
        Args:
            model_version: Model version data
            
        Returns:
            Dict: Added model version
        """
        # Generate an ID if not provided
        if "id" not in model_version:
            model_version["id"] = model_version.get("version", f"v{len(self.model_versions)}")
        
        # Add created_at timestamp
        if "created_at" not in model_version:
            model_version["created_at"] = datetime.datetime.now().isoformat()
        
        self.model_versions.append(model_version)
        return model_version
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get a user from mock storage.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[Dict]: User data if found, None otherwise
        """
        for user in self.users:
            if user["id"] == user_id:
                return user
        return None
    
    def reset_demo(self) -> None:
        """Reset the demo environment with fresh mock data."""
        self.transactions = []
        self.model_versions = []
        self._initialize_mock_data()
        logger.info("Demo environment reset")


class SupabaseClient:
    """Client for interacting with Supabase."""
    
    def __init__(self, url: str = SUPABASE_URL, key: str = SUPABASE_KEY):
        """Initialize the Supabase client.
        
        Args:
            url: Supabase URL
            key: Supabase anonymous key
        """
        self.url = url
        self.key = key
        
        # Use mock client if Supabase is not available or credentials are missing
        if not SUPABASE_AVAILABLE or not url or not key:
            logger.warning("Using mock Supabase client")
            self.mock = MockSupabaseClient()
            self.client = None
        else:
            # Initialize Supabase client
            try:
                self.client = create_client(url, key)
                self.mock = None
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Error initializing Supabase client: {e}")
                logger.warning("Falling back to mock client")
                self.mock = MockSupabaseClient()
                self.client = None
    
    def is_mock(self) -> bool:
        """Check if using mock client.
        
        Returns:
            bool: True if using mock client, False otherwise
        """
        return self.client is None
    
    def get_transactions(self, limit: int = 1000) -> List[Dict]:
        """Get transactions from database.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List[Dict]: List of transactions
        """
        if self.is_mock():
            return self.mock.get_transactions(limit)
        
        try:
            response = self.client.table("transactions").select("*").limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return []
    
    def add_transaction(self, transaction: Dict) -> Dict:
        """Add a transaction to the database.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dict: Added transaction
        """
        if self.is_mock():
            return self.mock.add_transaction(transaction)
        
        try:
            response = self.client.table("transactions").insert(transaction).execute()
            return response.data[0] if response.data else transaction
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            return transaction
    
    def add_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Add multiple transactions to the database.
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List[Dict]: Added transactions
        """
        if self.is_mock():
            return self.mock.add_transactions(transactions)
        
        try:
            response = self.client.table("transactions").insert(transactions).execute()
            return response.data if response.data else transactions
        except Exception as e:
            logger.error(f"Error adding transactions: {e}")
            return transactions
    
    def get_model_versions(self) -> List[Dict]:
        """Get model versions from database.
        
        Returns:
            List[Dict]: List of model versions
        """
        if self.is_mock():
            return self.mock.get_model_versions()
        
        try:
            response = self.client.table("model_versions").select("*").order("created_at", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    def add_model_version(self, model_version: Dict) -> Dict:
        """Add a model version to the database.
        
        Args:
            model_version: Model version data
            
        Returns:
            Dict: Added model version
        """
        if self.is_mock():
            return self.mock.add_model_version(model_version)
        
        try:
            response = self.client.table("model_versions").insert(model_version).execute()
            return response.data[0] if response.data else model_version
        except Exception as e:
            logger.error(f"Error adding model version: {e}")
            return model_version
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get a user from the database.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[Dict]: User data if found, None otherwise
        """
        if self.is_mock():
            return self.mock.get_user(user_id)
        
        try:
            response = self.client.table("users").select("*").eq("id", user_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def reset_demo(self) -> None:
        """Reset the demo environment with fresh data."""
        if self.is_mock():
            self.mock.reset_demo()
            return
        
        try:
            # In a real implementation, this would use a SQL script to reset the database
            # For this demo, we'll just log that it would happen
            logger.info("Demo environment would be reset in Supabase")
        except Exception as e:
            logger.error(f"Error resetting demo environment: {e}")


# Dependency for FastAPI
def get_supabase_client() -> SupabaseClient:
    """Get a Supabase client instance.
    
    Returns:
        SupabaseClient: Supabase client
    """
    return SupabaseClient() 