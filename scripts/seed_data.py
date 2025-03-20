#!/usr/bin/env python3
"""Seed initial data into Supabase tables."""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return
    
    try:
        # Initialize Supabase client
        supabase = create_client(url, key)
        print("Connected to Supabase!")
        
        # Add demo model version
        print("\nAdding demo model version...")
        model_data = {
            "id": "v2023.1.1",
            "version": "v2023.1.1",
            "algorithm": "LightGBM",
            "performance_metrics": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.85,
                "f1": 0.88,
                "auc": 0.97
            },
            "last_trained": datetime.now().isoformat()
        }
        
        response = supabase.table("model_versions").insert(model_data).execute()
        print("✅ Model version added")
        
        # Add demo transactions
        print("\nAdding demo transactions...")
        transactions = [
            {
                "transaction_id": "tx_demo_1",
                "amount": 123.45,
                "merchant_category": "retail",
                "merchant_name": "ACME Store",
                "transaction_type": "purchase",
                "card_present": True,
                "country": "US",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "is_fraud": False
            },
            {
                "transaction_id": "tx_demo_2",
                "amount": 1500.00,
                "merchant_category": "electronics",
                "merchant_name": "TechGiant",
                "transaction_type": "purchase",
                "card_present": True,
                "country": "US",
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                "is_fraud": False
            },
            {
                "transaction_id": "tx_demo_3",
                "amount": 899.99,
                "merchant_category": "travel",
                "merchant_name": "Luxury Hotels",
                "transaction_type": "purchase",
                "card_present": False,
                "country": "FR",
                "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                "is_fraud": True
            }
        ]
        
        response = supabase.table("transactions").insert(transactions).execute()
        print("✅ Demo transactions added")
        
        print("\nSeeding completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 