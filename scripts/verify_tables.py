#!/usr/bin/env python3
"""Verify Supabase tables and their structures."""

import os
from dotenv import load_dotenv
from supabase import create_client
from typing import List, Dict

def check_table(supabase, table_name: str) -> Dict:
    """Check if a table exists and return its contents."""
    try:
        response = supabase.table(table_name).select("*").execute()
        return {
            "exists": True,
            "count": len(response.data),
            "data": response.data
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }

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
        
        # Tables to check
        tables = [
            "transactions",
            "model_versions",
            "users",
            "api_keys"
        ]
        
        # Check each table
        print("\nChecking tables:")
        print("-" * 50)
        
        for table in tables:
            result = check_table(supabase, table)
            if result["exists"]:
                print(f"✅ {table}:")
                print(f"   - Records: {result['count']}")
                if result['count'] > 0:
                    print(f"   - Sample record: {result['data'][0]}")
            else:
                print(f"❌ {table}:")
                print(f"   - Error: {result['error']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")

if __name__ == "__main__":
    main() 