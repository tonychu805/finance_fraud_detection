#!/usr/bin/env python3
"""Test Supabase connection."""

import os
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
        
        # Test connection by fetching model versions
        response = supabase.table('model_versions').select("*").execute()
        print("Supabase connection successful!")
        print(f"Found {len(response.data)} model versions")
        
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")

if __name__ == "__main__":
    main() 