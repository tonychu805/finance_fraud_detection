#!/usr/bin/env python3
"""Initialize Supabase database with required schema."""

import os
import sys
from pathlib import Path
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
        sys.exit(1)
    
    # Initialize Supabase client
    supabase = create_client(url, key)
    
    # Read SQL setup file
    sql_path = Path(__file__).parent.parent / "implementation" / "supabase_setup.sql"
    with open(sql_path) as f:
        sql = f.read()
    
    try:
        # Execute SQL setup
        result = supabase.table("model_versions").select("*").execute()
        print("Connected to Supabase successfully!")
        
        # Run setup SQL
        for statement in sql.split(';'):
            if statement.strip():
                supabase.query(statement).execute()
        
        print("Database schema initialized successfully!")
        
        # Run demo seed function
        supabase.rpc('reset_demo').execute()
        print("Demo data seeded successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 