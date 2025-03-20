#!/usr/bin/env python3
"""Initialize Supabase database tables."""

import os
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
        return
    
    try:
        # Initialize Supabase client
        supabase = create_client(url, key)
        print("Connected to Supabase!")
        
        # Read SQL setup file
        sql_path = Path(__file__).parent.parent / "implementation" / "supabase_setup.sql"
        with open(sql_path) as f:
            sql = f.read()
        
        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
        
        # Execute each statement
        for stmt in statements:
            try:
                supabase.rpc('exec_sql', {'sql': stmt}).execute()
                print(f"Executed: {stmt[:50]}...")
            except Exception as e:
                print(f"Error executing statement: {e}")
                print(f"Statement was: {stmt}")
                continue
        
        print("\nDatabase initialization completed!")
        
        # Test by querying the model_versions table
        response = supabase.table('model_versions').select("*").execute()
        print(f"Found {len(response.data)} model versions")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 