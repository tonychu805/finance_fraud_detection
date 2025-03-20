"""
Database module for fraud detection system.

This module handles all database operations including
Supabase integration and data persistence.
"""

from fraud_detection.database.supabase_client import SupabaseClient

__all__ = ["SupabaseClient"] 