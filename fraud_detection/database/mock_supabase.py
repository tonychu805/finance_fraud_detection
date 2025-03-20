"""
Mock Supabase client for testing.
"""
from typing import Dict, Any, List
from datetime import datetime

class MockResponse:
    """Mock response from Supabase."""
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

class MockTable:
    """Mock table operations."""
    def __init__(self):
        self.data = []
        
    def insert(self, record: Dict[str, Any]) -> 'MockTable':
        """Mock insert operation."""
        self.data.append(record)
        return self
        
    def select(self, *fields: str) -> 'MockTable':
        """Mock select operation."""
        return self
        
    def order(self, field: str, desc: bool = False) -> 'MockTable':
        """Mock order operation."""
        return self
        
    def limit(self, n: int) -> 'MockTable':
        """Mock limit operation."""
        return self
        
    def in_(self, field: str, values: List[Any]) -> 'MockTable':
        """Mock in_ operation."""
        return self
        
    def eq(self, field: str, value: Any) -> 'MockTable':
        """Mock eq operation."""
        return self
        
    async def execute(self) -> MockResponse:
        """Mock execute operation."""
        return MockResponse(self.data)

class MockClient:
    """Mock Supabase client."""
    def __init__(self):
        self._tables = {}
        
    def table(self, name: str) -> MockTable:
        """Get or create a mock table."""
        if name not in self._tables:
            self._tables[name] = MockTable()
        return self._tables[name]

class MockSupabaseClient:
    """Mock Supabase client for testing."""
    def __init__(self):
        """Initialize mock client."""
        self.client = MockClient() 