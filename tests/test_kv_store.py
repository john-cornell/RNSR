"""Tests for Key-Value Store."""

import tempfile
from pathlib import Path

import pytest
from rnsr.indexing.kv_store import InMemoryKVStore, SQLiteKVStore


class TestInMemoryKVStore:
    """Tests for in-memory KV store."""
    
    def test_put_and_get(self):
        store = InMemoryKVStore()
        store.put("key1", "value1")
        
        assert store.get("key1") == "value1"
    
    def test_get_nonexistent(self):
        store = InMemoryKVStore()
        assert store.get("nonexistent") is None
    
    def test_multiple_entries(self):
        store = InMemoryKVStore()
        store.put("a", "alpha")
        store.put("b", "beta")
        store.put("c", "gamma")
        
        assert store.get("a") == "alpha"
        assert store.get("b") == "beta"
        assert store.get("c") == "gamma"
    
    def test_overwrite(self):
        store = InMemoryKVStore()
        store.put("key", "original")
        store.put("key", "updated")
        
        assert store.get("key") == "updated"
    
    def test_delete(self):
        store = InMemoryKVStore()
        store.put("key", "value")
        store.delete("key")
        
        assert store.get("key") is None
    
    def test_count(self):
        store = InMemoryKVStore()
        assert store.count() == 0
        
        store.put("a", "1")
        store.put("b", "2")
        assert store.count() == 2
    
    def test_exists(self):
        store = InMemoryKVStore()
        store.put("exists", "value")
        
        assert store.exists("exists") is True
        assert store.exists("notexists") is False


class TestSQLiteKVStore:
    """Tests for SQLite-backed KV store."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_put_and_get(self, temp_db):
        store = SQLiteKVStore(temp_db)
        store.put("key1", "value1")
        
        assert store.get("key1") == "value1"
    
    def test_persistence(self, temp_db):
        # Write
        store1 = SQLiteKVStore(temp_db)
        store1.put("persistent", "data")
        
        # Read with new instance
        store2 = SQLiteKVStore(temp_db)
        assert store2.get("persistent") == "data"
    
    def test_large_content(self, temp_db):
        store = SQLiteKVStore(temp_db)
        large_content = "x" * 100000  # 100KB
        store.put("large", large_content)
        
        retrieved = store.get("large")
        assert retrieved is not None
        assert retrieved == large_content
        assert len(retrieved) == 100000
    
    def test_count(self, temp_db):
        store = SQLiteKVStore(temp_db)
        store.put("a", "1")
        store.put("b", "2")
        store.put("c", "3")
        
        assert store.count() == 3
    
    def test_delete(self, temp_db):
        store = SQLiteKVStore(temp_db)
        store.put("temp", "data")
        assert store.get("temp") == "data"
        
        store.delete("temp")
        assert store.get("temp") is None
    
    def test_exists(self, temp_db):
        store = SQLiteKVStore(temp_db)
        store.put("exists", "value")
        
        assert store.exists("exists") is True
        assert store.exists("notexists") is False
    
    def test_get_batch(self, temp_db):
        store = SQLiteKVStore(temp_db)
        store.put("a", "alpha")
        store.put("b", "beta")
        store.put("c", "gamma")
        
        results = store.get_batch(["a", "b", "missing"])
        
        assert results["a"] == "alpha"
        assert results["b"] == "beta"
        assert results["missing"] is None
