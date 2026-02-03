"""Tests for the LLM Response Cache system."""

import tempfile
import time
from pathlib import Path

import pytest

from rnsr.agent.llm_cache import (
    CacheEntry,
    CachedLLM,
    LLMCache,
    get_global_cache,
    wrap_llm_with_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_entry_creation(self):
        """Test creating a cache entry."""
        now = time.time()
        entry = CacheEntry(
            key="test_key",
            prompt_hash="abc123",
            prompt_preview="What is 2+2?",
            response="4",
            created_at=now,
            expires_at=now + 3600,
        )
        
        assert entry.key == "test_key"
        assert entry.response == "4"
        assert entry.hit_count == 0
    
    def test_entry_not_expired(self):
        """Test entry expiration check - not expired."""
        now = time.time()
        entry = CacheEntry(
            key="test",
            prompt_hash="hash",
            prompt_preview="test",
            response="response",
            created_at=now,
            expires_at=now + 3600,
        )
        
        assert not entry.is_expired()
    
    def test_entry_expired(self):
        """Test entry expiration check - expired."""
        now = time.time()
        entry = CacheEntry(
            key="test",
            prompt_hash="hash",
            prompt_preview="test",
            response="response",
            created_at=now - 7200,
            expires_at=now - 3600,
        )
        
        assert entry.is_expired()


class TestLLMCache:
    """Tests for LLMCache class."""
    
    def test_cache_creation_in_memory(self):
        """Test creating an in-memory cache."""
        cache = LLMCache(enable_persistence=False)
        assert cache is not None
        assert cache.enable_persistence is False
    
    def test_cache_creation_with_persistence(self, temp_dir):
        """Test creating a persistent cache."""
        cache_path = temp_dir / "test_cache.db"
        cache = LLMCache(storage_path=cache_path, enable_persistence=True)
        
        assert cache is not None
        assert cache.storage_path == cache_path
    
    def test_cache_set_and_get(self):
        """Test setting and getting a cached value."""
        cache = LLMCache(enable_persistence=False)
        
        prompt = "What is the capital of France?"
        response = "Paris"
        
        cache.set(prompt, response)
        result = cache.get(prompt)
        
        assert result == response
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LLMCache(enable_persistence=False)
        
        result = cache.get("Unknown prompt")
        
        assert result is None
    
    def test_cache_normalized_prompts(self):
        """Test that normalized prompts hit the same cache entry."""
        cache = LLMCache(enable_persistence=False)
        
        # Set with one formatting
        cache.set("What is 2+2?", "4")
        
        # Get with different whitespace
        result = cache.get("What  is  2+2?")
        
        assert result == "4"
    
    def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = LLMCache(enable_persistence=False, default_ttl_seconds=1)
        
        cache.set("prompt", "response")
        
        # Should exist immediately
        assert cache.get("prompt") == "response"
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired now
        assert cache.get("prompt") is None
    
    def test_cache_invalidate(self):
        """Test invalidating a cache entry."""
        cache = LLMCache(enable_persistence=False)
        
        cache.set("prompt", "response")
        assert cache.get("prompt") == "response"
        
        result = cache.invalidate("prompt")
        assert result is True
        
        assert cache.get("prompt") is None
    
    def test_cache_invalidate_nonexistent(self):
        """Test invalidating a non-existent entry."""
        cache = LLMCache(enable_persistence=False)
        
        result = cache.invalidate("nonexistent")
        assert result is False
    
    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = LLMCache(enable_persistence=False)
        
        cache.set("prompt1", "response1")
        cache.set("prompt2", "response2")
        
        cache.clear()
        
        assert cache.get("prompt1") is None
        assert cache.get("prompt2") is None
    
    def test_cache_stats(self):
        """Test getting cache statistics."""
        cache = LLMCache(enable_persistence=False)
        
        cache.set("prompt1", "response1")
        cache.get("prompt1")  # Hit
        cache.get("prompt1")  # Hit
        cache.get("unknown")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["entries"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
    
    def test_cache_max_entries_eviction(self):
        """Test that old entries are evicted when at capacity."""
        cache = LLMCache(enable_persistence=False, max_entries=5)
        
        # Fill cache
        for i in range(5):
            cache.set(f"prompt_{i}", f"response_{i}")
        
        # Add one more to trigger eviction
        cache.set("prompt_new", "response_new")
        
        # Should still have max_entries or less
        stats = cache.get_stats()
        assert stats["entries"] <= 5
    
    def test_cache_persistence_save_load(self, temp_dir):
        """Test that cache persists and loads correctly."""
        cache_path = temp_dir / "persistent_cache.db"
        
        # Create cache and add entries
        cache1 = LLMCache(storage_path=cache_path, enable_persistence=True)
        cache1.set("prompt1", "response1")
        cache1.set("prompt2", "response2")
        
        # Create new cache instance (simulates restart)
        cache2 = LLMCache(storage_path=cache_path, enable_persistence=True)
        
        # Should load persisted entries
        assert cache2.get("prompt1") == "response1"
        assert cache2.get("prompt2") == "response2"


class TestCachedLLM:
    """Tests for CachedLLM wrapper."""
    
    def test_cached_llm_calls_underlying(self):
        """Test that CachedLLM calls underlying LLM on miss."""
        call_count = 0
        
        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return f"Response to: {prompt}"
        
        cache = LLMCache(enable_persistence=False)
        cached_llm = CachedLLM(mock_llm, cache=cache)
        
        result = cached_llm("Test prompt")
        
        assert result == "Response to: Test prompt"
        assert call_count == 1
    
    def test_cached_llm_returns_cached(self):
        """Test that CachedLLM returns cached response on hit."""
        call_count = 0
        
        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return f"Response {call_count}"
        
        cache = LLMCache(enable_persistence=False)
        cached_llm = CachedLLM(mock_llm, cache=cache)
        
        result1 = cached_llm("Test prompt")
        result2 = cached_llm("Test prompt")
        
        assert result1 == result2  # Same cached response
        assert call_count == 1  # Only called once
    
    def test_cached_llm_bypass_cache(self):
        """Test bypassing cache with use_cache=False."""
        call_count = 0
        
        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return f"Response {call_count}"
        
        cache = LLMCache(enable_persistence=False)
        cached_llm = CachedLLM(mock_llm, cache=cache)
        
        result1 = cached_llm("Test prompt", use_cache=False)
        result2 = cached_llm("Test prompt", use_cache=False)
        
        assert result1 == "Response 1"
        assert result2 == "Response 2"
        assert call_count == 2
    
    def test_cached_llm_complete_method(self):
        """Test the complete() alias method."""
        def mock_llm(prompt):
            return "Response"
        
        cache = LLMCache(enable_persistence=False)
        cached_llm = CachedLLM(mock_llm, cache=cache)
        
        result = cached_llm.complete("Test prompt")
        
        assert result == "Response"


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_wrap_llm_with_cache(self):
        """Test wrap_llm_with_cache function."""
        def mock_llm(prompt):
            return "Response"
        
        cached = wrap_llm_with_cache(mock_llm, ttl_seconds=3600)
        
        assert isinstance(cached, CachedLLM)
        assert cached("Test") == "Response"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
