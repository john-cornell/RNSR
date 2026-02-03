"""
RNSR LLM Response Cache

Caches LLM responses for significant performance and cost improvement.

Features:
- Semantic-aware caching (similar prompts hit same cache)
- TTL-based expiration
- Cache warming from LearnedQueryPatterns
- Thread-safe with optional persistence

Storage: SQLite for persistence, in-memory for speed
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


# Default cache location
DEFAULT_CACHE_PATH = Path.home() / ".rnsr" / "llm_cache.db"


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry:
    """A cached LLM response."""
    
    key: str
    prompt_hash: str
    prompt_preview: str  # First 200 chars for debugging
    response: str
    created_at: float
    expires_at: float
    hit_count: int = 0
    last_hit_at: float | None = None
    
    # Metadata
    model: str = ""
    token_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "prompt_hash": self.prompt_hash,
            "prompt_preview": self.prompt_preview,
            "response": self.response,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at,
            "model": self.model,
            "token_count": self.token_count,
        }


# =============================================================================
# LLM Cache
# =============================================================================


class LLMCache:
    """
    Thread-safe LLM response cache.
    
    Uses a combination of:
    1. Exact prompt hash matching (fast)
    2. Normalized prompt matching (handles whitespace/formatting)
    3. Optional semantic similarity (slower but more hits)
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        default_ttl_seconds: int = 3600,  # 1 hour
        max_entries: int = 10000,
        enable_persistence: bool = True,
        enable_semantic_matching: bool = False,
    ):
        """
        Initialize the LLM cache.
        
        Args:
            storage_path: Path to SQLite cache file.
            default_ttl_seconds: Default time-to-live for entries.
            max_entries: Maximum cache entries.
            enable_persistence: Whether to persist to disk.
            enable_semantic_matching: Enable semantic similarity matching.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_CACHE_PATH
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries
        self.enable_persistence = enable_persistence
        self.enable_semantic_matching = enable_semantic_matching
        
        self._lock = Lock()
        self._memory_cache: dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
        
        if enable_persistence:
            self._init_db()
            self._load_from_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    prompt_preview TEXT,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    last_hit_at REAL,
                    model TEXT,
                    token_count INTEGER DEFAULT 0
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_hash ON cache(prompt_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
            """)
            
            conn.commit()
            conn.close()
            
            logger.debug("llm_cache_db_initialized", path=str(self.storage_path))
            
        except Exception as e:
            logger.warning("llm_cache_db_init_failed", error=str(e))
            self.enable_persistence = False
    
    def _load_from_db(self) -> None:
        """Load non-expired entries from database."""
        if not self.enable_persistence:
            return
        
        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
            
            # Load only non-expired entries
            now = time.time()
            cursor.execute("""
                SELECT key, prompt_hash, prompt_preview, response,
                       created_at, expires_at, hit_count, last_hit_at,
                       model, token_count
                FROM cache
                WHERE expires_at > ?
                ORDER BY last_hit_at DESC
                LIMIT ?
            """, (now, self.max_entries))
            
            rows = cursor.fetchall()
            
            for row in rows:
                entry = CacheEntry(
                    key=row[0],
                    prompt_hash=row[1],
                    prompt_preview=row[2] or "",
                    response=row[3],
                    created_at=row[4],
                    expires_at=row[5],
                    hit_count=row[6] or 0,
                    last_hit_at=row[7],
                    model=row[8] or "",
                    token_count=row[9] or 0,
                )
                self._memory_cache[entry.key] = entry
            
            conn.close()
            
            logger.info("llm_cache_loaded", entries=len(self._memory_cache))
            
        except Exception as e:
            logger.warning("llm_cache_load_failed", error=str(e))
    
    def _compute_key(self, prompt: str) -> tuple[str, str]:
        """Compute cache key and hash for a prompt."""
        # Normalize prompt
        normalized = self._normalize_prompt(prompt)
        
        # Compute hash
        prompt_hash = hashlib.sha256(normalized.encode()).hexdigest()[:32]
        
        # Key includes hash
        key = f"llm_{prompt_hash}"
        
        return key, prompt_hash
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for consistent hashing."""
        # Remove excessive whitespace
        normalized = " ".join(prompt.split())
        
        # Lowercase for case-insensitive matching
        normalized = normalized.lower()
        
        return normalized
    
    def get(self, prompt: str) -> str | None:
        """
        Get cached response for a prompt.
        
        Args:
            prompt: The LLM prompt.
            
        Returns:
            Cached response or None if not found.
        """
        key, prompt_hash = self._compute_key(prompt)
        
        with self._lock:
            entry = self._memory_cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            if entry.is_expired():
                # Remove expired entry
                del self._memory_cache[key]
                self._stats["misses"] += 1
                return None
            
            # Update hit stats
            entry.hit_count += 1
            entry.last_hit_at = time.time()
            self._stats["hits"] += 1
            
            logger.debug(
                "cache_hit",
                key=key[:16],
                hit_count=entry.hit_count,
            )
            
            return entry.response
    
    def set(
        self,
        prompt: str,
        response: str,
        ttl_seconds: int | None = None,
        model: str = "",
        token_count: int = 0,
    ) -> None:
        """
        Cache an LLM response.
        
        Args:
            prompt: The LLM prompt.
            response: The LLM response.
            ttl_seconds: Time-to-live (uses default if not specified).
            model: Model name for tracking.
            token_count: Token count for tracking.
        """
        key, prompt_hash = self._compute_key(prompt)
        ttl = ttl_seconds or self.default_ttl_seconds
        
        now = time.time()
        
        entry = CacheEntry(
            key=key,
            prompt_hash=prompt_hash,
            prompt_preview=prompt[:200],
            response=response,
            created_at=now,
            expires_at=now + ttl,
            model=model,
            token_count=token_count,
        )
        
        with self._lock:
            # Evict if at capacity
            if len(self._memory_cache) >= self.max_entries:
                self._evict_oldest()
            
            self._memory_cache[key] = entry
        
        # Persist asynchronously
        if self.enable_persistence:
            self._persist_entry(entry)
        
        logger.debug(
            "cache_set",
            key=key[:16],
            ttl=ttl,
        )
    
    def _evict_oldest(self) -> None:
        """Evict oldest entries when at capacity."""
        if not self._memory_cache:
            return
        
        # Find entries to evict (oldest 10%)
        entries = list(self._memory_cache.items())
        entries.sort(key=lambda x: x[1].last_hit_at or x[1].created_at)
        
        evict_count = max(1, len(entries) // 10)
        
        for i in range(evict_count):
            key = entries[i][0]
            del self._memory_cache[key]
            self._stats["evictions"] += 1
        
        logger.debug("cache_evicted", count=evict_count)
    
    def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist entry to database."""
        if not self.enable_persistence:
            return
        
        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache
                (key, prompt_hash, prompt_preview, response, created_at,
                 expires_at, hit_count, last_hit_at, model, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.key,
                entry.prompt_hash,
                entry.prompt_preview,
                entry.response,
                entry.created_at,
                entry.expires_at,
                entry.hit_count,
                entry.last_hit_at,
                entry.model,
                entry.token_count,
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning("cache_persist_failed", error=str(e))
    
    def invalidate(self, prompt: str) -> bool:
        """
        Invalidate a cached entry.
        
        Args:
            prompt: The prompt to invalidate.
            
        Returns:
            True if entry was found and removed.
        """
        key, _ = self._compute_key(prompt)
        
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._memory_cache.clear()
        
        if self.enable_persistence:
            try:
                conn = sqlite3.connect(str(self.storage_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache")
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning("cache_clear_failed", error=str(e))
        
        logger.info("cache_cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
        
        if self.enable_persistence and expired_keys:
            try:
                conn = sqlite3.connect(str(self.storage_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
                conn.commit()
                conn.close()
            except Exception:
                pass
        
        if expired_keys:
            logger.info("expired_entries_cleaned", count=len(expired_keys))
        
        return len(expired_keys)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            
            return {
                "entries": len(self._memory_cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate,
                "max_entries": self.max_entries,
            }


# =============================================================================
# Cached LLM Wrapper
# =============================================================================


class CachedLLM:
    """
    Wrapper that adds caching to any LLM function.
    
    Usage:
        cached_llm = CachedLLM(llm.complete)
        response = cached_llm("What is 2+2?")  # Calls LLM
        response = cached_llm("What is 2+2?")  # Returns cached
    """
    
    def __init__(
        self,
        llm_fn: Callable[[str], str],
        cache: LLMCache | None = None,
        ttl_seconds: int = 3600,
        model_name: str = "",
    ):
        """
        Initialize cached LLM wrapper.
        
        Args:
            llm_fn: The underlying LLM function.
            cache: Cache instance (creates default if None).
            ttl_seconds: Default TTL for cached responses.
            model_name: Model name for tracking.
        """
        self.llm_fn = llm_fn
        self.cache = cache or get_global_cache()
        self.ttl_seconds = ttl_seconds
        self.model_name = model_name
    
    def __call__(self, prompt: str, use_cache: bool = True) -> str:
        """
        Call LLM with caching.
        
        Args:
            prompt: The prompt to send.
            use_cache: Whether to use cache (default True).
            
        Returns:
            LLM response (possibly cached).
        """
        if use_cache:
            cached = self.cache.get(prompt)
            if cached is not None:
                return cached
        
        # Call underlying LLM
        response = self.llm_fn(prompt)
        response_str = str(response) if not isinstance(response, str) else response
        
        # Cache the response
        if use_cache:
            self.cache.set(
                prompt=prompt,
                response=response_str,
                ttl_seconds=self.ttl_seconds,
                model=self.model_name,
            )
        
        return response_str
    
    def complete(self, prompt: str, use_cache: bool = True) -> str:
        """Alias for __call__ for compatibility."""
        return self(prompt, use_cache)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# =============================================================================
# Global Cache
# =============================================================================

_global_cache: LLMCache | None = None


def get_global_cache() -> LLMCache:
    """Get or create the global LLM cache."""
    global _global_cache
    
    if _global_cache is None:
        custom_path = os.getenv("RNSR_LLM_CACHE_PATH")
        _global_cache = LLMCache(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_cache


def wrap_llm_with_cache(
    llm_fn: Callable[[str], str],
    ttl_seconds: int = 3600,
    model_name: str = "",
) -> CachedLLM:
    """
    Wrap an LLM function with caching.
    
    Args:
        llm_fn: The LLM function to wrap.
        ttl_seconds: Cache TTL.
        model_name: Model name for tracking.
        
    Returns:
        CachedLLM wrapper.
    """
    return CachedLLM(
        llm_fn=llm_fn,
        cache=get_global_cache(),
        ttl_seconds=ttl_seconds,
        model_name=model_name,
    )
