"""
RNSR Client - Simple High-Level API

Provides the simplest possible interface for using RNSR.
Handles all the complexity of ingestion, indexing, and navigation.

Usage:
    from rnsr import RNSRClient
    
    # One-line document Q&A
    client = RNSRClient()
    answer = client.ask("contract.pdf", "What are the payment terms?")
    
    # With caching (faster for repeated queries)
    client = RNSRClient(cache_dir="./cache")
    answer = client.ask("contract.pdf", "What are the terms?")
    answer2 = client.ask("contract.pdf", "Who are the parties?")  # Uses cache
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import structlog

from rnsr.agent import run_navigator
from rnsr.document_store import DocumentStore
from rnsr.indexing import build_skeleton_index, save_index, load_index
from rnsr.indexing.kv_store import InMemoryKVStore, KVStore
from rnsr.ingestion import ingest_document
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


class RNSRClient:
    """
    High-level client for RNSR document Q&A.
    
    This is the simplest way to use RNSR. It handles:
    - Document ingestion
    - Skeleton index building
    - Optional caching/persistence
    - Navigation and answer generation
    
    Example:
        # Basic usage (no caching)
        client = RNSRClient()
        answer = client.ask("document.pdf", "What is the main topic?")
        
        # With caching (recommended for production)
        client = RNSRClient(cache_dir="./rnsr_cache")
        answer = client.ask("document.pdf", "What is the main topic?")
        
        # From raw text
        answer = client.ask_text(
            "This is my document content...",
            "What is this about?"
        )
    """
    
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        """
        Initialize the RNSR client.
        
        Args:
            cache_dir: Optional directory for caching indexes.
                       If provided, indexes are persisted and reused.
            llm_provider: LLM provider ("openai", "anthropic", "gemini")
            llm_model: LLM model name
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
        # In-memory cache for session
        self._session_cache: dict[str, tuple[dict[str, SkeletonNode], KVStore]] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "rnsr_client_initialized",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
    
    def ask(
        self,
        document: str | Path,
        question: str,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """
        Ask a question about a PDF document.
        
        Args:
            document: Path to PDF file
            question: Question to ask
            force_reindex: If True, re-process even if cached
            
        Returns:
            Result dictionary from the navigator
            
        Example:
            answer = client.ask("contract.pdf", "What are the payment terms?")
        """
        doc_path = Path(document)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        # Get or create index
        skeleton, kv_store = self._get_or_create_index(doc_path, force_reindex)
        
        # Run navigator
        result = run_navigator(question, skeleton, kv_store)
        return result.get("answer", "No answer found.")
    
    def ask_text(
        self,
        text: str | list[str],
        question: str,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Ask a question about raw text.
        
        Args:
            text: Text content or list of text chunks
            question: Question to ask
            cache_key: Optional key for caching (if cache_dir is set)
            
        Returns:
            Answer string
            
        Example:
            answer = client.ask_text(
                "The company was founded in 2020...",
                "When was the company founded?"
            )
        """
        from rnsr.ingestion import build_tree_from_text
        
        # Generate cache key from content if not provided
        if cache_key is None:
            content = text if isinstance(text, str) else "\n".join(text)
            cache_key = f"text_{hashlib.md5(content[:1000].encode()).hexdigest()[:12]}"
        
        # Check caches
        if cache_key in self._session_cache:
            skeleton, kv_store = self._session_cache[cache_key]
        elif self.cache_dir and (self.cache_dir / cache_key).exists():
            skeleton, kv_store = load_index(self.cache_dir / cache_key)
            self._session_cache[cache_key] = (skeleton, kv_store)
        else:
            # Build index
            tree = build_tree_from_text(text)
            skeleton, kv_store = build_skeleton_index(tree)
            self._session_cache[cache_key] = (skeleton, kv_store)
            
            # Persist if cache_dir is set
            if self.cache_dir:
                save_index(skeleton, kv_store, self.cache_dir / cache_key)
        
        result = run_navigator(question, skeleton, kv_store)
        return result.get("answer", "No answer found.")
    
    def ask_multiple(
        self,
        document: str | Path,
        questions: list[str],
        force_reindex: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ask multiple questions about a document.
        
        More efficient than calling ask() multiple times because
        the document is only indexed once.
        
        Args:
            document: Path to PDF file
            questions: List of questions
            force_reindex: If True, re-process even if cached
            
        Returns:
            List of answers
            
        Example:
            answers = client.ask_multiple(
                "contract.pdf",
                ["What are the terms?", "Who are the parties?"]
            )
        """
        doc_path = Path(document)
        skeleton, kv_store = self._get_or_create_index(doc_path, force_reindex)
        
        return [
            run_navigator(q, skeleton, kv_store).get("answer", "No answer found.")
            for q in questions
        ]
    
    def get_document_info(self, document: str | Path) -> dict[str, Any]:
        """
        Get information about a document without querying it.
        
        Args:
            document: Path to PDF file
            
        Returns:
            Dictionary with document metadata
        """
        doc_path = Path(document)
        cache_key = self._get_cache_key(doc_path)
        
        # Check if cached
        if cache_key in self._session_cache:
            skeleton, _ = self._session_cache[cache_key]
            return {
                "path": str(doc_path),
                "cached": True,
                "nodes": len(skeleton),
            }
        
        if self.cache_dir and (self.cache_dir / cache_key).exists():
            from rnsr.indexing import get_index_info
            info = get_index_info(self.cache_dir / cache_key)
            info["path"] = str(doc_path)
            info["cached"] = True
            return info
        
        return {
            "path": str(doc_path),
            "cached": False,
            "exists": doc_path.exists(),
        }
    
    def clear_cache(self) -> int:
        """
        Clear all cached indexes.
        
        Returns:
            Number of caches cleared
        """
        count = len(self._session_cache)
        self._session_cache.clear()
        
        if self.cache_dir:
            import shutil
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    count += 1
        
        logger.info("cache_cleared", count=count)
        return count
    
    def _get_cache_key(self, doc_path: Path) -> str:
        """Generate a cache key for a document."""
        stat = doc_path.stat()
        hash_input = f"{doc_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _get_or_create_index(
        self,
        doc_path: Path,
        force_reindex: bool,
    ) -> tuple[dict[str, SkeletonNode], KVStore]:
        """Get cached index or create new one."""
        cache_key = self._get_cache_key(doc_path)
        
        # Check session cache first
        if not force_reindex and cache_key in self._session_cache:
            logger.debug("using_session_cache", key=cache_key)
            return self._session_cache[cache_key]
        
        # Check persistent cache
        if not force_reindex and self.cache_dir:
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                logger.debug("using_persistent_cache", key=cache_key)
                skeleton, kv_store = load_index(cache_path)
                self._session_cache[cache_key] = (skeleton, kv_store)
                return skeleton, kv_store
        
        # Create new index
        logger.info("creating_new_index", path=str(doc_path))
        result = ingest_document(str(doc_path))
        skeleton, kv_store = build_skeleton_index(result.tree)
        
        # Store in session cache
        self._session_cache[cache_key] = (skeleton, kv_store)
        
        # Persist if cache_dir is set
        if self.cache_dir:
            save_index(skeleton, kv_store, self.cache_dir / cache_key)
        
        return skeleton, kv_store
