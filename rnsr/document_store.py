"""
Document Store - Multi-Document Management

Provides a high-level interface for managing multiple indexed documents.
Handles persistence, loading, and querying across a document collection.

Usage:
    from rnsr import DocumentStore
    
    # Create or open a document store
    store = DocumentStore("./my_documents/")
    
    # Add documents
    store.add_document("contract.pdf")
    store.add_document("report.pdf", metadata={"year": 2024})
    
    # Query a specific document
    answer = store.query("contract", "What are the payment terms?")
    
    # List all documents
    for doc in store.list_documents():
        print(f"{doc['id']}: {doc['title']}")
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import structlog

from rnsr.exceptions import IndexingError
from rnsr.indexing.kv_store import KVStore, SQLiteKVStore
from rnsr.indexing.persistence import (
    save_index,
    load_index,
    get_index_info,
    delete_index,
)
from rnsr.indexing.skeleton_index import build_skeleton_index
from rnsr.ingestion import ingest_document
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


@dataclass
class DocumentInfo:
    """Information about an indexed document."""
    
    id: str
    title: str
    source_path: str | None
    node_count: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentStore:
    """
    Manages a collection of indexed documents.
    
    Provides:
    - Add/remove documents
    - Persistent storage
    - Query individual documents
    - List and search documents
    
    Example:
        store = DocumentStore("./documents/")
        store.add_document("contract.pdf")
        answer = store.query("contract", "What are the terms?")
    """
    
    def __init__(self, store_path: str | Path):
        """
        Initialize or open a document store.
        
        Args:
            store_path: Directory for storing document indexes
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self._catalog_path = self.store_path / "catalog.json"
        self._catalog: dict[str, DocumentInfo] = {}
        
        # Load existing catalog if present
        if self._catalog_path.exists():
            self._load_catalog()
        
        logger.info(
            "document_store_initialized",
            path=str(self.store_path),
            documents=len(self._catalog),
        )
    
    def _load_catalog(self) -> None:
        """Load the document catalog from disk."""
        with open(self._catalog_path) as f:
            data = json.load(f)
        
        self._catalog = {
            doc_id: DocumentInfo(**info)
            for doc_id, info in data.get("documents", {}).items()
        }
    
    def _save_catalog(self) -> None:
        """Save the document catalog to disk."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "documents": {
                doc_id: info.to_dict()
                for doc_id, info in self._catalog.items()
            }
        }
        
        with open(self._catalog_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_document(
        self,
        source: str | Path,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add and index a document.
        
        Args:
            source: Path to PDF file
            doc_id: Optional custom ID (defaults to filename hash)
            title: Optional title (defaults to filename)
            metadata: Optional metadata dictionary
            
        Returns:
            Document ID
            
        Example:
            doc_id = store.add_document("report.pdf", metadata={"year": 2024})
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise IndexingError(f"Source file not found: {source_path}")
        
        # Generate ID if not provided
        if doc_id is None:
            # Hash of filename + file size for uniqueness
            hash_input = f"{source_path.name}_{source_path.stat().st_size}"
            doc_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        # Check if already exists
        if doc_id in self._catalog:
            logger.warning("document_already_exists", doc_id=doc_id)
            return doc_id
        
        # Ingest document
        logger.info("ingesting_document", source=str(source_path))
        result = ingest_document(str(source_path))
        
        # Build skeleton index
        skeleton, kv_store = build_skeleton_index(result.tree)
        
        # Save to store
        index_path = self.store_path / doc_id
        save_index(skeleton, kv_store, index_path)
        
        # Update catalog
        info = DocumentInfo(
            id=doc_id,
            title=title or source_path.stem,
            source_path=str(source_path),
            node_count=len(skeleton),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._catalog[doc_id] = info
        self._save_catalog()
        
        logger.info(
            "document_added",
            doc_id=doc_id,
            title=info.title,
            nodes=info.node_count,
        )
        
        return doc_id
    
    def add_from_text(
        self,
        text: str | list[str],
        doc_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add and index a document from raw text.
        
        Args:
            text: Text content or list of text chunks
            doc_id: Document ID
            title: Optional title
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        from rnsr.ingestion import build_tree_from_text
        
        # Check if already exists
        if doc_id in self._catalog:
            logger.warning("document_already_exists", doc_id=doc_id)
            return doc_id
        
        # Build tree from text
        tree = build_tree_from_text(text)
        
        # Build skeleton index
        skeleton, kv_store = build_skeleton_index(tree)
        
        # Save to store
        index_path = self.store_path / doc_id
        save_index(skeleton, kv_store, index_path)
        
        # Update catalog
        info = DocumentInfo(
            id=doc_id,
            title=title or doc_id,
            source_path=None,
            node_count=len(skeleton),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._catalog[doc_id] = info
        self._save_catalog()
        
        logger.info(
            "document_added_from_text",
            doc_id=doc_id,
            title=info.title,
            nodes=info.node_count,
        )
        
        return doc_id
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if doc_id not in self._catalog:
            return False
        
        # Delete index files
        index_path = self.store_path / doc_id
        delete_index(index_path)
        
        # Remove from catalog
        del self._catalog[doc_id]
        self._save_catalog()
        
        logger.info("document_removed", doc_id=doc_id)
        return True
    
    def get_document(
        self,
        doc_id: str,
    ) -> tuple[dict[str, SkeletonNode], KVStore] | None:
        """
        Load a document's index.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (skeleton, kv_store) or None if not found
        """
        if doc_id not in self._catalog:
            return None
        
        index_path = self.store_path / doc_id
        return load_index(index_path)
    
    def query(
        self,
        doc_id: str,
        question: str,
    ) -> str:
        """
        Query a document.
        
        Args:
            doc_id: Document ID
            question: Question to ask
            
        Returns:
            Answer string
            
        Example:
            answer = store.query("contract_123", "What are the payment terms?")
        """
        from rnsr.agent import run_navigator
        
        index_result = self.get_document(doc_id)
        if index_result is None:
            raise IndexingError(f"Document not found: {doc_id}")
        
        skeleton, kv_store = index_result
        nav_result = run_navigator(question, skeleton, kv_store)
        return nav_result.get("answer", "No answer found.")
    
    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the store.
        
        Returns:
            List of document info dictionaries
        """
        return [info.to_dict() for info in self._catalog.values()]
    
    def get_document_info(self, doc_id: str) -> DocumentInfo | None:
        """
        Get information about a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentInfo or None if not found
        """
        return self._catalog.get(doc_id)
    
    def search_documents(
        self,
        query: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentInfo]:
        """
        Search documents by title or metadata.
        
        Args:
            query: Optional text to search in titles
            metadata_filter: Optional metadata key-value pairs to match
            
        Returns:
            List of matching DocumentInfo objects
        """
        results = []
        
        for info in self._catalog.values():
            # Title search
            if query and query.lower() not in info.title.lower():
                continue
            
            # Metadata filter
            if metadata_filter:
                match = all(
                    info.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                )
                if not match:
                    continue
            
            results.append(info)
        
        return results
    
    def __len__(self) -> int:
        """Number of documents in the store."""
        return len(self._catalog)
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document exists."""
        return doc_id in self._catalog
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over document IDs."""
        return iter(self._catalog.keys())
