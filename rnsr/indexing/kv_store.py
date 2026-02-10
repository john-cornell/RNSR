"""
KV Store - SQLite-Backed Key-Value Storage for Full Text

The Skeleton Index pattern requires separating:
- Summaries (stored in vector index for retrieval)
- Full Text (stored externally in this KV Store)

This prevents full text from polluting the LLM context until
explicitly requested during synthesis.

Usage:
    kv = SQLiteKVStore("./data/document_kv.db")
    kv.put("node_123", "Full text content here...")
    content = kv.get("node_123")
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

import structlog

from rnsr.exceptions import IndexingError

logger = structlog.get_logger(__name__)


class SQLiteKVStore:
    """
    SQLite-backed key-value store for document content.
    
    Stores full text content separately from the vector index,
    allowing the skeleton index to contain only summaries.
    
    Attributes:
        db_path: Path to the SQLite database file.
    """
    
    def __init__(self, db_path: Path | str):
        """
        Initialize the KV store.
        
        Args:
            db_path: Path to the SQLite database file.
                     Will be created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        
        logger.info("kv_store_initialized", db_path=str(self.db_path))
    
    def _init_db(self) -> None:
        """Create the database schema if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash 
                ON documents(content_hash)
            """)
            
            conn.commit()
    
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def put(self, node_id: str, content: str) -> str:
        """
        Store content for a node.
        
        Args:
            node_id: Unique identifier for the node.
            content: Full text content to store.
            
        Returns:
            SHA256 hash of the content.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        char_count = len(content)
        
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (node_id, content, content_hash, char_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    char_count = excluded.char_count,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (node_id, content, content_hash, char_count),
            )
            conn.commit()
        
        logger.debug(
            "kv_put",
            node_id=node_id,
            char_count=char_count,
            hash=content_hash,
        )
        
        return content_hash
    
    def get(self, node_id: str) -> str | None:
        """
        Retrieve content for a node.
        
        Args:
            node_id: Unique identifier for the node.
            
        Returns:
            Full text content, or None if not found.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT content FROM documents WHERE node_id = ?",
                (node_id,),
            )
            row = cursor.fetchone()
        
        if row is None:
            logger.debug("kv_miss", node_id=node_id)
            return None
        
        logger.debug("kv_hit", node_id=node_id)
        return row["content"]
    
    def get_batch(self, node_ids: list[str]) -> dict[str, str | None]:
        """
        Retrieve content for multiple nodes.
        
        Args:
            node_ids: List of node identifiers.
            
        Returns:
            Dictionary mapping node_id to content (or None if not found).
        """
        result: dict[str, str | None] = {nid: None for nid in node_ids}
        
        if not node_ids:
            return result
        
        placeholders = ",".join("?" * len(node_ids))
        
        with self._connect() as conn:
            cursor = conn.execute(
                f"SELECT node_id, content FROM documents WHERE node_id IN ({placeholders})",
                node_ids,
            )
            for row in cursor:
                result[row["node_id"]] = row["content"]
        
        found = sum(1 for v in result.values() if v is not None)
        logger.debug("kv_batch_get", requested=len(node_ids), found=found)
        
        return result
    
    def delete(self, node_id: str) -> bool:
        """
        Delete content for a node.
        
        Args:
            node_id: Unique identifier for the node.
            
        Returns:
            True if deleted, False if not found.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE node_id = ?",
                (node_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        
        logger.debug("kv_delete", node_id=node_id, deleted=deleted)
        return deleted
    
    def exists(self, node_id: str) -> bool:
        """Check if a node exists in the store."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM documents WHERE node_id = ? LIMIT 1",
                (node_id,),
            )
            return cursor.fetchone() is not None
    
    def count(self) -> int:
        """Get the total number of stored documents."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]
    
    def get_metadata(self, node_id: str) -> dict | None:
        """
        Get metadata about a stored document.
        
        Args:
            node_id: Unique identifier for the node.
            
        Returns:
            Dictionary with hash, char_count, timestamps, or None.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT content_hash, char_count, created_at, updated_at 
                FROM documents WHERE node_id = ?
                """,
                (node_id,),
            )
            row = cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            "content_hash": row["content_hash"],
            "char_count": row["char_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    
    def put_image(self, node_id: str, image_bytes: bytes) -> None:
        """
        Store image bytes associated with a node.
        
        Uses a separate table so existing text queries are unaffected.
        """
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_images (
                    node_id TEXT PRIMARY KEY,
                    image_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                """
                INSERT INTO node_images (node_id, image_data)
                VALUES (?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    image_data = excluded.image_data
                """,
                (node_id, image_bytes),
            )
            conn.commit()
        logger.debug("kv_put_image", node_id=node_id, size=len(image_bytes))
    
    def get_image(self, node_id: str) -> bytes | None:
        """
        Retrieve image bytes for a node.
        
        Returns:
            Image bytes, or None if no image stored for this node.
        """
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    "SELECT image_data FROM node_images WHERE node_id = ?",
                    (node_id,),
                )
                row = cursor.fetchone()
            except Exception:
                # Table may not exist yet
                return None
        
        if row is None:
            return None
        return row["image_data"]
    
    def clear(self) -> int:
        """
        Delete all documents from the store.
        
        Returns:
            Number of documents deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents")
            count = cursor.rowcount
            try:
                conn.execute("DELETE FROM node_images")
            except Exception:
                pass  # Table may not exist
            conn.commit()
        
        logger.warning("kv_store_cleared", count=count)
        return count


class InMemoryKVStore:
    """
    In-memory key-value store for testing and ephemeral usage.
    
    API-compatible with SQLiteKVStore.
    Supports optional image storage per node for vision-augmented navigation.
    """
    
    def __init__(self):
        self._store: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}
        self._images: dict[str, bytes] = {}
    
    def put(self, node_id: str, content: str) -> str:
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        self._store[node_id] = content
        self._metadata[node_id] = {
            "content_hash": content_hash,
            "char_count": len(content),
        }
        return content_hash
    
    def get(self, node_id: str) -> str | None:
        return self._store.get(node_id)
    
    def get_batch(self, node_ids: list[str]) -> dict[str, str | None]:
        return {nid: self._store.get(nid) for nid in node_ids}
    
    def put_image(self, node_id: str, image_bytes: bytes) -> None:
        """Store image bytes associated with a node."""
        self._images[node_id] = image_bytes
    
    def get_image(self, node_id: str) -> bytes | None:
        """Retrieve image bytes for a node, or None if no image."""
        return self._images.get(node_id)
    
    def delete(self, node_id: str) -> bool:
        if node_id in self._store:
            del self._store[node_id]
            del self._metadata[node_id]
            self._images.pop(node_id, None)
            return True
        return False
    
    def exists(self, node_id: str) -> bool:
        return node_id in self._store
    
    def count(self) -> int:
        return len(self._store)
    
    def get_metadata(self, node_id: str) -> dict | None:
        return self._metadata.get(node_id)
    
    def clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        self._metadata.clear()
        self._images.clear()
        return count


# Type alias for either store
KVStore = Union[SQLiteKVStore, InMemoryKVStore]
