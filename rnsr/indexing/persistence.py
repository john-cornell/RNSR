"""
Persistence Module - Save and Load RNSR Indexes

Provides functionality to persist and restore:
- Skeleton Index (SkeletonNode structures)
- KV Store (already SQLite-backed, but needs export/import)
- Document metadata

Usage:
    from rnsr.indexing import save_index, load_index
    
    # Save after indexing
    skeleton, kv_store = build_skeleton_index(tree)
    save_index(skeleton, kv_store, "./my_document_index/")
    
    # Load later (no re-processing needed!)
    skeleton, kv_store = load_index("./my_document_index/")
    answer = run_navigator("question", skeleton, kv_store)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from rnsr.exceptions import IndexingError
from rnsr.indexing.kv_store import InMemoryKVStore, KVStore, SQLiteKVStore
from rnsr.models import DetectedTable, SkeletonNode

logger = structlog.get_logger(__name__)

# Version for format compatibility
INDEX_FORMAT_VERSION = "1.1"  # Updated for tables support


def save_index(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    index_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    tables: list[DetectedTable] | None = None,
) -> Path:
    """
    Save a skeleton index, KV store, and detected tables to disk.
    
    Creates a directory structure:
        index_dir/
            manifest.json      # Version, metadata, timestamps
            skeleton.json      # SkeletonNode structures
            content.db         # SQLite KV store (copied or created)
            tables.json        # Detected tables (if any)
    
    Args:
        skeleton: Dictionary of node_id -> SkeletonNode
        kv_store: KV store containing full text
        index_dir: Directory to save the index
        metadata: Optional metadata (title, source, etc.)
        tables: Optional list of DetectedTable objects from ingestion
    
    Returns:
        Path to the index directory
        
    Example:
        skeleton, kv = build_skeleton_index(tree)
        save_index(skeleton, kv, "./indexes/contract_2024/", tables=result.tables)
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Build manifest
    manifest = {
        "version": INDEX_FORMAT_VERSION,
        "created_at": datetime.now().isoformat(),
        "node_count": len(skeleton),
        "table_count": len(tables) if tables else 0,
        "metadata": metadata or {},
    }
    
    # Find root node for extra info
    root = next((n for n in skeleton.values() if n.level == 0), None)
    if root:
        manifest["root_id"] = root.node_id
        manifest["root_header"] = root.header
    
    # Save manifest
    manifest_path = index_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save skeleton nodes
    skeleton_path = index_path / "skeleton.json"
    skeleton_data = {
        node_id: _skeleton_node_to_dict(node)
        for node_id, node in skeleton.items()
    }
    with open(skeleton_path, "w") as f:
        json.dump(skeleton_data, f, indent=2)
    
    # Save detected tables
    if tables:
        tables_path = index_path / "tables.json"
        tables_data = [_detected_table_to_dict(t) for t in tables]
        with open(tables_path, "w") as f:
            json.dump(tables_data, f, indent=2)
    
    # Handle KV store
    content_path = index_path / "content.db"
    
    if isinstance(kv_store, SQLiteKVStore):
        # Copy the existing database
        if kv_store.db_path != content_path:
            shutil.copy2(kv_store.db_path, content_path)
    elif isinstance(kv_store, InMemoryKVStore):
        # Export in-memory store to SQLite
        sqlite_kv = SQLiteKVStore(content_path)
        for node_id in skeleton.keys():
            content = kv_store.get(node_id)
            if content:
                sqlite_kv.put(node_id, content)
    else:
        raise IndexingError(f"Unsupported KV store type: {type(kv_store)}")
    
    logger.info(
        "index_saved",
        path=str(index_path),
        nodes=len(skeleton),
        tables=len(tables) if tables else 0,
    )
    
    return index_path


def load_index(
    index_dir: str | Path,
    in_memory: bool = False,
) -> tuple[dict[str, SkeletonNode], KVStore, list[DetectedTable]]:
    """
    Load a skeleton index, KV store, and detected tables from disk.
    
    Args:
        index_dir: Directory containing the saved index
        in_memory: If True, load KV store into memory (faster but uses more RAM)
    
    Returns:
        Tuple of (skeleton dict, kv_store, tables list)
        
    Example:
        skeleton, kv, tables = load_index("./indexes/contract_2024/")
        answer = run_navigator("What are the payment terms?", skeleton, kv)
    """
    index_path = Path(index_dir)
    
    if not index_path.exists():
        raise IndexingError(f"Index directory not found: {index_path}")
    
    # Load and validate manifest
    manifest_path = index_path / "manifest.json"
    if not manifest_path.exists():
        raise IndexingError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    version = manifest.get("version", "unknown")
    # Accept both 1.0 and 1.1 versions (1.0 just won't have tables)
    if version not in (INDEX_FORMAT_VERSION, "1.0"):
        logger.warning(
            "index_version_mismatch",
            expected=INDEX_FORMAT_VERSION,
            found=version,
        )
    
    # Load skeleton nodes
    skeleton_path = index_path / "skeleton.json"
    if not skeleton_path.exists():
        raise IndexingError(f"Skeleton index not found: {skeleton_path}")
    
    with open(skeleton_path) as f:
        skeleton_data = json.load(f)
    
    skeleton: dict[str, SkeletonNode] = {
        node_id: _dict_to_skeleton_node(data)
        for node_id, data in skeleton_data.items()
    }
    
    # Load detected tables (optional - may not exist for older indexes)
    tables: list[DetectedTable] = []
    tables_path = index_path / "tables.json"
    if tables_path.exists():
        with open(tables_path) as f:
            tables_data = json.load(f)
        tables = [_dict_to_detected_table(t) for t in tables_data]
    
    # Load KV store
    content_path = index_path / "content.db"
    if not content_path.exists():
        raise IndexingError(f"Content database not found: {content_path}")
    
    if in_memory:
        # Load into memory for faster access
        sqlite_kv = SQLiteKVStore(content_path)
        kv_store = InMemoryKVStore()
        for node_id in skeleton.keys():
            content = sqlite_kv.get(node_id)
            if content:
                kv_store.put(node_id, content)
    else:
        # Use SQLite directly
        kv_store = SQLiteKVStore(content_path)
    
    logger.info(
        "index_loaded",
        path=str(index_path),
        nodes=len(skeleton),
        tables=len(tables),
        version=version,
    )
    
    return skeleton, kv_store, tables


def get_index_info(index_dir: str | Path) -> dict[str, Any]:
    """
    Get information about a saved index without loading it.
    
    Args:
        index_dir: Directory containing the saved index
        
    Returns:
        Dictionary with index metadata
        
    Example:
        info = get_index_info("./indexes/contract_2024/")
        print(f"Index has {info['node_count']} nodes")
    """
    index_path = Path(index_dir)
    manifest_path = index_path / "manifest.json"
    
    if not manifest_path.exists():
        raise IndexingError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Add file size info
    content_path = index_path / "content.db"
    if content_path.exists():
        manifest["content_size_bytes"] = content_path.stat().st_size
    
    skeleton_path = index_path / "skeleton.json"
    if skeleton_path.exists():
        manifest["skeleton_size_bytes"] = skeleton_path.stat().st_size
    
    return manifest


def delete_index(index_dir: str | Path) -> bool:
    """
    Delete a saved index.
    
    Args:
        index_dir: Directory containing the saved index
        
    Returns:
        True if deleted, False if not found
    """
    index_path = Path(index_dir)
    
    if not index_path.exists():
        return False
    
    shutil.rmtree(index_path)
    logger.info("index_deleted", path=str(index_path))
    return True


def list_indexes(base_dir: str | Path) -> list[dict[str, Any]]:
    """
    List all indexes in a directory.
    
    Args:
        base_dir: Directory to search for indexes
        
    Returns:
        List of index info dictionaries
        
    Example:
        indexes = list_indexes("./indexes/")
        for idx in indexes:
            print(f"{idx['path']}: {idx['node_count']} nodes")
    """
    base_path = Path(base_dir)
    indexes = []
    
    if not base_path.exists():
        return indexes
    
    for item in base_path.iterdir():
        if item.is_dir() and (item / "manifest.json").exists():
            try:
                info = get_index_info(item)
                info["path"] = str(item)
                indexes.append(info)
            except Exception as e:
                logger.warning("failed_to_read_index", path=str(item), error=str(e))
    
    return indexes


# =============================================================================
# Serialization Helpers
# =============================================================================

def _skeleton_node_to_dict(node: SkeletonNode) -> dict[str, Any]:
    """Convert SkeletonNode to JSON-serializable dict."""
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "level": node.level,
        "header": node.header,
        "summary": node.summary,
        "child_ids": node.child_ids,
        "page_num": node.page_num,
        "metadata": node.metadata,
    }


def _dict_to_skeleton_node(data: dict[str, Any]) -> SkeletonNode:
    """Convert dict back to SkeletonNode."""
    return SkeletonNode(
        node_id=data["node_id"],
        parent_id=data.get("parent_id"),
        level=data["level"],
        header=data.get("header", ""),
        summary=data.get("summary", ""),
        child_ids=data.get("child_ids", []),
        page_num=data.get("page_num"),
        metadata=data.get("metadata", {}),
    )


def _detected_table_to_dict(table: DetectedTable) -> dict[str, Any]:
    """Convert DetectedTable to JSON-serializable dict."""
    return {
        "id": table.id,
        "node_id": table.node_id,
        "page_num": table.page_num,
        "title": table.title,
        "headers": table.headers,
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "data": table.data,
    }


def _dict_to_detected_table(data: dict[str, Any]) -> DetectedTable:
    """Convert dict back to DetectedTable."""
    return DetectedTable(
        id=data["id"],
        node_id=data["node_id"],
        page_num=data.get("page_num"),
        title=data.get("title", ""),
        headers=data.get("headers", []),
        num_rows=data.get("num_rows", 0),
        num_cols=data.get("num_cols", 0),
        data=data.get("data", []),
    )
