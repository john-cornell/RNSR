"""
Indexing Module - Skeleton Index Construction

Responsible for:
1. IndexNode construction (summaries only)
2. Summary generation via LLM
3. KV store for full text
"""

from rnsr.indexing.kv_store import InMemoryKVStore, KVStore, SQLiteKVStore
from rnsr.indexing.skeleton_index import (
    SkeletonIndexBuilder,
    build_skeleton_index,
    create_llama_index_nodes,
    generate_summary,
)
from rnsr.models import SkeletonNode

__all__ = [
    # KV Store
    "KVStore",
    "SQLiteKVStore",
    "InMemoryKVStore",
    # Skeleton Index
    "SkeletonIndexBuilder",
    "build_skeleton_index",
    "create_llama_index_nodes",
    "generate_summary",
    "SkeletonNode",
]
