"""
Indexing Module - Skeleton Index Construction and Knowledge Graph

Responsible for:
1. IndexNode construction (summaries only)
2. Summary generation via LLM
3. KV store for full text
4. Persistence (save/load indexes)
5. Semantic search for O(log N) retrieval
6. Knowledge graph for entity and relationship storage
"""

from rnsr.indexing.kv_store import InMemoryKVStore, KVStore, SQLiteKVStore
from rnsr.indexing.skeleton_index import (
    SkeletonIndexBuilder,
    build_skeleton_index,
    create_llama_index_nodes,
    generate_summary,
)
from rnsr.indexing.persistence import (
    save_index,
    load_index,
    get_index_info,
    delete_index,
    list_indexes,
)
from rnsr.indexing.semantic_search import (
    SemanticSearcher,
    create_semantic_searcher,
)
from rnsr.indexing.knowledge_graph import (
    KnowledgeGraph,
    InMemoryKnowledgeGraph,
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
    # Persistence
    "save_index",
    "load_index",
    "get_index_info",
    "delete_index",
    "list_indexes",
    # Semantic Search
    "SemanticSearcher",
    "create_semantic_searcher",
    # Knowledge Graph
    "KnowledgeGraph",
    "InMemoryKnowledgeGraph",
]
