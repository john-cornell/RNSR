"""
RNSR - Recursive Neural-Symbolic Retriever

A document retrieval system that reconstructs document hierarchies
using font histogram analysis and navigates them with a recursive
LangGraph agent.

Key Features:
- Font Histogram Algorithm (Section 6.1 - NOT vision models)
- Recursive XY-Cut (Section 4.1.1 - Visual-geometric segmentation)
- Hierarchical Clustering (Section 4.2.2 - Multi-resolution topics)
- Synthetic Header Generation (Section 6.3 - LLM-generated titles)
- Skeleton Index pattern (summaries in vector index, full text in KV store)
- Pointer-based Variable Stitching (prevents context pollution)
- 3-Tier Graceful Degradation (Font → Semantic → OCR)
- Multi-provider LLM support (OpenAI, Anthropic, Gemini)

Usage:
    from rnsr import ingest_document, build_skeleton_index, run_navigator
    
    # Ingest document (standard pipeline)
    result = ingest_document("contract.pdf")
    tree = result.tree
    
    # Or use enhanced ingestion with full research paper features
    from rnsr import ingest_document_enhanced
    result = ingest_document_enhanced(
        "complex_report.pdf",
        use_xy_cut=True,  # For multi-column layouts
        use_hierarchical_clustering=True,  # For multi-resolution topics
    )
    
    # Build skeleton index
    skeleton, kv_store = build_skeleton_index(tree)
    
    # Run navigator agent
    answer = run_navigator("What are the payment terms?", skeleton, kv_store)
    
LLM Provider Configuration:
    Set one of these environment variables:
    - GOOGLE_API_KEY (Gemini)
    - OPENAI_API_KEY (OpenAI)
    - ANTHROPIC_API_KEY (Anthropic)
    
    Or configure explicitly:
        from rnsr.llm import get_llm, get_embed_model, LLMProvider
        llm = get_llm(provider=LLMProvider.GEMINI)
"""

__version__ = "0.1.0"

# Re-export main entry points
from rnsr.ingestion import ingest_document, IngestionResult
from rnsr.ingestion.pipeline import ingest_document_enhanced
from rnsr.indexing import build_skeleton_index, SQLiteKVStore, InMemoryKVStore
from rnsr.indexing import save_index, load_index, get_index_info, list_indexes
from rnsr.agent import run_navigator, VariableStore
from rnsr.document_store import DocumentStore
from rnsr.client import RNSRClient
from rnsr.llm import get_llm, get_embed_model, LLMProvider

__all__ = [
    # Version
    "__version__",
    # High-Level Client (Simplest API)
    "RNSRClient",
    # Ingestion
    "ingest_document",
    "ingest_document_enhanced",  # Full research paper implementation
    "IngestionResult",
    # Indexing
    "build_skeleton_index",
    "SQLiteKVStore",
    "InMemoryKVStore",
    # Persistence
    "save_index",
    "load_index",
    "get_index_info",
    "list_indexes",
    # Document Store
    "DocumentStore",
    # Agent
    "run_navigator",
    "VariableStore",
    # LLM
    "get_llm",
    "get_embed_model",
    "LLMProvider",
]
