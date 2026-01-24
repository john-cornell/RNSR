"""
RNSR - Recursive Neural-Symbolic Retriever

A document retrieval system that reconstructs document hierarchies
using font histogram analysis and navigates them with a recursive
LangGraph agent.

Key Features:
- Font Histogram Algorithm (NOT vision models)
- Skeleton Index pattern (summaries in vector index, full text in KV store)
- Pointer-based Variable Stitching (prevents context pollution)
- 3-Tier Graceful Degradation (Font → Semantic → OCR)
- Multi-provider LLM support (OpenAI, Anthropic, Gemini)

Usage:
    from rnsr import ingest_document, build_skeleton_index, run_navigator
    
    # Ingest document
    result = ingest_document("contract.pdf")
    tree = result.tree
    
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
from rnsr.indexing import build_skeleton_index, SQLiteKVStore, InMemoryKVStore
from rnsr.agent import run_navigator, VariableStore
from rnsr.llm import get_llm, get_embed_model, LLMProvider

__all__ = [
    # Version
    "__version__",
    # Ingestion
    "ingest_document",
    "IngestionResult",
    # Indexing
    "build_skeleton_index",
    "SQLiteKVStore",
    "InMemoryKVStore",
    # Agent
    "run_navigator",
    "VariableStore",
    # LLM
    "get_llm",
    "get_embed_model",
    "LLMProvider",
]
