#!/usr/bin/env python3
"""
Test semantic search functionality with QuALITY dataset.

This demonstrates:
1. Building a skeleton index from a document
2. Creating a semantic searcher
3. Ranking all nodes by relevance
4. Comparing semantic search vs ToT evaluation
"""

import json
import logging
import os
import structlog
from pathlib import Path

from rnsr.ingestion.pipeline import ingest_document
from rnsr.indexing import build_skeleton_index, create_semantic_searcher
from rnsr.agent.graph import run_navigator

logger = structlog.get_logger(__name__)

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)


def load_quality_sample():
    """Load first QuALITY sample."""
    quality_path = Path.home() / ".cache" / "huggingface" / "datasets" / "quality"
    
    # Try to load from HuggingFace datasets
    try:
        from datasets import load_dataset
        dataset = load_dataset("emozilla/quality", split="train", trust_remote_code=True)
        sample = dataset[0]
        return sample["article"], sample["question"], sample["options"], sample["gold_label"]
    except Exception as e:
        logger.error("failed_to_load_quality", error=str(e))
        return None, None, None, None


def test_semantic_search():
    """Test semantic search on QuALITY dataset."""
    print("="*80)
    print("SEMANTIC SEARCH TEST - QuALITY Dataset")
    print("="*80)
    
    # Load sample
    article, question, options, gold_label = load_quality_sample()
    
    if article is None:
        print("\n❌ Failed to load QuALITY dataset")
        print("Install with: pip install datasets")
        return
    
    print(f"\n📄 Article length: {len(article)} chars")
    print(f"❓ Question: {question}")
    print(f"📋 Options: {len(options) if options else 0}")
    print(f"✅ Correct answer: {gold_label}")
    
    # Ingest document
    print("\n🔨 Building tree structure...")
    result = ingest_document(
        pdf_path=article,
    )
    
    tree = result.tree
    print(f"   Total nodes: {tree.total_nodes}")
    print(f"   Root children: {len(tree.root.child_ids)}")
    
    # Build skeleton index
    print("\n📇 Building skeleton index...")
    skeleton, kv_store = build_skeleton_index(tree)
    print(f"   Indexed nodes: {len(skeleton)}")
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    result_with = None
    try:
        searcher = create_semantic_searcher(skeleton, kv_store, provider="openai")
        
        if searcher and question:
            # Rank all leaf nodes
            all_ranked = searcher.rank_all_nodes(
                query=question,
                filter_leaves_only=True,
            )
            
            print(f"   ✅ Semantic search working!")
            print(f"   Ranked {len(all_ranked)} leaf nodes")
            print(f"\n   Top 5 most relevant nodes:")
            for i, (node, score) in enumerate(all_ranked[:5], 1):
                print(f"      {i}. {node.node_id} (score: {score:.3f})")
                print(f"         {node.summary[:100]}...")
            
            # Test with agent
            print(f"\n🤖 Running agent WITH semantic search...")
            result_with = run_navigator(
                question=question,
                skeleton=skeleton,
                kv_store=kv_store,
                use_semantic_search=True,
                semantic_searcher=searcher,
                max_iterations=20,
            )
            
            print(f"   Nodes visited: {len(result_with['nodes_visited'])}")
            print(f"   Variables: {len(result_with['variables_used'])}")
            print(f"   Answer: {result_with['answer'][:200]}...")
            
        else:
            print("   ⚠️  Semantic search unavailable (no API key or dependencies)")
            
    except Exception as e:
        print(f"   ❌ Semantic search failed: {e}")
        logger.exception("semantic_search_error")
    
    # Test without semantic search (ToT)
    print(f"\n🤖 Running agent WITHOUT semantic search (ToT)...")
    if not question:
        print("   ❌ No question available")
        return
    
    result_without = run_navigator(
        question=question,
        skeleton=skeleton,
        kv_store=kv_store,
        use_semantic_search=False,
        max_iterations=20,
    )
    
    print(f"   Nodes visited: {len(result_without['nodes_visited'])}")
    print(f"   Variables: {len(result_without['variables_used'])}")
    print(f"   Answer: {result_without['answer'][:200]}...")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"{'Method':<20} {'Nodes Visited':<15} {'Variables':<12} {'Answer Length':<15}")
    print("-"*80)
    if result_with:
        print(f"{'Semantic Search':<20} {len(result_with['nodes_visited']):<15} {len(result_with['variables_used']):<12} {len(result_with['answer']):<15}")
    else:
        print(f"{'Semantic Search':<20} {'N/A':<15} {'N/A':<12} {'N/A':<15}")
    print(f"{'Tree of Thoughts':<20} {len(result_without['nodes_visited']):<15} {len(result_without['variables_used']):<12} {len(result_without['answer']):<15}")


if __name__ == "__main__":
    import logging
    test_semantic_search()
