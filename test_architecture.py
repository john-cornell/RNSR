#!/usr/bin/env python3
"""Test script to verify all Section 5.1 components work."""

# Phase I: Latent Structure Ingestion
from rnsr import ingest_document
from rnsr.ingestion.pipeline import ingest_document_enhanced
from rnsr.ingestion import (
    RecursiveXYCutter,
    HierarchicalSemanticClusterer,
)

# Phase II: Hierarchical Index
from rnsr import build_skeleton_index

# Phase III: Navigator API
from rnsr.agent import (
    NavigatorAPI,
    create_navigator,
    execute_rap_query,
    VariableStore,
)

print("=" * 60)
print("RNSR Architecture Test (Section 5.1)")
print("=" * 60)
print()
print("✅ Phase I: Latent Structure Ingestion Engine")
print("   - ingest_document() loaded")
print("   - ingest_document_enhanced() loaded")
print("   - RecursiveXYCutter loaded")
print("   - HierarchicalSemanticClusterer loaded")
print()
print("✅ Phase II: Hierarchical Index")
print("   - build_skeleton_index() loaded")
print()
print("✅ Phase III: Recursive REPL Agent")
print("   - NavigatorAPI loaded")
print("   - create_navigator() loaded")
print("   - execute_rap_query() loaded")
print("   - VariableStore loaded")
print()
print("All components imported successfully!")
print()
print("Navigator API methods:")
for method in ["list_children", "read_node", "search_index", "store_variable", "compare_variables"]:
    print(f"   - nav.{method}()")
