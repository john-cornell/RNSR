"""
Ingestion Module - Latent TOC Reconstruction

Implements the "Latent Hierarchy Generator" from the research paper (Section 4):
- Visual-Geometric Analysis (Font Histogram + XY-Cut)
- Semantic Boundary Detection (SemanticSplitter + Hierarchical Clustering)
- Synthetic Header Generation (LLM-based titles for flat documents)

Responsible for:
1. Font Histogram Analysis (PRIMARY - Section 6.1)
2. Recursive XY-Cut (Visual Segmentation - Section 4.1.1)
3. Hierarchical Clustering (Multi-resolution topics - Section 4.2.2)
4. Synthetic Header Generation (LLM titles - Section 6.3)
5. Graceful Degradation (3-tier fallback)

Primary Entry Point:
    ingest_document(pdf_path) -> IngestionResult
    
    ALWAYS use ingest_document() - it handles fallbacks automatically.
"""

from rnsr.ingestion.font_histogram import FontHistogramAnalyzer, FontAnalysis
from rnsr.ingestion.header_classifier import HeaderClassifier, classify_headers
from rnsr.ingestion.tree_builder import TreeBuilder, build_document_tree
from rnsr.ingestion.text_builder import build_tree_from_text, build_tree_from_contexts
from rnsr.ingestion.pipeline import ingest_document
from rnsr.ingestion.semantic_fallback import try_semantic_splitter_ingestion
from rnsr.ingestion.ocr_fallback import try_ocr_ingestion, check_ocr_available
from rnsr.ingestion.xy_cut import (
    RecursiveXYCutter,
    segment_pdf_with_xy_cut,
    SegmentNode,
    BoundingRegion,
)
from rnsr.ingestion.hierarchical_cluster import (
    HierarchicalSemanticClusterer,
    cluster_document_hierarchically,
    TextCluster,
)
from rnsr.ingestion.layout_detector import detect_layout_complexity, LayoutComplexity
from rnsr.ingestion.layout_model import (
    get_layout_model,
    classify_layout_blocks,
    check_layout_model_available,
    get_layout_model_info,
    LAYOUT_MODEL_BASE,
    LAYOUT_MODEL_LARGE,
)
from rnsr.models import IngestionResult

__all__ = [
    # Pipeline (Primary Entry Point)
    "ingest_document",
    "IngestionResult",
    # Tier 1: Font Histogram
    "FontHistogramAnalyzer",
    "FontAnalysis",
    "HeaderClassifier",
    "classify_headers",
    "TreeBuilder",
    "build_document_tree",
    # Text-to-Tree (for benchmarks using raw text)
    "build_tree_from_text",
    "build_tree_from_contexts",
    # Tier 1b: Visual Analysis (LayoutLM + XY-Cut)
    "detect_layout_complexity",
    "LayoutComplexity",
    "get_layout_model",
    "classify_layout_blocks",
    "check_layout_model_available",
    "get_layout_model_info",
    "LAYOUT_MODEL_BASE",
    "LAYOUT_MODEL_LARGE",
    # Tier 1b: Recursive XY-Cut (Visual Segmentation)
    "RecursiveXYCutter",
    "segment_pdf_with_xy_cut",
    "SegmentNode",
    "BoundingRegion",
    # Tier 2: Semantic Splitter
    "try_semantic_splitter_ingestion",
    # Tier 2b: Hierarchical Clustering
    "HierarchicalSemanticClusterer",
    "cluster_document_hierarchically",
    "TextCluster",
    # Tier 3: OCR
    "try_ocr_ingestion",
    "check_ocr_available",
]
