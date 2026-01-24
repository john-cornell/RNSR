"""
Ingestion Module - Latent TOC Reconstruction

Responsible for:
1. Font Histogram Analysis (PRIMARY)
2. Header Classification
3. Tree Assembly
4. Graceful Degradation (3-tier fallback)

Primary Entry Point:
    ingest_document(pdf_path) -> IngestionResult
    
    ALWAYS use ingest_document() - it handles fallbacks automatically.
"""

from rnsr.ingestion.font_histogram import FontHistogramAnalyzer, FontAnalysis
from rnsr.ingestion.header_classifier import HeaderClassifier, classify_headers
from rnsr.ingestion.tree_builder import TreeBuilder, build_document_tree
from rnsr.ingestion.pipeline import ingest_document
from rnsr.ingestion.semantic_fallback import try_semantic_splitter_ingestion
from rnsr.ingestion.ocr_fallback import try_ocr_ingestion, check_ocr_available
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
    # Tier 2: Semantic Splitter
    "try_semantic_splitter_ingestion",
    # Tier 3: OCR
    "try_ocr_ingestion",
    "check_ocr_available",
]
