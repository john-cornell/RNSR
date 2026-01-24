"""
Ingestion Pipeline - Master Function with 3-Tier Graceful Degradation

This module provides the main `ingest_document()` function that implements
the 3-tier fallback chain:

TIER 1: PyMuPDF Font Histogram (Primary)
    - If headers detected via font variance → Build hierarchical tree
    - If NO font variance detected → Fallback to Tier 2

TIER 2: Semantic Splitter (Fallback 1 - Flat Text)
    - Use LlamaIndex SemanticSplitterNodeParser
    - Generate "synthetic" sections based on embedding shifts
    - If document is scanned/image-only → Fallback to Tier 3

TIER 3: OCR + Re-analyze (Fallback 2 - Scanned PDFs)
    - Apply Tesseract or Doctr OCR
    - Generate text layer from images
    - Build tree from OCR output

ALWAYS call `ingest_document()` - never call individual tiers directly.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from rnsr.exceptions import IngestionError
from rnsr.ingestion.font_histogram import FontHistogramAnalyzer
from rnsr.ingestion.header_classifier import classify_headers
from rnsr.ingestion.ocr_fallback import has_extractable_text, try_ocr_ingestion
from rnsr.ingestion.semantic_fallback import try_semantic_splitter_ingestion
from rnsr.ingestion.tree_builder import build_document_tree
from rnsr.models import DocumentTree, IngestionResult

logger = structlog.get_logger(__name__)


def ingest_document(pdf_path: Path | str) -> IngestionResult:
    """
    Master ingestion function implementing 3-tier graceful degradation.
    
    ALWAYS call this function - never call individual tiers directly.
    
    Args:
        pdf_path: Path to the PDF file to ingest.
        
    Returns:
        IngestionResult containing the DocumentTree and metadata.
        
    Raises:
        IngestionError: If all tiers fail.
        
    Example:
        result = ingest_document("contract.pdf")
        print(f"Tier used: {result.tier_used}")
        print(f"Total nodes: {result.tree.total_nodes}")
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise IngestionError(f"PDF file not found: {pdf_path}")
    
    logger.info("ingestion_started", path=str(pdf_path))
    
    warnings: list[str] = []
    stats: dict = {"path": str(pdf_path)}
    
    # Check if document has extractable text
    if not has_extractable_text(pdf_path):
        # No text - go directly to Tier 3 (OCR)
        logger.info("no_extractable_text", path=str(pdf_path))
        return _try_tier_3(pdf_path, warnings, stats)
    
    # TIER 1: Try PyMuPDF Font Histogram
    result = _try_tier_1(pdf_path, warnings, stats)
    if result is not None:
        return result
    
    # TIER 2: Try Semantic Splitter
    result = _try_tier_2(pdf_path, warnings, stats)
    if result is not None:
        return result
    
    # This shouldn't happen, but just in case
    raise IngestionError("All ingestion tiers failed")


def _try_tier_1(
    pdf_path: Path,
    warnings: list[str],
    stats: dict,
) -> IngestionResult | None:
    """
    TIER 1: Try Font Histogram ingestion.
    
    Returns None if should fall back to Tier 2.
    """
    logger.debug("trying_tier_1", path=str(pdf_path))
    
    try:
        analyzer = FontHistogramAnalyzer()
        analysis, spans = analyzer.analyze(pdf_path)
        
        stats["span_count"] = len(spans)
        stats["unique_sizes"] = analysis.unique_sizes
        stats["body_size"] = analysis.body_size
        
        # Check if we have font variance
        if not analyzer.has_font_variance(analysis):
            logger.info("no_font_variance", path=str(pdf_path))
            warnings.append("No font variance detected - using semantic splitter")
            return None  # Trigger Tier 2
        
        # Check if we can detect headers
        if not analyzer.has_detectable_headers(analysis, spans):
            logger.info("no_headers_detected", path=str(pdf_path))
            warnings.append("No headers detected - using semantic splitter")
            return None  # Trigger Tier 2
        
        # Classify spans
        classified = classify_headers(spans, analysis)
        
        header_count = sum(1 for s in classified if s.role == "header")
        stats["header_count"] = header_count
        
        # Build tree
        tree = build_document_tree(classified, title=pdf_path.stem)
        tree.ingestion_tier = 1
        tree.ingestion_method = "font_histogram"
        
        logger.info(
            "tier_1_success",
            path=str(pdf_path),
            nodes=tree.total_nodes,
        )
        
        return IngestionResult(
            tree=tree,
            tier_used=1,
            method="font_histogram",
            warnings=warnings,
            stats=stats,
        )
        
    except Exception as e:
        logger.warning("tier_1_failed", path=str(pdf_path), error=str(e))
        warnings.append(f"Font histogram failed: {e}")
        return None


def _try_tier_2(
    pdf_path: Path,
    warnings: list[str],
    stats: dict,
) -> IngestionResult | None:
    """
    TIER 2: Try Semantic Splitter ingestion.
    """
    logger.debug("trying_tier_2", path=str(pdf_path))
    
    try:
        tree = try_semantic_splitter_ingestion(pdf_path)
        
        logger.info(
            "tier_2_success",
            path=str(pdf_path),
            nodes=tree.total_nodes,
        )
        
        return IngestionResult(
            tree=tree,
            tier_used=2,
            method="semantic_splitter",
            warnings=warnings,
            stats=stats,
        )
        
    except Exception as e:
        logger.warning("tier_2_failed", path=str(pdf_path), error=str(e))
        warnings.append(f"Semantic splitter failed: {e}")
        # Continue to Tier 3
        return _try_tier_3(pdf_path, warnings, stats)


def _try_tier_3(
    pdf_path: Path,
    warnings: list[str],
    stats: dict,
) -> IngestionResult:
    """
    TIER 3: Try OCR ingestion (last resort).
    """
    logger.debug("trying_tier_3", path=str(pdf_path))
    
    try:
        tree = try_ocr_ingestion(pdf_path)
        
        logger.info(
            "tier_3_success",
            path=str(pdf_path),
            nodes=tree.total_nodes,
        )
        
        return IngestionResult(
            tree=tree,
            tier_used=3,
            method="ocr",
            warnings=warnings,
            stats=stats,
        )
        
    except Exception as e:
        logger.error("tier_3_failed", path=str(pdf_path), error=str(e))
        raise IngestionError(f"All ingestion tiers failed. Last error: {e}") from e


# Convenience exports
__all__ = ["ingest_document", "IngestionResult"]
