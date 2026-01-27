"""
Ingestion Pipeline - Master Function with Enhanced Latent Hierarchy Generation

This module provides the main `ingest_document()` function that implements
the full Latent Hierarchy Generator from the research paper (Section 4-6).

TIER 1: Visual-Geometric Analysis (Primary)
    1a. PyMuPDF Font Histogram (Section 6.1)
        - If headers detected via font variance → Build hierarchical tree
    1b. Recursive XY-Cut (Section 4.1.1) - Optional for complex layouts
        - For multi-column documents, L-shaped text wraps
        
TIER 2: Semantic Boundary Detection (Fallback 1 - Flat Text)
    2a. LlamaIndex SemanticSplitterNodeParser (Section 4.2.1)
        - Embedding-based splitting at topic shifts
    2b. Hierarchical Clustering (Section 4.2.2) - Enhanced option
        - Multi-resolution: micro-clusters → macro-clusters
    2c. Synthetic Header Generation (Section 6.3)
        - LLM-generated titles for each section

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
from rnsr.ingestion.layout_detector import detect_layout_complexity
from rnsr.ingestion.ocr_fallback import has_extractable_text, try_ocr_ingestion
from rnsr.ingestion.semantic_fallback import try_semantic_splitter_ingestion
from rnsr.ingestion.tree_builder import build_document_tree
from rnsr.models import DocumentTree, IngestionResult

logger = structlog.get_logger(__name__)


def ingest_document(
    pdf_path: Path | str,
    use_visual_analysis: bool = True,
    complexity_threshold: float = 0.3,
) -> IngestionResult:
    """
    Master ingestion function implementing 3-tier graceful degradation.
    
    ALWAYS call this function - never call individual tiers directly.
    
    Ingestion Flow:
    0. Pre-analysis: Detect layout complexity (multi-column, empty pages)
    1. Tier 1a: Font Histogram (simple layouts)
    1. Tier 1b: LayoutLM + XY-Cut (complex layouts, if use_visual_analysis=True)
    2. Tier 2: Semantic Splitter (flat text, no structure)
    3. Tier 3: OCR (scanned/image-only PDFs)
    
    Args:
        pdf_path: Path to the PDF file to ingest.
        use_visual_analysis: Enable LayoutLM for complex layouts (default: True).
        complexity_threshold: Threshold for triggering visual analysis (0.0-1.0).
        
    Returns:
        IngestionResult containing the DocumentTree and metadata.
        
    Raises:
        IngestionError: If all tiers fail.
        
    Example:
        # Auto-detect layout complexity
        result = ingest_document("contract.pdf")
        
        # Force visual analysis
        result = ingest_document("report.pdf", use_visual_analysis=True)
        
        # Disable visual analysis
        result = ingest_document("simple.pdf", use_visual_analysis=False)
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
    
    # PRE-ANALYSIS: Detect layout complexity
    if use_visual_analysis:
        try:
            complexity = detect_layout_complexity(pdf_path, threshold=complexity_threshold)
            
            stats["layout_complexity"] = complexity.complexity_score
            stats["needs_visual"] = complexity.needs_visual_analysis
            stats["complexity_reason"] = complexity.reason
            
            if complexity.needs_visual_analysis:
                logger.info(
                    "complex_layout_detected",
                    path=str(pdf_path),
                    score=complexity.complexity_score,
                    reason=complexity.reason,
                )
                
                # Try visual analysis first
                result = _try_tier_1b_visual(pdf_path, warnings, stats)
                if result is not None:
                    return result
                
                # Fall through to standard font histogram if visual fails
                warnings.append(f"Visual analysis failed, using font histogram fallback")
        except Exception as e:
            logger.warning("layout_detection_failed", error=str(e))
            warnings.append(f"Layout detection failed: {e}")
    
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


def _try_tier_1b_visual(
    pdf_path: Path,
    warnings: list[str],
    stats: dict,
) -> IngestionResult | None:
    """
    TIER 1b: Try LayoutLM + XY-Cut for complex layouts.
    
    Uses visual analysis to detect document structure when
    layout is too complex for simple font histogram.
    """
    logger.debug("trying_tier_1b_visual", path=str(pdf_path))
    
    try:
        from rnsr.ingestion.layout_model import check_layout_model_available
        
        if not check_layout_model_available():
            logger.warning("layout_model_unavailable")
            warnings.append("LayoutLM not available - falling back to font histogram")
            return None
        
        from rnsr.ingestion.xy_cut import analyze_document_with_xycut
        
        # Use XY-Cut + LayoutLM for visual analysis
        tree = analyze_document_with_xycut(pdf_path)
        tree.ingestion_tier = 1
        tree.ingestion_method = "layoutlm_xycut"
        
        logger.info(
            "tier_1b_visual_success",
            path=str(pdf_path),
            nodes=tree.total_nodes,
        )
        
        return IngestionResult(
            tree=tree,
            tier_used=1,
            method="layoutlm_xycut",
            warnings=warnings,
            stats=stats,
        )
        
    except Exception as e:
        logger.warning("tier_1b_visual_failed", path=str(pdf_path), error=str(e))
        warnings.append(f"LayoutLM visual analysis failed: {e}")
        return None


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
    use_hierarchical_clustering: bool = False,
) -> IngestionResult | None:
    """
    TIER 2: Try Semantic Splitter or Hierarchical Clustering ingestion.
    
    Implements Section 4.2 of the research paper:
    - 4.2.1: SemanticSplitterNodeParser for breakpoint detection
    - 4.2.2: Hierarchical Clustering for multi-resolution topics
    - 6.3: Synthetic Header Generation via LLM
    """
    logger.debug("trying_tier_2", path=str(pdf_path))
    
    # Option: Use hierarchical clustering for richer structure
    if use_hierarchical_clustering:
        try:
            from rnsr.ingestion.hierarchical_cluster import cluster_document_hierarchically
            
            tree = cluster_document_hierarchically(pdf_path)
            
            logger.info(
                "tier_2_hierarchical_success",
                path=str(pdf_path),
                nodes=tree.total_nodes,
            )
            
            return IngestionResult(
                tree=tree,
                tier_used=2,
                method="hierarchical_clustering",
                warnings=warnings,
                stats=stats,
            )
        except Exception as e:
            logger.warning("hierarchical_clustering_failed", error=str(e))
            warnings.append(f"Hierarchical clustering failed: {e}")
            # Fall through to semantic splitter
    
    # Default: Semantic Splitter (with LLM-generated headers)
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


def ingest_document_enhanced(
    pdf_path: Path | str,
    use_xy_cut: bool = False,
    use_hierarchical_clustering: bool = False,
) -> IngestionResult:
    """
    Enhanced ingestion with all research paper features.
    
    This exposes the full Latent Hierarchy Generator from the paper:
    - XY-Cut for complex multi-column layouts (Section 4.1.1)
    - Hierarchical Clustering for multi-resolution topics (Section 4.2.2)
    - Synthetic Header Generation via LLM (Section 6.3)
    
    Args:
        pdf_path: Path to the PDF file to ingest.
        use_xy_cut: Enable Recursive XY-Cut for complex layouts.
        use_hierarchical_clustering: Use clustering instead of simple splits.
        
    Returns:
        IngestionResult containing the DocumentTree and metadata.
        
    Example:
        # For a complex multi-column PDF:
        result = ingest_document_enhanced("report.pdf", use_xy_cut=True)
        
        # For flat text that needs hierarchical structure:
        result = ingest_document_enhanced(
            "transcript.pdf",
            use_hierarchical_clustering=True
        )
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise IngestionError(f"PDF file not found: {pdf_path}")
    
    logger.info(
        "enhanced_ingestion_started",
        path=str(pdf_path),
        xy_cut=use_xy_cut,
        hierarchical=use_hierarchical_clustering,
    )
    
    warnings: list[str] = []
    stats: dict = {"path": str(pdf_path)}
    
    # Check if document has extractable text
    if not has_extractable_text(pdf_path):
        return _try_tier_3(pdf_path, warnings, stats)
    
    # Try XY-Cut first if enabled (for complex layouts)
    if use_xy_cut:
        result = _try_xy_cut_ingestion(pdf_path, warnings, stats)
        if result is not None:
            return result
    
    # TIER 1: Try Font Histogram
    result = _try_tier_1(pdf_path, warnings, stats)
    if result is not None:
        return result
    
    # TIER 2: Semantic analysis with optional hierarchical clustering
    result = _try_tier_2(pdf_path, warnings, stats, use_hierarchical_clustering)
    if result is not None:
        return result
    
    raise IngestionError("All ingestion tiers failed")


def _try_xy_cut_ingestion(
    pdf_path: Path,
    warnings: list[str],
    stats: dict,
) -> IngestionResult | None:
    """
    Optional: Use Recursive XY-Cut + LayoutLM for complex layouts.
    
    Implements Section 4.1.1:
    "A top-down page segmentation technique that is particularly 
    effective for discovering document structure."
    """
    logger.debug("trying_xy_cut_with_layoutlm", path=str(pdf_path))
    
    try:
        # Check if LayoutLM is available
        from rnsr.ingestion.layout_model import check_layout_model_available
        
        if not check_layout_model_available():
            logger.warning("layout_model_unavailable")
            warnings.append("LayoutLM not available for XY-Cut enhancement")
            return None
        
        from rnsr.ingestion.xy_cut import analyze_document_with_xycut
        
        # Use XY-Cut + LayoutLM for visual analysis
        tree = analyze_document_with_xycut(pdf_path)
        tree.ingestion_tier = 1
        tree.ingestion_method = "xy_cut_layoutlm"
        
        logger.info(
            "xy_cut_layoutlm_success",
            path=str(pdf_path),
            nodes=tree.total_nodes,
        )
        
        return IngestionResult(
            tree=tree,
            tier_used=1,
            method="xy_cut_layoutlm",
            warnings=warnings,
            stats=stats,
        )
        
    except Exception as e:
        logger.warning("xy_cut_layoutlm_failed", path=str(pdf_path), error=str(e))
        warnings.append(f"XY-Cut + LayoutLM failed: {e}")
        return None


def _try_xy_cut_ingestion_legacy(
    pdf_path: Path,
    warnings: list[str],
    stats: dict,
) -> IngestionResult | None:
    """
    Legacy XY-Cut implementation without LayoutLM.
    
    Implements Section 4.1.1:
    "A top-down page segmentation technique that is particularly 
    effective for discovering document structure."
    """
    logger.debug("trying_xy_cut", path=str(pdf_path))
    
    try:
        from rnsr.ingestion.xy_cut import RecursiveXYCutter
        import fitz
        
        cutter = RecursiveXYCutter()
        page_trees = cutter.segment_pdf(pdf_path)
        
        # Extract text for each leaf region
        doc = fitz.open(pdf_path)
        for page_num, tree in enumerate(page_trees):
            cutter.extract_text_in_regions(doc[page_num], tree)
        doc.close()
        
        # Convert XY-Cut tree to DocumentTree
        from rnsr.models import DocumentNode, DocumentTree
        
        root = DocumentNode(
            id="root",
            level=0,
            header=pdf_path.stem,
        )
        
        section_num = 0
        for page_tree in page_trees:
            for leaf in _get_xy_cut_leaves(page_tree):
                if leaf.text.strip():
                    section_num += 1
                    # Generate synthetic header
                    from rnsr.ingestion.semantic_fallback import _generate_synthetic_header
                    
                    section = DocumentNode(
                        id=f"xycut_{section_num:03d}",
                        level=1,
                        header=_generate_synthetic_header(leaf.text, section_num),
                        content=leaf.text,
                    )
                    root.children.append(section)
        
        if section_num == 0:
            warnings.append("XY-Cut found no text regions")
            return None
        
        tree = DocumentTree(
            title=pdf_path.stem,
            root=root,
            total_nodes=section_num + 1,
            ingestion_tier=1,
            ingestion_method="xy_cut",
        )
        
        logger.info("xy_cut_success", path=str(pdf_path), nodes=tree.total_nodes)
        
        return IngestionResult(
            tree=tree,
            tier_used=1,
            method="xy_cut",
            warnings=warnings,
            stats=stats,
        )
        
    except Exception as e:
        logger.warning("xy_cut_failed", path=str(pdf_path), error=str(e))
        warnings.append(f"XY-Cut failed: {e}")
        return None


def _get_xy_cut_leaves(node) -> list:
    """Get all leaf nodes from an XY-Cut segment tree."""
    if node.is_leaf:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(_get_xy_cut_leaves(child))
    return leaves


# Convenience exports
__all__ = ["ingest_document", "ingest_document_enhanced", "IngestionResult"]
