"""
Layout Complexity Detector - Auto-detect when to use visual analysis

Analyzes document layout characteristics to determine when LayoutLM
visual analysis should be triggered:

- Multi-column layouts (text bboxes overlap vertically)
- Empty/image-only pages (no extractable text)
- Complex L-shaped wraps (irregular bounding box patterns)

Usage:
    from rnsr.ingestion.layout_detector import detect_layout_complexity
    
    complexity = detect_layout_complexity("document.pdf")
    
    if complexity.needs_visual_analysis:
        # Use LayoutLM + XY-Cut
        pass
    else:
        # Use simple font histogram
        pass
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import structlog

from rnsr.models import SpanInfo, BoundingBox

logger = structlog.get_logger(__name__)


@dataclass
class LayoutComplexity:
    """Result of layout complexity analysis."""
    
    # Detection flags
    has_multi_column: bool = False
    has_empty_pages: bool = False
    has_complex_wrapping: bool = False
    
    # Metrics
    avg_columns_per_page: float = 1.0
    empty_page_ratio: float = 0.0
    bbox_overlap_score: float = 0.0
    
    # Overall assessment
    complexity_score: float = 0.0  # 0.0 (simple) to 1.0 (complex)
    needs_visual_analysis: bool = False
    
    # Reasoning
    reason: str = ""


def detect_multi_column(spans: list[SpanInfo], page_height: float) -> bool:
    """
    Detect if page has multi-column layout.
    
    Algorithm:
    1. Group spans by vertical position (Y coordinate)
    2. For each row, count distinct horizontal regions (columns)
    3. If >30% of rows have 2+ columns, it's multi-column
    
    Args:
        spans: List of text spans from a page.
        page_height: Height of the page.
        
    Returns:
        True if multi-column layout detected.
    """
    if len(spans) < 10:
        return False
    
    # Group spans by vertical bands (rows)
    row_height = 20.0  # Approximate line height
    rows: dict[int, list[SpanInfo]] = {}
    
    for span in spans:
        row_idx = int(span.bbox.y0 / row_height)
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append(span)
    
    # Count columns per row
    multi_column_rows = 0
    total_rows = len(rows)
    
    for row_spans in rows.values():
        if len(row_spans) < 2:
            continue
        
        # Sort by X position
        sorted_spans = sorted(row_spans, key=lambda s: s.bbox.x0)
        
        # Check for gaps indicating columns
        gaps = []
        for i in range(len(sorted_spans) - 1):
            gap = sorted_spans[i + 1].bbox.x0 - sorted_spans[i].bbox.x1
            if gap > 30:  # Significant gap
                gaps.append(gap)
        
        # If we have 1+ large gap, it's multi-column
        if gaps and max(gaps) > 50:
            multi_column_rows += 1
    
    # Threshold: >30% rows are multi-column
    if total_rows > 0:
        ratio = multi_column_rows / total_rows
        is_multi_column = ratio > 0.3
        
        logger.debug(
            "multi_column_detection",
            multi_column_rows=multi_column_rows,
            total_rows=total_rows,
            ratio=ratio,
            result=is_multi_column,
        )
        
        return is_multi_column
    
    return False


def detect_empty_pages(pdf_path: Path | str, min_text_threshold: int = 10) -> tuple[bool, float]:
    """
    Detect if document has empty or image-only pages.
    
    Args:
        pdf_path: Path to PDF file.
        min_text_threshold: Minimum word count to consider page non-empty.
        
    Returns:
        Tuple of (has_empty_pages, empty_page_ratio).
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error("pdf_open_failed", path=str(pdf_path), error=str(e))
        return False, 0.0
    
    empty_pages = 0
    total_pages = len(doc)
    
    for page in doc:
        text = page.get_text().strip()  # type: ignore[union-attr]
        word_count = len(text.split())
        
        if word_count < min_text_threshold:
            empty_pages += 1
            logger.debug("empty_page_detected", page_num=page.number, words=word_count)
    
    doc.close()
    
    if total_pages > 0:
        ratio = empty_pages / total_pages
        has_empty = ratio > 0.1  # >10% pages empty
        
        logger.info(
            "empty_page_detection",
            empty_pages=empty_pages,
            total_pages=total_pages,
            ratio=ratio,
            result=has_empty,
        )
        
        return has_empty, ratio
    
    return False, 0.0


def calculate_bbox_overlap_score(spans: list[SpanInfo]) -> float:
    """
    Calculate how much text bounding boxes overlap vertically.
    
    High overlap suggests complex wrapping or multi-column layout.
    
    Args:
        spans: List of text spans.
        
    Returns:
        Overlap score from 0.0 (no overlap) to 1.0 (high overlap).
    """
    if len(spans) < 2:
        return 0.0
    
    overlap_count = 0
    total_comparisons = 0
    
    # Compare each span with others
    for i, span1 in enumerate(spans):
        for span2 in spans[i + 1:]:
            total_comparisons += 1
            
            # Check if Y ranges overlap
            y1_min, y1_max = span1.bbox.y0, span1.bbox.y1
            y2_min, y2_max = span2.bbox.y0, span2.bbox.y1
            
            # Check for vertical overlap
            if not (y1_max < y2_min or y2_max < y1_min):
                # Check if they're in different horizontal regions
                x1_center = (span1.bbox.x0 + span1.bbox.x1) / 2
                x2_center = (span2.bbox.x0 + span2.bbox.x1) / 2
                
                if abs(x1_center - x2_center) > 100:  # Separated horizontally
                    overlap_count += 1
    
    if total_comparisons > 0:
        score = overlap_count / total_comparisons
        logger.debug("bbox_overlap_calculated", score=score, overlaps=overlap_count)
        return score
    
    return 0.0


def detect_layout_complexity(
    pdf_path: Path | str,
    threshold: float = 0.3,
) -> LayoutComplexity:
    """
    Analyze document layout to determine if visual analysis is needed.
    
    Args:
        pdf_path: Path to PDF file.
        threshold: Complexity threshold (0.0-1.0) for triggering visual analysis.
        
    Returns:
        LayoutComplexity object with analysis results.
        
    Example:
        complexity = detect_layout_complexity("report.pdf")
        
        if complexity.needs_visual_analysis:
            print(f"Reason: {complexity.reason}")
            # Use LayoutLM + XY-Cut
        else:
            # Use simple font histogram
    """
    pdf_path = Path(pdf_path)
    
    logger.info("detecting_layout_complexity", path=pdf_path.name)
    
    # Initialize result
    result = LayoutComplexity()
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error("pdf_open_failed", error=str(e))
        return result
    
    # 1. Check for empty pages
    has_empty, empty_ratio = detect_empty_pages(pdf_path)
    result.has_empty_pages = has_empty
    result.empty_page_ratio = empty_ratio
    
    # 2. Analyze first few pages for multi-column and overlap
    pages_to_check = min(3, len(doc))
    multi_column_pages = 0
    total_overlap_score = 0.0
    
    for page_num in range(pages_to_check):
        page = doc[page_num]
        
        # Extract spans
        page_dict = page.get_text("dict")  # type: ignore[assignment]
        spans: list[SpanInfo] = []
        
        for block in page_dict.get("blocks", []):  # type: ignore[union-attr]
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                for span in line["spans"]:
                    text = str(span.get("text", "")).strip()
                    if len(text) < 2:
                        continue
                    
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    font_size = float(span.get("size", 12.0))
                    
                    spans.append(SpanInfo(
                        text=text,
                        font_size=font_size,
                        font_name=str(span.get("font", "Unknown")),
                        is_bold=False,
                        is_italic=False,
                        bbox=BoundingBox(
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                        ),
                        page_num=page_num,
                    ))
        
        # Check multi-column
        if detect_multi_column(spans, page.rect.height):
            multi_column_pages += 1
        
        # Calculate overlap
        overlap = calculate_bbox_overlap_score(spans)
        total_overlap_score += overlap
    
    doc.close()
    
    # 3. Calculate metrics
    if pages_to_check > 0:
        result.avg_columns_per_page = (multi_column_pages / pages_to_check) * 2
        result.bbox_overlap_score = total_overlap_score / pages_to_check
    
    result.has_multi_column = multi_column_pages >= 2 or result.bbox_overlap_score > 0.2
    result.has_complex_wrapping = result.bbox_overlap_score > 0.3
    
    # 4. Calculate overall complexity score
    complexity_factors = []
    
    if result.has_multi_column:
        complexity_factors.append(0.5)
    if result.has_empty_pages:
        complexity_factors.append(0.3)
    if result.has_complex_wrapping:
        complexity_factors.append(0.4)
    
    # Weighted average
    if complexity_factors:
        result.complexity_score = sum(complexity_factors) / len(complexity_factors)
    
    # 5. Determine if visual analysis needed
    result.needs_visual_analysis = result.complexity_score > threshold
    
    # 6. Generate reasoning
    reasons = []
    if result.has_multi_column:
        reasons.append("multi-column layout detected")
    if result.has_empty_pages:
        reasons.append(f"{result.empty_page_ratio:.0%} pages are empty/image-only")
    if result.has_complex_wrapping:
        reasons.append("complex text wrapping detected")
    
    if reasons:
        result.reason = "; ".join(reasons)
    else:
        result.reason = "simple single-column layout"
    
    logger.info(
        "layout_complexity_detected",
        score=result.complexity_score,
        needs_visual=result.needs_visual_analysis,
        reason=result.reason,
    )
    
    return result
