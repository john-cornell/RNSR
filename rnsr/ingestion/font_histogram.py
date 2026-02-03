"""
Font Histogram Analyzer - PRIMARY Ingestion Method

This module implements the Font Histogram Algorithm for Latent TOC reconstruction.
It replaces heavy vision models (LayoutLM) with a lightweight, heuristic approach.

The algorithm:
1. Iterates through ALL text spans using page.get_text("dict")
2. Collects font sizes and styles into a frequency histogram
3. Identifies "Body Text" size (most frequent = mode)
4. Infers Headers: font_size > body_text_size + threshold
5. Builds hierarchy using font size magnitude (24pt = H1, 18pt = H2, etc.)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mode, stdev
from typing import Any, cast

import fitz  # PyMuPDF
import structlog

from rnsr.exceptions import FontAnalysisError
from rnsr.models import BoundingBox, FontAnalysis, SpanInfo

logger = structlog.get_logger(__name__)


class FontHistogramAnalyzer:
    """
    Analyzes PDF font statistics to identify document structure.
    
    This is the PRIMARY method for Latent TOC reconstruction.
    Do NOT use vision models - use this font-based approach.
    """

    def __init__(
        self,
        min_text_length: int = 2,
        size_rounding: int = 1,
    ):
        """
        Initialize the Font Histogram Analyzer.
        
        Args:
            min_text_length: Minimum text length to include in analysis.
            size_rounding: Decimal places for font size rounding.
        """
        self.min_text_length = min_text_length
        self.size_rounding = size_rounding

    def extract_all_spans(self, pdf_path: Path | str) -> list[SpanInfo]:
        """
        Extract all text spans with font metadata from a PDF.
        
        Uses PyMuPDF's page.get_text("dict") to access span-level information.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of SpanInfo objects with font metadata.
            
        Raises:
            FontAnalysisError: If PDF cannot be opened or parsed.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FontAnalysisError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise FontAnalysisError(f"Failed to open PDF: {e}") from e
        
        all_spans: list[SpanInfo] = []
        
        logger.debug("extracting_spans", path=str(pdf_path), pages=len(doc))
        
        for page in doc:
            try:
                # MUST use "dict" mode to get font information
                page_dict = cast(dict[str, Any], page.get_text(
                    "dict", 
                    flags=fitz.TEXT_PRESERVE_WHITESPACE
                ))
                blocks: list[dict[str, Any]] = page_dict.get("blocks", [])
                
                for block in blocks:
                    # Skip image blocks
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = str(span.get("text", "")).strip()
                            
                            # Skip very short text
                            if len(text) < self.min_text_length:
                                continue
                            
                            # Extract font properties
                            font_name = str(span.get("font", ""))
                            flags = int(span.get("flags", 0))
                            is_bold = (
                                "Bold" in font_name 
                                or "bold" in font_name.lower()
                                or bool(flags & 2 ** 4)  # fitz bold flag
                            )
                            is_italic = (
                                "Italic" in font_name 
                                or "italic" in font_name.lower()
                                or bool(flags & 2 ** 1)  # fitz italic flag
                            )
                            
                            font_size = float(span.get("size", 12.0))
                            bbox = span.get("bbox", [0.0, 0.0, 0.0, 0.0])
                            page_num = page.number if page.number is not None else 0
                            
                            span_info = SpanInfo(
                                text=text,
                                font_size=round(font_size, self.size_rounding),
                                font_name=font_name,
                                is_bold=bool(is_bold),
                                is_italic=bool(is_italic),
                                bbox=BoundingBox(
                                    x0=float(bbox[0]),
                                    y0=float(bbox[1]),
                                    x1=float(bbox[2]),
                                    y1=float(bbox[3]),
                                ),
                                page_num=page_num,
                            )
                            all_spans.append(span_info)
                            
            except Exception as e:
                logger.warning(
                    "page_extraction_failed",
                    page=page.number,
                    error=str(e),
                )
                continue
        
        doc.close()
        
        logger.info(
            "spans_extracted",
            path=str(pdf_path),
            span_count=len(all_spans),
        )
        
        return all_spans

    def analyze(self, pdf_path: Path | str) -> tuple[FontAnalysis, list[SpanInfo]]:
        """
        Perform font histogram analysis on a PDF.
        
        This is the PRIMARY method for structure detection:
        1. Extract all spans
        2. Build frequency histogram of font sizes
        3. Identify body text (mode)
        4. Calculate header threshold
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Tuple of (FontAnalysis, list of SpanInfo).
            
        Raises:
            FontAnalysisError: If analysis fails.
        """
        # Step 1: Extract all spans
        spans = self.extract_all_spans(pdf_path)
        
        if not spans:
            raise FontAnalysisError("No text spans found in PDF")
        
        # Step 2: Build frequency histogram of font sizes
        sizes = [s.font_size for s in spans]
        size_counts = Counter(sizes)
        
        # Step 3: Identify Body Text (most frequent font size = mode)
        try:
            body_size = mode(sizes)
        except Exception:
            # If no clear mode, use most common
            body_size = size_counts.most_common(1)[0][0]
        
        # Step 4: Calculate header threshold
        unique_sizes = len(set(sizes))
        
        if unique_sizes > 1 and len(sizes) > 2:
            try:
                threshold = stdev(sizes)
            except Exception:
                threshold = 2.0
        else:
            threshold = 2.0  # Default if no variance
        
        header_threshold = body_size + threshold
        
        analysis = FontAnalysis(
            body_size=body_size,
            header_threshold=header_threshold,
            size_histogram={float(k): v for k, v in size_counts.items()},
            span_count=len(spans),
            unique_sizes=unique_sizes,
        )
        
        logger.info(
            "font_analysis_complete",
            body_size=body_size,
            header_threshold=header_threshold,
            unique_sizes=unique_sizes,
            span_count=len(spans),
        )
        
        return analysis, spans

    def analyze_spans(self, spans: list[SpanInfo]) -> FontAnalysis:
        """
        Perform font histogram analysis on pre-extracted spans.
        
        This is useful when spans have already been extracted and segmented
        (e.g., for multi-document PDFs).
        
        Args:
            spans: List of SpanInfo objects.
            
        Returns:
            FontAnalysis result.
        """
        if not spans:
            return FontAnalysis(
                body_size=12.0,
                header_threshold=14.0,
                size_histogram={12.0: 1},
                span_count=0,
                unique_sizes=1,
            )
        
        # Build frequency histogram of font sizes
        sizes = [s.font_size for s in spans]
        size_counts = Counter(sizes)
        
        # Identify Body Text (most frequent font size = mode)
        try:
            body_size = mode(sizes)
        except Exception:
            body_size = size_counts.most_common(1)[0][0]
        
        # Calculate header threshold
        unique_sizes = len(set(sizes))
        
        if unique_sizes > 1 and len(sizes) > 2:
            try:
                threshold = stdev(sizes)
            except Exception:
                threshold = 2.0
        else:
            threshold = 2.0
        
        header_threshold = body_size + threshold
        
        return FontAnalysis(
            body_size=body_size,
            header_threshold=header_threshold,
            size_histogram={float(k): v for k, v in size_counts.items()},
            span_count=len(spans),
            unique_sizes=unique_sizes,
        )

    def has_font_variance(self, analysis: FontAnalysis) -> bool:
        """
        Check if the document has enough font variance for hierarchical extraction.
        
        If there's no variance, we should fall back to Tier 2 (Semantic Splitter).
        
        Args:
            analysis: The FontAnalysis result.
            
        Returns:
            True if there's enough variance to detect headers.
        """
        return analysis.unique_sizes >= 2

    def has_detectable_headers(
        self, 
        analysis: FontAnalysis, 
        spans: list[SpanInfo],
    ) -> bool:
        """
        Check if headers can be detected from the font analysis.
        
        Args:
            analysis: The FontAnalysis result.
            spans: List of spans from the document.
            
        Returns:
            True if at least one span qualifies as a header.
        """
        for span in spans:
            if span.font_size > analysis.header_threshold:
                return True
            # Also check for bold text at body size (often headers)
            if span.is_bold and span.font_size >= analysis.body_size:
                return True
        return False


def analyze_font_histogram(pdf_path: Path | str) -> tuple[FontAnalysis, list[SpanInfo]]:
    """
    Convenience function for font histogram analysis.
    
    This is the PRIMARY ingestion method - use this instead of vision models.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Tuple of (FontAnalysis, list of SpanInfo).
        
    Example:
        analysis, spans = analyze_font_histogram("document.pdf")
        if analysis.unique_sizes >= 2:
            # Can detect headers
            headers = [s for s in spans if s.font_size > analysis.header_threshold]
    """
    analyzer = FontHistogramAnalyzer()
    return analyzer.analyze(pdf_path)
