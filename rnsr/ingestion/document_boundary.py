"""
Document Boundary Detection for Multi-Document PDFs

This module detects boundaries between separate documents within a single PDF file.
It analyzes visual and textual signals to identify where one document ends and
another begins, enabling proper segmentation before tree building.

Boundary Detection Signals:
1. Title Page Patterns: Large font text at top of page (potential new document)
2. Page Number Resets: Page numbers restarting from 1 (strong signal)
3. Style Discontinuity: Dramatic font/style changes between pages
4. Document Type Indicators: Form headers, letterheads, report titles
5. LLM Validation: Optional LLM review of potential boundaries
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import structlog

from rnsr.models import BoundingBox, SpanInfo

logger = structlog.get_logger(__name__)

# Regex patterns for page number detection
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^page\s*(\d+)$", re.IGNORECASE),           # "Page 1", "page 1"
    re.compile(r"^(\d+)\s*of\s*\d+$", re.IGNORECASE),       # "1 of 10"
    re.compile(r"^-\s*(\d+)\s*-$"),                          # "- 1 -"
    re.compile(r"^(\d+)$"),                                  # Just "1"
    re.compile(r"^page\s*(\d+)\s*of\s*\d+$", re.IGNORECASE), # "Page 1 of 10"
]


@dataclass
class DocumentBoundary:
    """Represents a detected document boundary."""
    
    page_num: int  # Page where new document starts
    confidence: float  # 0.0 - 1.0
    signals: list[str]  # What triggered detection
    title_candidate: str = ""  # Potential document title


@dataclass
class DocumentSegment:
    """A segment of spans belonging to one logical document."""
    
    doc_index: int
    start_page: int
    end_page: int
    spans: list[SpanInfo] = field(default_factory=list)
    title: str = ""
    boundary: DocumentBoundary | None = None


class DocumentBoundaryDetector:
    """
    Detects boundaries between documents in a multi-document PDF.
    
    Works by analyzing:
    - Font size patterns (title pages have larger fonts)
    - Page number resets (page numbers restarting from 1)
    - Style discontinuities between pages
    - Document type indicators
    - Optional LLM validation of boundaries
    """
    
    def __init__(
        self,
        title_size_ratio: float = 2.0,  # Title must be 2x body size (very conservative)
        min_title_size: float = 18.0,   # Minimum font size for title (larger)
        page_top_fraction: float = 0.15,  # "Top of page" = top 15% only
        min_confidence: float = 0.75,   # High confidence required
        min_signals: int = 2,           # Require at least 2 signals (page reset is strong)
        use_llm_validation: bool = True,  # Use LLM to validate boundaries
    ):
        self.title_size_ratio = title_size_ratio
        self.min_title_size = min_title_size
        self.page_top_fraction = page_top_fraction
        self.min_confidence = min_confidence
        self.min_signals = min_signals
        self.use_llm_validation = use_llm_validation
    
    def _extract_page_number(self, spans: list[SpanInfo], page_height: float) -> int | None:
        """
        Extract page number from footer/header of a page.
        
        Looks for page number patterns in the bottom 15% or top 10% of the page.
        """
        if not spans:
            return None
        
        # Look in footer (bottom 15%) and header (top 10%)
        footer_threshold = page_height * 0.85
        header_threshold = page_height * 0.10
        
        # Get spans from footer and header areas
        edge_spans = [
            s for s in spans 
            if s.bbox.y0 > footer_threshold or s.bbox.y1 < header_threshold
        ]
        
        for span in edge_spans:
            text = span.text.strip()
            
            # Try each pattern
            for pattern in PAGE_NUMBER_PATTERNS:
                match = pattern.match(text)
                if match:
                    try:
                        return int(match.group(1))
                    except (ValueError, IndexError):
                        continue
        
        return None
    
    def _detect_page_number_resets(
        self,
        pages: dict[int, list[SpanInfo]],
        page_heights: dict[int, float] | None,
    ) -> dict[int, bool]:
        """
        Detect pages where page numbers reset to 1 or restart a sequence.
        
        Returns dict mapping page_num -> True if page number reset detected.
        """
        page_nums = sorted(pages.keys())
        resets: dict[int, bool] = {}
        
        if len(page_nums) < 2:
            return resets
        
        prev_page_num: int | None = None
        
        for pdf_page in page_nums:
            page_spans = pages[pdf_page]
            
            # Estimate page height
            if page_heights and pdf_page in page_heights:
                page_height = page_heights[pdf_page]
            else:
                max_y = max((s.bbox.y1 for s in page_spans), default=800)
                page_height = max_y * 1.1
            
            # Extract page number from this page
            doc_page_num = self._extract_page_number(page_spans, page_height)
            
            if doc_page_num is not None:
                # Check for reset: page number is 1, or significantly less than previous
                if doc_page_num == 1 and prev_page_num is not None and prev_page_num > 1:
                    resets[pdf_page] = True
                    logger.debug(
                        "page_number_reset_detected",
                        pdf_page=pdf_page,
                        doc_page=doc_page_num,
                        prev_doc_page=prev_page_num,
                    )
                
                prev_page_num = doc_page_num
        
        return resets
    
    def detect_boundaries(
        self,
        spans: list[SpanInfo],
        page_heights: dict[int, float] | None = None,
    ) -> list[DocumentBoundary]:
        """
        Detect document boundaries in a list of spans.
        
        Args:
            spans: List of SpanInfo from the PDF
            page_heights: Optional dict mapping page_num -> page_height
            
        Returns:
            List of DocumentBoundary objects (sorted by page number)
        """
        if not spans:
            return []
        
        # Group spans by page
        pages = self._group_by_page(spans)
        page_nums = sorted(pages.keys())
        
        if len(page_nums) < 2:
            return []  # Single page, no boundaries to detect
        
        # Calculate global statistics
        all_sizes = [s.font_size for s in spans]
        body_size = self._estimate_body_size(all_sizes)
        
        # STEP 1: Detect page number resets (very strong signal)
        page_resets = self._detect_page_number_resets(pages, page_heights)
        
        logger.info(
            "page_number_resets_detected",
            count=len(page_resets),
            pages=list(page_resets.keys()),
        )
        
        boundaries: list[DocumentBoundary] = []
        
        # Analyze each page (skip first - it's always the start)
        for i, page_num in enumerate(page_nums[1:], 1):
            prev_page_num = page_nums[i - 1]
            
            # Check if this page has a page number reset
            has_page_reset = page_resets.get(page_num, False)
            
            boundary = self._analyze_page_boundary(
                current_page=pages[page_num],
                prev_page=pages[prev_page_num],
                page_num=page_num,
                body_size=body_size,
                page_heights=page_heights,
                has_page_reset=has_page_reset,
            )
            
            if boundary and boundary.confidence >= self.min_confidence:
                boundaries.append(boundary)
                logger.debug(
                    "boundary_detected",
                    page=page_num,
                    confidence=boundary.confidence,
                    signals=boundary.signals,
                    title=boundary.title_candidate[:50] if boundary.title_candidate else "",
                )
        
        # STEP 2: Optional LLM validation
        if self.use_llm_validation and boundaries:
            boundaries = self._validate_boundaries_with_llm(boundaries, pages)
        
        logger.info(
            "boundary_detection_complete",
            total_pages=len(page_nums),
            boundaries_found=len(boundaries),
        )
        
        return boundaries
    
    def segment_spans(
        self,
        spans: list[SpanInfo],
        boundaries: list[DocumentBoundary],
    ) -> list[DocumentSegment]:
        """
        Split spans into document segments based on detected boundaries.
        
        Args:
            spans: All spans from the PDF
            boundaries: Detected document boundaries
            
        Returns:
            List of DocumentSegment, each containing spans for one document
        """
        if not spans:
            return []
        
        # Sort boundaries by page number
        sorted_boundaries = sorted(boundaries, key=lambda b: b.page_num)
        
        # Get boundary page numbers
        boundary_pages = {b.page_num for b in sorted_boundaries}
        
        # Group spans by page
        pages = self._group_by_page(spans)
        page_nums = sorted(pages.keys())
        
        segments: list[DocumentSegment] = []
        current_segment = DocumentSegment(
            doc_index=0,
            start_page=page_nums[0] if page_nums else 0,
            end_page=page_nums[0] if page_nums else 0,
        )
        
        for page_num in page_nums:
            page_spans = pages[page_num]
            
            # Check if this page starts a new document
            if page_num in boundary_pages:
                # Save current segment
                if current_segment.spans:
                    segments.append(current_segment)
                
                # Find the boundary for this page
                boundary = next(b for b in sorted_boundaries if b.page_num == page_num)
                
                # Start new segment
                current_segment = DocumentSegment(
                    doc_index=len(segments),
                    start_page=page_num,
                    end_page=page_num,
                    title=boundary.title_candidate,
                    boundary=boundary,
                )
            
            # Add page spans to current segment
            current_segment.spans.extend(page_spans)
            current_segment.end_page = page_num
        
        # Don't forget the last segment
        if current_segment.spans:
            segments.append(current_segment)
        
        # If no title was detected for first segment, try to extract one
        if segments and not segments[0].title:
            segments[0].title = self._extract_title_from_spans(segments[0].spans)
        
        logger.info(
            "spans_segmented",
            total_segments=len(segments),
            spans_per_segment=[len(s.spans) for s in segments],
        )
        
        return segments
    
    def _group_by_page(self, spans: list[SpanInfo]) -> dict[int, list[SpanInfo]]:
        """Group spans by page number."""
        pages: dict[int, list[SpanInfo]] = {}
        for span in spans:
            if span.page_num not in pages:
                pages[span.page_num] = []
            pages[span.page_num].append(span)
        return pages
    
    def _estimate_body_size(self, sizes: list[float]) -> float:
        """Estimate body text size (mode of font sizes)."""
        if not sizes:
            return 12.0
        
        # Round to 1 decimal and find mode
        rounded = [round(s, 1) for s in sizes]
        return max(set(rounded), key=rounded.count)
    
    def _analyze_page_boundary(
        self,
        current_page: list[SpanInfo],
        prev_page: list[SpanInfo],
        page_num: int,
        body_size: float,
        page_heights: dict[int, float] | None,
        has_page_reset: bool = False,
    ) -> DocumentBoundary | None:
        """
        Analyze whether a page represents a document boundary.
        
        Returns DocumentBoundary if signals indicate a new document, None otherwise.
        """
        signals: list[str] = []
        confidence_factors: list[float] = []
        title_candidate = ""
        
        # STRONGEST SIGNAL: Page number reset to 1
        # This is a very reliable indicator of a new document
        if has_page_reset:
            signals.append("page_number_reset")
            confidence_factors.append(0.7)  # Very high weight
        
        # Get page dimensions
        if page_heights and page_num in page_heights:
            page_height = page_heights[page_num]
        else:
            # Estimate from spans
            max_y = max((s.bbox.y1 for s in current_page), default=800)
            page_height = max_y * 1.1  # Add margin
        
        top_threshold = page_height * self.page_top_fraction
        
        # Signal 1: Large text at top of page (title pattern)
        top_spans = [s for s in current_page if s.bbox.y0 < top_threshold]
        if top_spans:
            max_top_size = max(s.font_size for s in top_spans)
            
            # Check for title-like text
            if max_top_size >= body_size * self.title_size_ratio:
                signals.append(f"large_title_font_{max_top_size:.1f}pt")
                confidence_factors.append(0.4)  # Reduced from 0.7 - needs other signals
                
                # Get the title text
                title_spans = [s for s in top_spans if s.font_size == max_top_size]
                if title_spans:
                    title_candidate = " ".join(s.text.strip() for s in title_spans[:3])
            
            # Check for very large text (stronger signal)
            if max_top_size >= self.min_title_size * 1.5:
                signals.append("very_large_header")
                confidence_factors.append(0.35)
        
        # Signal 2: Complete style discontinuity with previous page
        # Only triggers if there's a DRAMATIC change, not just minor differences
        if prev_page and current_page:
            prev_sizes = {round(s.font_size, 1) for s in prev_page}
            curr_sizes = {round(s.font_size, 1) for s in current_page}
            
            # Check for completely different font sizes (NO overlap)
            overlap = prev_sizes & curr_sizes
            if len(overlap) == 0:  # Zero overlap = very strong signal
                signals.append("complete_style_change")
                confidence_factors.append(0.5)
            
            # Check for completely different fonts (strong indicator)
            prev_fonts = {s.font_name for s in prev_page}
            curr_fonts = {s.font_name for s in current_page}
            if prev_fonts and curr_fonts and not prev_fonts & curr_fonts:
                signals.append("font_family_change")
                confidence_factors.append(0.4)
        
        # Signal 4: Previous page ended with typical document end patterns
        if prev_page:
            last_spans = sorted(prev_page, key=lambda s: s.bbox.y1, reverse=True)[:3]
            last_text = " ".join(s.text.strip().lower() for s in last_spans)
            
            end_patterns = [
                "signature", "signed", "date:", "end of document",
                "appendix", "attachment", "annex", "exhibit",
                "page", "of", "---", "___",
            ]
            
            if any(p in last_text for p in end_patterns):
                signals.append("prev_page_end_pattern")
                confidence_factors.append(0.3)
        
        # Signal 5: Current page has STRONG document type indicators
        # These are patterns that strongly suggest a new document, not just a section
        if top_spans:
            top_text = " ".join(s.text.strip().lower() for s in top_spans)
            
            # Strong indicators - things that typically start new documents
            strong_patterns = [
                ("certificate of", 0.5),   # Certificate of capacity, etc.
                ("report to", 0.45),       # Report to Parliament, etc.
                ("form 10:", 0.5),         # Form numbers
                ("whs form", 0.5),         # WHS forms
                ("incident", 0.35),        # Incident reports
                ("court of", 0.45),        # Supreme Court of...
                ("comprehensive", 0.35),   # Comprehensive checkup, etc.
                ("clinical notes", 0.45),  # Medical notes
                ("court attendance", 0.5), # Court attendance notice
                ("machine safety", 0.4),   # Safety documents
            ]
            
            for pattern, weight in strong_patterns:
                if pattern in top_text:
                    signals.append(f"strong_doc_indicator_{pattern.replace(' ', '_')}")
                    confidence_factors.append(weight)
                    break  # Only count the strongest match
        
        # Signal 6: Sparse content page (potential divider or cover page)
        total_chars = sum(len(s.text) for s in current_page)
        if total_chars < 200 and top_spans:  # Little text but has header
            max_size = max(s.font_size for s in current_page)
            if max_size >= body_size * self.title_size_ratio:
                signals.append("sparse_title_page")
                confidence_factors.append(0.4)
        
        # Calculate overall confidence
        if not confidence_factors:
            return None
        
        # Require minimum number of signals for a document boundary
        # This prevents section headers from being mistaken for doc boundaries
        if len(signals) < self.min_signals:
            return None
        
        # NEGATIVE SIGNAL: Check for style continuity with previous page
        # If styles are very similar, this is likely a continuation, not a new doc
        if prev_page and current_page:
            prev_sizes = [round(s.font_size, 0) for s in prev_page]
            curr_sizes = [round(s.font_size, 0) for s in current_page]
            
            if prev_sizes and curr_sizes:
                # Check if the most common sizes match
                prev_common = max(set(prev_sizes), key=prev_sizes.count)
                curr_common = max(set(curr_sizes), key=curr_sizes.count)
                
                if prev_common == curr_common:
                    # Same body text size - likely same document
                    # Reduce confidence significantly
                    confidence_factors = [f * 0.6 for f in confidence_factors]
                    signals.append("style_continuity_penalty")
        
        # Combine confidence factors (diminishing returns)
        confidence = 0.0
        for i, factor in enumerate(sorted(confidence_factors, reverse=True)):
            confidence += factor * (0.7 ** i)
        
        # Cap at 1.0
        confidence = min(confidence, 1.0)
        
        if confidence >= self.min_confidence:
            return DocumentBoundary(
                page_num=page_num,
                confidence=confidence,
                signals=signals,
                title_candidate=title_candidate,
            )
        
        return None
    
    def _validate_boundaries_with_llm(
        self,
        boundaries: list[DocumentBoundary],
        pages: dict[int, list[SpanInfo]],
    ) -> list[DocumentBoundary]:
        """
        Use LLM to validate detected boundaries.
        
        Sends context from around each boundary to the LLM to confirm
        if it's a true document boundary or just a section break.
        """
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
        except Exception as e:
            logger.warning("llm_validation_unavailable", error=str(e))
            return boundaries
        
        validated: list[DocumentBoundary] = []
        
        for boundary in boundaries:
            # Get context: last 500 chars of previous page, first 500 chars of current page
            prev_page_num = boundary.page_num - 1
            
            prev_text = ""
            if prev_page_num in pages:
                prev_spans = sorted(pages[prev_page_num], key=lambda s: (s.bbox.y0, s.bbox.x0))
                prev_text = " ".join(s.text.strip() for s in prev_spans)[-500:]
            
            curr_text = ""
            if boundary.page_num in pages:
                curr_spans = sorted(pages[boundary.page_num], key=lambda s: (s.bbox.y0, s.bbox.x0))
                curr_text = " ".join(s.text.strip() for s in curr_spans)[:500]
            
            # Build prompt for LLM
            prompt = f"""You are analyzing a multi-document PDF to detect document boundaries.

The system detected a potential document boundary at this location. 
Review the text from BEFORE and AFTER the boundary and determine if this is:
- A TRUE document boundary (a completely new, separate document starts here)
- A FALSE boundary (this is just a section/chapter break within the same document)

SIGNALS DETECTED: {', '.join(boundary.signals)}
POTENTIAL NEW DOCUMENT TITLE: {boundary.title_candidate or 'Unknown'}
CONFIDENCE: {boundary.confidence:.2f}

--- END OF PREVIOUS DOCUMENT ---
{prev_text}

--- START OF POTENTIAL NEW DOCUMENT ---
{curr_text}

Is this a TRUE document boundary (a completely separate document starts here)?
Answer ONLY with: TRUE or FALSE

Your answer:"""

            try:
                response = llm.complete(prompt)
                response_text = str(response).strip().upper()
                
                is_valid = "TRUE" in response_text
                
                logger.debug(
                    "llm_boundary_validation",
                    page=boundary.page_num,
                    title=boundary.title_candidate[:30] if boundary.title_candidate else "",
                    llm_response=response_text[:50],
                    is_valid=is_valid,
                )
                
                if is_valid:
                    validated.append(boundary)
                else:
                    logger.info(
                        "boundary_rejected_by_llm",
                        page=boundary.page_num,
                        title=boundary.title_candidate[:30] if boundary.title_candidate else "",
                    )
                    
            except Exception as e:
                logger.warning("llm_validation_failed", page=boundary.page_num, error=str(e))
                # On LLM error, keep boundaries with high confidence
                if boundary.confidence >= 0.8:
                    validated.append(boundary)
        
        logger.info(
            "llm_validation_complete",
            original_count=len(boundaries),
            validated_count=len(validated),
            rejected_count=len(boundaries) - len(validated),
        )
        
        return validated

    def _extract_title_from_spans(self, spans: list[SpanInfo]) -> str:
        """Extract a title from the first page's spans."""
        if not spans:
            return "Document"
        
        # Get spans from first page
        first_page = min(s.page_num for s in spans)
        first_page_spans = [s for s in spans if s.page_num == first_page]
        
        if not first_page_spans:
            return "Document"
        
        # Find largest text near top
        page_height = max(s.bbox.y1 for s in first_page_spans)
        top_spans = [s for s in first_page_spans if s.bbox.y0 < page_height * 0.3]
        
        if top_spans:
            max_size = max(s.font_size for s in top_spans)
            title_spans = [s for s in top_spans if s.font_size == max_size]
            return " ".join(s.text.strip() for s in title_spans[:3])
        
        return "Document"


def detect_document_boundaries(
    spans: list[SpanInfo],
    page_heights: dict[int, float] | None = None,
    min_confidence: float = 0.5,
) -> list[DocumentBoundary]:
    """
    Convenience function to detect document boundaries.
    
    Args:
        spans: List of SpanInfo from the PDF
        page_heights: Optional dict mapping page_num -> page_height
        min_confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        List of DocumentBoundary objects
    """
    detector = DocumentBoundaryDetector(min_confidence=min_confidence)
    return detector.detect_boundaries(spans, page_heights)


def segment_by_documents(
    spans: list[SpanInfo],
    page_heights: dict[int, float] | None = None,
    min_confidence: float = 0.5,
) -> list[DocumentSegment]:
    """
    Convenience function to segment spans into separate documents.
    
    Args:
        spans: List of SpanInfo from the PDF
        page_heights: Optional dict mapping page_num -> page_height
        min_confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        List of DocumentSegment, each containing spans for one document
    """
    detector = DocumentBoundaryDetector(min_confidence=min_confidence)
    boundaries = detector.detect_boundaries(spans, page_heights)
    return detector.segment_spans(spans, boundaries)
