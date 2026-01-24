"""
Header Classifier - Detect and Classify Headers by Level

This module classifies text spans into:
- Headers (H1, H2, H3) based on font size magnitude
- Body text (most frequent font size)
- Captions/footnotes (smaller than body)

Header Level Mapping:
- >= 24pt: H1 (Document Title)
- 18-23pt: H2 (Chapter/Section)
- 14-17pt: H3 (Subsection)
- Body size: Regular paragraph text
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog
from sklearn.cluster import KMeans

from rnsr.models import ClassifiedSpan, FontAnalysis, SpanInfo

logger = structlog.get_logger(__name__)


class HeaderClassifier:
    """
    Classifies text spans into headers and body text.
    
    Uses font size analysis and optional k-means clustering
    to determine header levels.
    """

    # Default thresholds for header levels (in points)
    H1_MIN_SIZE = 24.0
    H2_MIN_SIZE = 18.0
    H3_MIN_SIZE = 14.0

    def __init__(
        self,
        use_clustering: bool = True,
        n_header_levels: int = 3,
    ):
        """
        Initialize the Header Classifier.
        
        Args:
            use_clustering: Whether to use k-means clustering for header levels.
            n_header_levels: Number of header levels to detect (default: 3).
        """
        self.use_clustering = use_clustering
        self.n_header_levels = n_header_levels

    def classify_spans(
        self,
        spans: list[SpanInfo],
        analysis: FontAnalysis,
    ) -> list[ClassifiedSpan]:
        """
        Classify all spans into headers or body text.
        
        Args:
            spans: List of SpanInfo from font analysis.
            analysis: FontAnalysis with body size and threshold.
            
        Returns:
            List of ClassifiedSpan with role and header_level assigned.
        """
        if not spans:
            return []
        
        # First pass: identify potential headers
        potential_headers: list[SpanInfo] = []
        for span in spans:
            if self._is_header_candidate(span, analysis):
                potential_headers.append(span)
        
        # Determine header levels
        if potential_headers and self.use_clustering:
            level_mapping = self._cluster_header_levels(potential_headers)
        else:
            level_mapping = {}
        
        # Classify all spans
        classified: list[ClassifiedSpan] = []
        for span in spans:
            role, level = self._classify_single_span(span, analysis, level_mapping)
            
            classified_span = ClassifiedSpan(
                text=span.text,
                font_size=span.font_size,
                font_name=span.font_name,
                is_bold=span.is_bold,
                is_italic=span.is_italic,
                bbox=span.bbox,
                page_num=span.page_num,
                role=role,
                header_level=level,
            )
            classified.append(classified_span)
        
        # Log classification stats
        header_count = sum(1 for s in classified if s.role == "header")
        logger.info(
            "spans_classified",
            total=len(classified),
            headers=header_count,
            body=len(classified) - header_count,
        )
        
        return classified

    def _is_header_candidate(
        self, 
        span: SpanInfo, 
        analysis: FontAnalysis,
    ) -> bool:
        """
        Determine if a span is a header candidate.
        
        A span is a header candidate if:
        1. Font size > header_threshold, OR
        2. Bold text at or above body size
        """
        # Size-based detection
        if span.font_size > analysis.header_threshold:
            return True
        
        # Bold text at body size or larger
        if span.is_bold and span.font_size >= analysis.body_size:
            return True
        
        return False

    def _classify_single_span(
        self,
        span: SpanInfo,
        analysis: FontAnalysis,
        level_mapping: dict[float, int],
    ) -> tuple[Literal["header", "body", "caption", "footnote"], int]:
        """
        Classify a single span.
        
        Returns:
            Tuple of (role, header_level).
        """
        # Check if it's a header
        if self._is_header_candidate(span, analysis):
            # Use clustering-based level if available
            if span.font_size in level_mapping:
                level = level_mapping[span.font_size]
            else:
                # Fall back to absolute thresholds
                level = self._get_level_by_size(span.font_size)
            
            return ("header", level)
        
        # Check for captions/footnotes (smaller than body)
        caption_threshold = analysis.body_size - (analysis.body_size * 0.2)
        if span.font_size < caption_threshold:
            return ("caption", 0)
        
        # Default to body text
        return ("body", 0)

    def _get_level_by_size(self, font_size: float) -> int:
        """
        Get header level based on absolute font size thresholds.
        
        Args:
            font_size: The font size in points.
            
        Returns:
            Header level (1, 2, or 3).
        """
        if font_size >= self.H1_MIN_SIZE:
            return 1
        elif font_size >= self.H2_MIN_SIZE:
            return 2
        else:
            return 3

    def _cluster_header_levels(
        self, 
        headers: list[SpanInfo],
    ) -> dict[float, int]:
        """
        Use k-means clustering to determine header levels from actual data.
        
        This adapts to documents with non-standard font sizes.
        
        Args:
            headers: List of spans identified as headers.
            
        Returns:
            Dict mapping font_size to header_level (1, 2, or 3).
        """
        if len(headers) < self.n_header_levels:
            # Not enough headers to cluster
            return {}
        
        # Get unique font sizes
        unique_sizes = list(set(h.font_size for h in headers))
        
        if len(unique_sizes) < self.n_header_levels:
            # Fewer unique sizes than levels - assign directly
            sorted_sizes = sorted(unique_sizes, reverse=True)
            return {size: i + 1 for i, size in enumerate(sorted_sizes)}
        
        # Perform k-means clustering
        try:
            X = np.array(unique_sizes).reshape(-1, 1)
            n_clusters = min(self.n_header_levels, len(unique_sizes))
            
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42,
                n_init=10,
            ).fit(X)
            
            # Map clusters to levels by size (largest = H1)
            cluster_centers = [
                (i, kmeans.cluster_centers_[i][0]) 
                for i in range(n_clusters)
            ]
            cluster_centers.sort(key=lambda x: -x[1])  # Descending by size
            
            cluster_to_level = {
                cluster: level + 1 
                for level, (cluster, _) in enumerate(cluster_centers)
            }
            
            # Map each unique size to its level
            size_to_level = {}
            for size in unique_sizes:
                cluster = kmeans.predict([[size]])[0]
                size_to_level[size] = cluster_to_level[cluster]
            
            logger.debug(
                "header_levels_clustered",
                mapping=size_to_level,
            )
            
            return size_to_level
            
        except Exception as e:
            logger.warning("clustering_failed", error=str(e))
            return {}


def classify_headers(
    spans: list[SpanInfo],
    analysis: FontAnalysis,
    use_clustering: bool = True,
) -> list[ClassifiedSpan]:
    """
    Convenience function to classify spans into headers and body text.
    
    Args:
        spans: List of SpanInfo from font analysis.
        analysis: FontAnalysis with body size and threshold.
        use_clustering: Whether to use k-means for header levels.
        
    Returns:
        List of ClassifiedSpan with roles assigned.
        
    Example:
        analysis, spans = analyze_font_histogram("doc.pdf")
        classified = classify_headers(spans, analysis)
        headers = [s for s in classified if s.role == "header"]
    """
    classifier = HeaderClassifier(use_clustering=use_clustering)
    return classifier.classify_spans(spans, analysis)
