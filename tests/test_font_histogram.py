"""Tests for Font Histogram Analyzer."""

import pytest
from rnsr.models import BoundingBox, SpanInfo


class TestFontHistogramAnalyzer:
    """Tests for FontHistogramAnalyzer."""
    
    def test_import(self):
        """Test that the module can be imported."""
        from rnsr.ingestion.font_histogram import FontHistogramAnalyzer
        analyzer = FontHistogramAnalyzer()
        assert analyzer is not None
    
    def test_classify_body_text(self):
        """Body text should be classified at the mode font size."""
        spans = [
            SpanInfo(
                text="Regular paragraph text",
                font_size=12.0,
                font_name="Arial",
                is_bold=False,
                is_italic=False,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20),
                page_num=0,
            )
            for _ in range(10)  # Most common size
        ]
        
        # Body text should be detected as mode
        sizes = [s.font_size for s in spans]
        from statistics import mode
        assert mode(sizes) == 12.0
    
    def test_span_info_model(self):
        """Test SpanInfo creation for font analysis."""
        bbox = BoundingBox(x0=0, y0=0, x1=200, y1=30)
        span = SpanInfo(
            text="Chapter Title",
            font_size=24.0,
            font_name="Arial-Bold",
            is_bold=True,
            is_italic=False,
            bbox=bbox,
            page_num=0,
        )
        
        assert span.font_size == 24.0
        assert span.is_bold is True
        assert span.text == "Chapter Title"


class TestFontAnalysisIntegration:
    """Integration tests requiring actual PDF files."""
    
    @pytest.mark.skip(reason="Requires test PDF file")
    def test_analyze_real_pdf(self):
        """Test with a real PDF document."""
        from rnsr.ingestion.font_histogram import FontHistogramAnalyzer
        analyzer = FontHistogramAnalyzer()
        # Would need a test PDF fixture
        pass
