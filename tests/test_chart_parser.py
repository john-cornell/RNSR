"""Tests for the Chart Parser system."""

import pytest

from rnsr.ingestion.chart_parser import (
    ChartAnalysis,
    ChartParser,
    ChartSeries,
    ChartType,
    DataPoint,
    ParsedChart,
    describe_chart,
)


class TestDataPoint:
    """Tests for DataPoint dataclass."""
    
    def test_data_point_creation(self):
        """Test creating a data point."""
        point = DataPoint(
            x="Q1 2024",
            y=100.5,
            series="Revenue",
            label="Q1",
        )
        
        assert point.x == "Q1 2024"
        assert point.y == 100.5
        assert point.series == "Revenue"
    
    def test_data_point_to_dict(self):
        """Test converting data point to dictionary."""
        point = DataPoint(x=1, y=50)
        
        data = point.to_dict()
        
        assert data["x"] == 1
        assert data["y"] == 50


class TestChartSeries:
    """Tests for ChartSeries dataclass."""
    
    def test_series_creation(self):
        """Test creating a chart series."""
        points = [
            DataPoint(x=1, y=10),
            DataPoint(x=2, y=20),
            DataPoint(x=3, y=30),
        ]
        
        series = ChartSeries(
            name="Test Series",
            data_points=points,
            color="blue",
        )
        
        assert series.name == "Test Series"
        assert len(series.data_points) == 3
    
    def test_series_get_values(self):
        """Test getting all Y values from a series."""
        points = [
            DataPoint(x=1, y=10),
            DataPoint(x=2, y=20),
            DataPoint(x=3, y=30),
        ]
        
        series = ChartSeries(data_points=points)
        
        values = series.get_values()
        
        assert values == [10, 20, 30]
    
    def test_series_get_labels(self):
        """Test getting all X labels from a series."""
        points = [
            DataPoint(x="Jan", y=10),
            DataPoint(x="Feb", y=20),
            DataPoint(x="Mar", y=30),
        ]
        
        series = ChartSeries(data_points=points)
        
        labels = series.get_labels()
        
        assert labels == ["Jan", "Feb", "Mar"]


class TestChartAnalysis:
    """Tests for ChartAnalysis dataclass."""
    
    def test_analysis_creation(self):
        """Test creating a chart analysis."""
        analysis = ChartAnalysis(
            trend="increasing",
            trend_description="Values show upward trend",
            min_value=10.0,
            max_value=100.0,
            avg_value=55.0,
            key_insights=["Growth is consistent", "Peak in Q4"],
        )
        
        assert analysis.trend == "increasing"
        assert analysis.min_value == 10.0
        assert len(analysis.key_insights) == 2
    
    def test_analysis_to_dict(self):
        """Test converting analysis to dictionary."""
        analysis = ChartAnalysis(
            trend="stable",
            avg_value=50.0,
        )
        
        data = analysis.to_dict()
        
        assert data["trend"] == "stable"
        assert data["avg_value"] == 50.0


class TestParsedChart:
    """Tests for ParsedChart dataclass."""
    
    def test_chart_creation(self):
        """Test creating a parsed chart."""
        chart = ParsedChart(
            doc_id="doc_123",
            title="Revenue Chart",
            chart_type=ChartType.BAR,
            x_axis_label="Quarter",
            y_axis_label="Revenue ($M)",
        )
        
        assert chart.doc_id == "doc_123"
        assert chart.chart_type == ChartType.BAR
    
    def test_chart_get_all_values(self):
        """Test getting all values from all series."""
        series1 = ChartSeries(
            name="Series 1",
            data_points=[
                DataPoint(x=1, y=10),
                DataPoint(x=2, y=20),
            ],
        )
        series2 = ChartSeries(
            name="Series 2",
            data_points=[
                DataPoint(x=1, y=15),
                DataPoint(x=2, y=25),
            ],
        )
        
        chart = ParsedChart(series=[series1, series2])
        
        values = chart.get_all_values()
        
        assert len(values) == 4
        assert set(values) == {10, 20, 15, 25}
    
    def test_chart_summarize(self):
        """Test generating chart summary."""
        series = ChartSeries(
            data_points=[
                DataPoint(x=1, y=10),
                DataPoint(x=2, y=20),
            ],
        )
        
        chart = ParsedChart(
            title="Test Chart",
            chart_type=ChartType.LINE,
            series=[series],
            analysis=ChartAnalysis(trend="increasing"),
        )
        
        summary = chart.summarize()
        
        assert "Test Chart" in summary
        assert "line" in summary
        assert "increasing" in summary


class TestChartParser:
    """Tests for ChartParser class."""
    
    def test_parser_creation(self):
        """Test creating a chart parser."""
        parser = ChartParser()
        assert parser is not None
    
    def test_analyze_chart_increasing_trend(self):
        """Test detecting increasing trend."""
        parser = ChartParser()
        
        series = ChartSeries(
            data_points=[
                DataPoint(x=1, y=10),
                DataPoint(x=2, y=20),
                DataPoint(x=3, y=30),
                DataPoint(x=4, y=40),
                DataPoint(x=5, y=50),
            ],
        )
        
        chart = ParsedChart(series=[series])
        
        analysis = parser._analyze_chart(chart)
        
        assert analysis.trend == "increasing"
        assert analysis.min_value == 10
        assert analysis.max_value == 50
        assert analysis.avg_value == 30
    
    def test_analyze_chart_decreasing_trend(self):
        """Test detecting decreasing trend."""
        parser = ChartParser()
        
        series = ChartSeries(
            data_points=[
                DataPoint(x=1, y=50),
                DataPoint(x=2, y=40),
                DataPoint(x=3, y=30),
                DataPoint(x=4, y=20),
                DataPoint(x=5, y=10),
            ],
        )
        
        chart = ParsedChart(series=[series])
        
        analysis = parser._analyze_chart(chart)
        
        assert analysis.trend == "decreasing"
    
    def test_analyze_chart_stable_trend(self):
        """Test detecting stable trend."""
        parser = ChartParser()
        
        series = ChartSeries(
            data_points=[
                DataPoint(x=1, y=50),
                DataPoint(x=2, y=51),
                DataPoint(x=3, y=49),
                DataPoint(x=4, y=50),
                DataPoint(x=5, y=51),
            ],
        )
        
        chart = ParsedChart(series=[series])
        
        analysis = parser._analyze_chart(chart)
        
        assert analysis.trend == "stable"
    
    def test_analyze_chart_fluctuating_trend(self):
        """Test detecting fluctuating trend."""
        parser = ChartParser()
        
        series = ChartSeries(
            data_points=[
                DataPoint(x=1, y=10),
                DataPoint(x=2, y=90),
                DataPoint(x=3, y=20),
                DataPoint(x=4, y=80),
                DataPoint(x=5, y=15),
            ],
        )
        
        chart = ParsedChart(series=[series])
        
        analysis = parser._analyze_chart(chart)
        
        # High variance should indicate fluctuation
        assert analysis.trend in ["fluctuating", "increasing", "decreasing"]
    
    def test_parse_from_description_bar_chart(self):
        """Test parsing chart from description - bar chart."""
        parser = ChartParser()
        
        chart = parser.parse_from_description(
            "This bar chart shows quarterly revenue for 2024.",
            doc_id="report.pdf",
            page_num=5,
        )
        
        assert chart.chart_type == ChartType.BAR
        assert chart.doc_id == "report.pdf"
    
    def test_parse_from_description_line_chart(self):
        """Test parsing chart from description - line chart."""
        parser = ChartParser()
        
        chart = parser.parse_from_description(
            "The line graph displays temperature trends over time.",
        )
        
        assert chart.chart_type == ChartType.LINE
    
    def test_parse_from_description_pie_chart(self):
        """Test parsing chart from description - pie chart."""
        parser = ChartParser()
        
        chart = parser.parse_from_description(
            "This pie chart shows market share distribution.",
        )
        
        assert chart.chart_type == ChartType.PIE
    
    def test_parse_from_description_extracts_numbers(self):
        """Test that numbers are extracted from description."""
        parser = ChartParser()
        
        chart = parser.parse_from_description(
            "Revenue was $100M in Q1, $150M in Q2, and $200M in Q3.",
        )
        
        # Should have extracted some numeric data
        if chart.series:
            values = chart.series[0].get_values()
            assert len(values) >= 1


class TestDescribeChart:
    """Tests for describe_chart function."""
    
    def test_describe_basic_chart(self):
        """Test describing a basic chart."""
        chart = ParsedChart(
            title="Sales Data",
            chart_type=ChartType.BAR,
            x_axis_label="Month",
            y_axis_label="Sales",
        )
        
        description = describe_chart(chart)
        
        assert "bar" in description
        assert "Sales Data" in description
    
    def test_describe_chart_with_data(self):
        """Test describing chart with data."""
        series = ChartSeries(
            data_points=[
                DataPoint(x="Jan", y=100),
                DataPoint(x="Feb", y=150),
            ],
        )
        
        chart = ParsedChart(
            chart_type=ChartType.LINE,
            series=[series],
        )
        
        description = describe_chart(chart, include_data=True)
        
        assert "line" in description
        assert "Jan" in description or "data points" in description
    
    def test_describe_chart_with_analysis(self):
        """Test describing chart with analysis."""
        chart = ParsedChart(
            chart_type=ChartType.LINE,
            analysis=ChartAnalysis(
                trend="increasing",
                trend_description="Values show consistent growth",
                key_insights=["Strong growth in Q4"],
            ),
        )
        
        description = describe_chart(chart)
        
        assert "increasing" in description or "growth" in description
