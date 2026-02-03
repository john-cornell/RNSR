"""
RNSR Chart Parser

Extracts and interprets charts from documents.
Enables answering questions about visual data representations.

Features:
- Chart type detection (bar, line, pie, scatter)
- Data point extraction
- Trend analysis
- LLM-based chart interpretation

Note: Requires vision capabilities for actual chart parsing.
This module provides the framework and LLM interpretation layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ChartType(str, Enum):
    """Types of charts."""
    
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    STACKED_BAR = "stacked_bar"
    COMBINATION = "combination"
    UNKNOWN = "unknown"


@dataclass
class DataPoint:
    """A single data point in a chart."""
    
    x: str | float  # X-axis value or label
    y: float  # Y-axis value
    series: str = ""  # Series name (for multi-series charts)
    label: str = ""  # Display label
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "series": self.series,
            "label": self.label,
        }


@dataclass
class ChartSeries:
    """A data series in a chart."""
    
    name: str = ""
    data_points: list[DataPoint] = field(default_factory=list)
    color: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "color": self.color,
        }
    
    def get_values(self) -> list[float]:
        """Get all Y values."""
        return [dp.y for dp in self.data_points]
    
    def get_labels(self) -> list[str]:
        """Get all X labels."""
        return [str(dp.x) for dp in self.data_points]


@dataclass
class ChartAnalysis:
    """Analysis of chart trends and insights."""
    
    trend: str = ""  # "increasing", "decreasing", "stable", "fluctuating"
    trend_description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    avg_value: float | None = None
    key_insights: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trend": self.trend,
            "trend_description": self.trend_description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "avg_value": self.avg_value,
            "key_insights": self.key_insights,
        }


@dataclass
class ParsedChart:
    """A fully parsed chart."""
    
    id: str = field(default_factory=lambda: f"chart_{str(uuid4())[:8]}")
    
    # Source information
    doc_id: str = ""
    page_num: int | None = None
    node_id: str = ""
    
    # Chart metadata
    title: str = ""
    chart_type: ChartType = ChartType.UNKNOWN
    
    # Axis information
    x_axis_label: str = ""
    y_axis_label: str = ""
    x_axis_unit: str = ""
    y_axis_unit: str = ""
    
    # Data
    series: list[ChartSeries] = field(default_factory=list)
    
    # Analysis
    analysis: ChartAnalysis | None = None
    
    # LLM interpretation
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "node_id": self.node_id,
            "title": self.title,
            "chart_type": self.chart_type.value,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "x_axis_unit": self.x_axis_unit,
            "y_axis_unit": self.y_axis_unit,
            "series": [s.to_dict() for s in self.series],
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "description": self.description,
        }
    
    def get_all_values(self) -> list[float]:
        """Get all Y values from all series."""
        values = []
        for series in self.series:
            values.extend(series.get_values())
        return values
    
    def summarize(self) -> str:
        """Generate a text summary of the chart."""
        parts = []
        
        if self.title:
            parts.append(f"Chart: {self.title}")
        
        parts.append(f"Type: {self.chart_type.value}")
        
        if self.series:
            parts.append(f"Series: {len(self.series)}")
            total_points = sum(len(s.data_points) for s in self.series)
            parts.append(f"Data points: {total_points}")
        
        if self.analysis:
            if self.analysis.trend:
                parts.append(f"Trend: {self.analysis.trend}")
            if self.analysis.min_value is not None:
                parts.append(f"Range: {self.analysis.min_value:.2f} - {self.analysis.max_value:.2f}")
        
        return "\n".join(parts)


# =============================================================================
# Chart Interpretation Prompts
# =============================================================================

CHART_INTERPRETATION_PROMPT = """Analyze this chart image and extract information.

Describe:
1. CHART TYPE: What type of chart is this? (bar, line, pie, scatter, etc.)
2. TITLE: What is the chart title?
3. AXES: What are the X and Y axis labels and units?
4. DATA: List the data points you can identify (approximate values are fine)
5. TREND: Is there a trend (increasing, decreasing, stable)?
6. KEY INSIGHTS: What are the main takeaways from this chart?

Respond in JSON:
{{
    "chart_type": "bar|line|pie|scatter|area|histogram|unknown",
    "title": "...",
    "x_axis": {{"label": "...", "unit": "..."}},
    "y_axis": {{"label": "...", "unit": "..."}},
    "series": [
        {{
            "name": "Series name",
            "data": [
                {{"x": "label or value", "y": numeric_value}}
            ]
        }}
    ],
    "trend": "increasing|decreasing|stable|fluctuating",
    "insights": ["insight 1", "insight 2"]
}}"""


CHART_QUESTION_PROMPT = """Answer this question about a chart.

CHART INFORMATION:
{chart_info}

QUESTION: {question}

Provide a specific, data-driven answer based on the chart information.
If the answer requires reading specific values, provide them.
If the answer involves a trend or comparison, explain your reasoning.

Answer:"""


# =============================================================================
# Chart Parser
# =============================================================================


class ChartParser:
    """
    Parses charts from images using LLM vision capabilities.
    
    Flow:
    1. Detect chart in image
    2. Use LLM to interpret chart contents
    3. Extract structured data
    4. Analyze trends
    """
    
    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        vision_fn: Callable[[str, bytes], str] | None = None,
    ):
        """
        Initialize the chart parser.
        
        Args:
            llm_fn: LLM function for text interpretation.
            vision_fn: Vision LLM function for image analysis.
        """
        self.llm_fn = llm_fn
        self.vision_fn = vision_fn
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self.llm_fn = llm_fn
    
    def set_vision_function(self, vision_fn: Callable[[str, bytes], str]) -> None:
        """Set the vision function."""
        self.vision_fn = vision_fn
    
    def parse_from_image(
        self,
        image_bytes: bytes,
        doc_id: str = "",
        page_num: int | None = None,
        node_id: str = "",
    ) -> ParsedChart | None:
        """
        Parse a chart from an image.
        
        Args:
            image_bytes: Image data.
            doc_id: Document ID.
            page_num: Page number.
            node_id: Node ID.
            
        Returns:
            ParsedChart or None if parsing fails.
        """
        if not self.vision_fn:
            logger.warning("no_vision_function_configured")
            return None
        
        try:
            # Use vision LLM to interpret the chart
            response = self.vision_fn(CHART_INTERPRETATION_PROMPT, image_bytes)
            
            # Parse the response
            chart = self._parse_interpretation(response)
            
            if chart:
                chart.doc_id = doc_id
                chart.page_num = page_num
                chart.node_id = node_id
                
                # Analyze the chart
                chart.analysis = self._analyze_chart(chart)
                
                logger.info(
                    "chart_parsed",
                    chart_type=chart.chart_type.value,
                    series=len(chart.series),
                )
            
            return chart
            
        except Exception as e:
            logger.warning("chart_parsing_failed", error=str(e))
            return None
    
    def _parse_interpretation(self, response: str) -> ParsedChart | None:
        """Parse LLM interpretation into structured chart."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            chart = ParsedChart(
                title=data.get("title", ""),
                x_axis_label=data.get("x_axis", {}).get("label", ""),
                x_axis_unit=data.get("x_axis", {}).get("unit", ""),
                y_axis_label=data.get("y_axis", {}).get("label", ""),
                y_axis_unit=data.get("y_axis", {}).get("unit", ""),
            )
            
            # Parse chart type
            try:
                chart.chart_type = ChartType(data.get("chart_type", "unknown"))
            except ValueError:
                chart.chart_type = ChartType.UNKNOWN
            
            # Parse series
            for series_data in data.get("series", []):
                series = ChartSeries(name=series_data.get("name", ""))
                
                for point in series_data.get("data", []):
                    dp = DataPoint(
                        x=point.get("x", ""),
                        y=float(point.get("y", 0)),
                    )
                    series.data_points.append(dp)
                
                chart.series.append(series)
            
            # Store insights in analysis
            insights = data.get("insights", [])
            trend = data.get("trend", "")
            
            if insights or trend:
                chart.analysis = ChartAnalysis(
                    trend=trend,
                    key_insights=insights,
                )
            
            return chart
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("chart_interpretation_parse_failed", error=str(e))
            return None
    
    def _analyze_chart(self, chart: ParsedChart) -> ChartAnalysis:
        """Analyze chart data for trends and statistics."""
        analysis = chart.analysis or ChartAnalysis()
        
        all_values = chart.get_all_values()
        
        if not all_values:
            return analysis
        
        # Basic statistics
        analysis.min_value = min(all_values)
        analysis.max_value = max(all_values)
        analysis.avg_value = sum(all_values) / len(all_values)
        
        # Trend detection (for time series)
        if len(all_values) >= 3 and not analysis.trend:
            # Simple trend: compare first third to last third
            third = len(all_values) // 3
            first_avg = sum(all_values[:third]) / third if third > 0 else 0
            last_avg = sum(all_values[-third:]) / third if third > 0 else 0
            
            if last_avg > first_avg * 1.1:
                analysis.trend = "increasing"
            elif last_avg < first_avg * 0.9:
                analysis.trend = "decreasing"
            else:
                # Check for fluctuation
                std_dev = (sum((v - analysis.avg_value) ** 2 for v in all_values) / len(all_values)) ** 0.5
                if std_dev > analysis.avg_value * 0.2:
                    analysis.trend = "fluctuating"
                else:
                    analysis.trend = "stable"
        
        # Generate trend description
        if analysis.trend == "increasing":
            change = ((all_values[-1] - all_values[0]) / all_values[0] * 100) if all_values[0] != 0 else 0
            analysis.trend_description = f"Values show an increasing trend with approximately {change:.1f}% growth"
        elif analysis.trend == "decreasing":
            change = ((all_values[0] - all_values[-1]) / all_values[0] * 100) if all_values[0] != 0 else 0
            analysis.trend_description = f"Values show a decreasing trend with approximately {change:.1f}% decline"
        elif analysis.trend == "stable":
            analysis.trend_description = f"Values remain relatively stable around {analysis.avg_value:.2f}"
        elif analysis.trend == "fluctuating":
            analysis.trend_description = f"Values show significant fluctuation between {analysis.min_value:.2f} and {analysis.max_value:.2f}"
        
        return analysis
    
    def parse_from_description(
        self,
        description: str,
        doc_id: str = "",
        page_num: int | None = None,
    ) -> ParsedChart | None:
        """
        Parse chart from text description.
        
        Useful when chart image is not available but description exists.
        
        Args:
            description: Text description of the chart.
            doc_id: Document ID.
            page_num: Page number.
            
        Returns:
            ParsedChart with extracted information.
        """
        chart = ParsedChart(
            doc_id=doc_id,
            page_num=page_num,
            description=description,
        )
        
        # Try to extract chart type from description
        description_lower = description.lower()
        
        if "bar chart" in description_lower or "bar graph" in description_lower:
            chart.chart_type = ChartType.BAR
        elif "line chart" in description_lower or "line graph" in description_lower:
            chart.chart_type = ChartType.LINE
        elif "pie chart" in description_lower:
            chart.chart_type = ChartType.PIE
        elif "scatter" in description_lower:
            chart.chart_type = ChartType.SCATTER
        
        # Try to extract numbers as data points
        numbers = re.findall(r'(\$?[\d,]+\.?\d*%?)', description)
        if numbers:
            series = ChartSeries(name="Extracted values")
            for i, num in enumerate(numbers[:10]):  # Limit to first 10
                try:
                    value = float(num.replace(',', '').replace('$', '').replace('%', ''))
                    series.data_points.append(DataPoint(x=i, y=value))
                except ValueError:
                    pass
            
            if series.data_points:
                chart.series.append(series)
        
        return chart
    
    def answer_question(
        self,
        chart: ParsedChart,
        question: str,
    ) -> str:
        """
        Answer a question about a chart.
        
        Args:
            chart: The parsed chart.
            question: The question to answer.
            
        Returns:
            Answer string.
        """
        if not self.llm_fn:
            return "LLM not configured for chart Q&A"
        
        # Build chart info
        chart_info = chart.summarize()
        
        if chart.series:
            chart_info += "\n\nData:\n"
            for series in chart.series:
                if series.name:
                    chart_info += f"\n{series.name}:\n"
                for dp in series.data_points[:20]:  # Limit data points
                    chart_info += f"  {dp.x}: {dp.y}\n"
        
        if chart.analysis:
            chart_info += f"\n\nAnalysis:\n{chart.analysis.trend_description}"
            if chart.analysis.key_insights:
                chart_info += "\n\nInsights:\n" + "\n".join(
                    f"- {insight}" for insight in chart.analysis.key_insights
                )
        
        prompt = CHART_QUESTION_PROMPT.format(
            chart_info=chart_info,
            question=question,
        )
        
        try:
            return self.llm_fn(prompt)
        except Exception as e:
            logger.warning("chart_qa_failed", error=str(e))
            return f"Error answering question: {str(e)}"


# =============================================================================
# Convenience Functions
# =============================================================================


def describe_chart(
    chart: ParsedChart,
    include_data: bool = True,
) -> str:
    """Generate a natural language description of a chart."""
    parts = []
    
    if chart.title:
        parts.append(f"This is a {chart.chart_type.value} chart titled '{chart.title}'.")
    else:
        parts.append(f"This is a {chart.chart_type.value} chart.")
    
    if chart.x_axis_label or chart.y_axis_label:
        parts.append(f"The X-axis shows {chart.x_axis_label or 'values'} and the Y-axis shows {chart.y_axis_label or 'values'}.")
    
    if chart.series:
        if len(chart.series) == 1:
            parts.append(f"It contains {len(chart.series[0].data_points)} data points.")
        else:
            parts.append(f"It contains {len(chart.series)} data series.")
        
        if include_data and chart.series[0].data_points:
            sample = chart.series[0].data_points[:3]
            sample_str = ", ".join([f"{dp.x}: {dp.y}" for dp in sample])
            parts.append(f"Sample data: {sample_str}...")
    
    if chart.analysis:
        if chart.analysis.trend_description:
            parts.append(chart.analysis.trend_description)
        
        if chart.analysis.key_insights:
            parts.append("Key insights: " + "; ".join(chart.analysis.key_insights[:2]))
    
    return " ".join(parts)
