"""
RNSR Data Models

Pydantic models for all data structures in the pipeline.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Ingestion Models
# =============================================================================


class BoundingBox(BaseModel):
    """Bounding box for a text element on a page."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


class SpanInfo(BaseModel):
    """Information about a single text span from PyMuPDF."""

    text: str
    font_size: float
    font_name: str
    is_bold: bool = False
    is_italic: bool = False
    bbox: BoundingBox
    page_num: int


class FontAnalysis(BaseModel):
    """Results of font histogram analysis."""

    body_size: float
    header_threshold: float
    size_histogram: dict[float, int]
    span_count: int
    unique_sizes: int


class ClassifiedSpan(SpanInfo):
    """A span with its classification (header level or body)."""

    role: Literal["header", "body", "caption", "footnote"] = "body"
    header_level: int = 0  # 0 = not a header, 1-3 = H1-H3


class DocumentNode(BaseModel):
    """A node in the document tree structure."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    level: int  # 0 = root, 1 = H1, 2 = H2, 3 = H3
    header: str = ""
    content: str = ""  # Full text content
    page_num: int | None = None
    bbox: BoundingBox | None = None
    children: list[DocumentNode] = Field(default_factory=list)

    @property
    def child_ids(self) -> list[str]:
        return [child.id for child in self.children]


class DocumentTree(BaseModel):
    """Complete document tree structure."""

    id: str = Field(default_factory=lambda: f"doc_{str(uuid4())[:8]}")
    title: str = ""
    root: DocumentNode
    total_nodes: int = 0
    ingestion_tier: Literal[1, 2, 3] = 1
    ingestion_method: IngestionMethod | None = None


# =============================================================================
# Indexing Models
# =============================================================================


class SkeletonNode(BaseModel):
    """A lightweight node for the skeleton index."""

    node_id: str
    parent_id: str | None
    level: int
    header: str
    summary: str  # 50-100 words max - this goes in vector store
    child_ids: list[str]
    page_num: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Result of document ingestion including metadata."""

    tree: DocumentTree
    tier_used: Literal[1, 2, 3]
    method: IngestionMethod
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Agent Models
# =============================================================================


class StoredVariable(BaseModel):
    """Metadata about a stored variable in the VariableStore."""

    pointer: str  # e.g., "$LIABILITY_CLAUSE"
    source_node_id: str
    content_hash: str
    char_count: int
    created_at: str


class TraceEntry(BaseModel):
    """A single entry in the retrieval trace log."""

    timestamp: str
    node_type: Literal["decomposition", "navigation", "variable_stitching", "synthesis"]
    action: str
    details: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    """Complete trace of agent's retrieval process."""

    query: str
    total_steps: int
    nodes_visited: list[str]
    nodes_rejected: list[dict[str, str]]
    variables_stored: list[str]
    final_path: str
    entries: list[TraceEntry] = Field(default_factory=list)

# Define the type alias for all valid ingestion methods
IngestionMethod = Literal[
    "font_histogram",
    "semantic_splitter", 
    "ocr",
    "xy_cut",
    "hierarchical_clustering",
]
