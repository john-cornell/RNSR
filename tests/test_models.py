"""Tests for data models."""

import pytest
from rnsr.models import (
    BoundingBox,
    SpanInfo,
    DocumentNode,
    DocumentTree,
    FontAnalysis,
    ClassifiedSpan,
    SkeletonNode,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""
    
    def test_creation(self):
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=50.0)
        assert bbox.x0 == 0.0
        assert bbox.y1 == 50.0
    
    def test_width_height(self):
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        assert bbox.width == 100.0
        assert bbox.height == 50.0
    
    def test_area(self):
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=10.0, y1=10.0)
        # Area computed as width * height
        assert bbox.width * bbox.height == 100.0


class TestSpanInfo:
    """Tests for SpanInfo model."""
    
    def test_creation(self):
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0)
        span = SpanInfo(
            text="Hello World",
            font_size=12.0,
            font_name="Arial",
            is_bold=False,
            is_italic=False,
            bbox=bbox,
            page_num=0,
        )
        assert span.text == "Hello World"
        assert span.font_size == 12.0
        assert span.is_bold is False


class TestDocumentNode:
    """Tests for DocumentNode model."""
    
    def test_creation(self):
        node = DocumentNode(
            level=1,
            header="Introduction",
            content="This is the introduction.",
            page_num=0,
        )
        assert node.level == 1
        assert node.header == "Introduction"
        assert node.children == []
    
    def test_auto_id(self):
        node1 = DocumentNode(level=1, header="Node 1")
        node2 = DocumentNode(level=1, header="Node 2")
        
        # Each should get a unique ID
        assert node1.id != node2.id
    
    def test_add_child(self):
        parent = DocumentNode(
            level=1,
            header="Chapter 1",
        )
        child = DocumentNode(
            level=2,
            header="Section 1.1",
        )
        parent.children.append(child)
        
        assert len(parent.children) == 1
        assert parent.child_ids[0] == child.id


class TestDocumentTree:
    """Tests for DocumentTree model."""
    
    def test_creation(self):
        root = DocumentNode(level=0, header="Document Root")
        tree = DocumentTree(
            title="Test Document",
            root=root,
            total_nodes=1,
        )
        assert tree.title == "Test Document"
        assert tree.root.level == 0
    
    def test_with_nested_nodes(self):
        child = DocumentNode(level=2, header="Section")
        parent = DocumentNode(level=1, header="Chapter", children=[child])
        root = DocumentNode(level=0, header="Document", children=[parent])
        
        tree = DocumentTree(
            root=root,
            total_nodes=3,
        )
        assert tree.total_nodes == 3
        assert len(tree.root.children) == 1
        assert len(tree.root.children[0].children) == 1


class TestFontAnalysis:
    """Tests for FontAnalysis model."""
    
    def test_creation(self):
        analysis = FontAnalysis(
            body_size=12.0,
            header_threshold=1.2,
            size_histogram={12.0: 50, 18.0: 10, 24.0: 5},
            span_count=100,
            unique_sizes=5,
        )
        assert analysis.body_size == 12.0
        assert analysis.header_threshold == 1.2
        assert analysis.span_count == 100


class TestClassifiedSpan:
    """Tests for ClassifiedSpan model."""
    
    def test_header_classification(self):
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=200.0, y1=30.0)
        span = ClassifiedSpan(
            text="Chapter 1",
            font_size=24.0,
            font_name="Arial-Bold",
            is_bold=True,
            is_italic=False,
            bbox=bbox,
            page_num=0,
            role="header",
            header_level=1,
        )
        assert span.role == "header"
        assert span.header_level == 1
    
    def test_body_classification(self):
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0)
        span = ClassifiedSpan(
            text="Regular paragraph text.",
            font_size=12.0,
            font_name="Arial",
            is_bold=False,
            is_italic=False,
            bbox=bbox,
            page_num=0,
            role="body",
            header_level=0,
        )
        assert span.role == "body"
        assert span.header_level == 0


class TestSkeletonNode:
    """Tests for SkeletonNode model."""
    
    def test_creation(self):
        node = SkeletonNode(
            node_id="ch1",
            header="Chapter 1: Introduction",
            summary="This chapter introduces the topic.",
            level=1,
            parent_id="root",
            child_ids=["sec1_1", "sec1_2"],
        )
        assert node.node_id == "ch1"
        assert node.level == 1
        assert len(node.child_ids) == 2
