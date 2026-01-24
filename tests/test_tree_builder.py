"""Tests for Tree Builder."""

import pytest
from rnsr.models import DocumentNode, DocumentTree


class TestDocumentTreeStructure:
    """Tests for DocumentTree structure."""
    
    def test_nested_structure(self):
        """Test creating a nested document structure."""
        sec1 = DocumentNode(
            level=2,
            header="Section 1.1",
            content="Section content",
            page_num=1,
        )
        ch1 = DocumentNode(
            level=1,
            header="Chapter 1",
            content="Chapter intro",
            page_num=0,
            children=[sec1],
        )
        root = DocumentNode(
            level=0,
            header="Document",
            children=[ch1],
        )
        
        tree = DocumentTree(
            title="Test Document",
            root=root,
            total_nodes=3,
        )
        
        assert tree.root.level == 0
        assert len(tree.root.children) == 1
        assert tree.root.children[0].header == "Chapter 1"
        assert len(tree.root.children[0].children) == 1
        assert tree.root.children[0].children[0].header == "Section 1.1"
    
    def test_child_ids_property(self):
        """Test the child_ids computed property."""
        child1 = DocumentNode(level=2, header="Child 1")
        child2 = DocumentNode(level=2, header="Child 2")
        parent = DocumentNode(
            level=1,
            header="Parent",
            children=[child1, child2],
        )
        
        assert len(parent.child_ids) == 2
        assert child1.id in parent.child_ids
        assert child2.id in parent.child_ids


class TestTreeTraversal:
    """Tests for tree traversal utilities."""
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample tree for traversal tests."""
        sec1 = DocumentNode(
            level=2,
            header="Section 1.1",
            page_num=1,
        )
        ch1 = DocumentNode(
            level=1,
            header="Chapter 1",
            page_num=0,
            children=[sec1],
        )
        ch2 = DocumentNode(
            level=1,
            header="Chapter 2",
            page_num=5,
        )
        root = DocumentNode(
            level=0,
            header="Document",
            children=[ch1, ch2],
        )
        
        return DocumentTree(
            title="Sample",
            root=root,
            total_nodes=4,
        )
    
    def test_traverse_children(self, sample_tree):
        root = sample_tree.root
        assert len(root.children) == 2
        
        child_headers = [c.header for c in root.children]
        assert "Chapter 1" in child_headers
        assert "Chapter 2" in child_headers
    
    def test_nested_access(self, sample_tree):
        ch1 = sample_tree.root.children[0]
        assert ch1.header == "Chapter 1"
        
        sec1 = ch1.children[0]
        assert sec1.header == "Section 1.1"
