"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like content structure for testing."""
    return {
        "pages": [
            {
                "page_num": 0,
                "blocks": [
                    {"text": "Document Title", "font_size": 28.0, "is_bold": True},
                    {"text": "Chapter 1: Introduction", "font_size": 24.0, "is_bold": True},
                    {"text": "This is the introduction paragraph.", "font_size": 12.0, "is_bold": False},
                ],
            },
            {
                "page_num": 1,
                "blocks": [
                    {"text": "Section 1.1: Overview", "font_size": 18.0, "is_bold": True},
                    {"text": "Overview content goes here.", "font_size": 12.0, "is_bold": False},
                    {"text": "More paragraph text.", "font_size": 12.0, "is_bold": False},
                ],
            },
        ]
    }


@pytest.fixture
def mock_skeleton():
    """Create a mock skeleton for agent testing."""
    from rnsr.models import SkeletonNode
    
    return {
        "root": SkeletonNode(
            node_id="root",
            header="Document",
            summary="A sample document for testing.",
            level=0,
            parent_id=None,
            child_ids=["ch1", "ch2"],
        ),
        "ch1": SkeletonNode(
            node_id="ch1",
            header="Chapter 1: Introduction",
            summary="Introduction to the topic.",
            level=1,
            parent_id="root",
            child_ids=["sec1_1"],
        ),
        "ch2": SkeletonNode(
            node_id="ch2",
            header="Chapter 2: Methods",
            summary="Description of methods used.",
            level=1,
            parent_id="root",
            child_ids=[],
        ),
        "sec1_1": SkeletonNode(
            node_id="sec1_1",
            header="Section 1.1: Background",
            summary="Background information.",
            level=2,
            parent_id="ch1",
            child_ids=[],
        ),
    }


@pytest.fixture
def mock_kv_store():
    """Create a mock KV store with content."""
    from rnsr.indexing.kv_store import InMemoryKVStore
    
    store = InMemoryKVStore()
    store.put("root", "Document\n\nThis is the full document content.")
    store.put("ch1", "Chapter 1: Introduction\n\nIntroduction text here.")
    store.put("ch2", "Chapter 2: Methods\n\nMethods description here.")
    store.put("sec1_1", "Section 1.1: Background\n\nBackground details.")
    
    return store
