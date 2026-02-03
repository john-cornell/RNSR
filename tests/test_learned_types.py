"""Tests for the Adaptive Learning registries."""

import tempfile
from pathlib import Path

import pytest

from rnsr.extraction.learned_types import (
    LearnedRelationshipTypeRegistry,
    LearnedTypeRegistry,
)
from rnsr.extraction.entity_linker import LearnedNormalizationPatterns
from rnsr.agent.rlm_navigator import LearnedQueryPatterns, LearnedStopWords
from rnsr.ingestion.header_classifier import LearnedHeaderThresholds


class TestLearnedTypeRegistry:
    """Tests for LearnedTypeRegistry (entity types)."""
    
    @pytest.fixture
    def temp_registry(self, temp_dir):
        """Create a temporary registry."""
        return LearnedTypeRegistry(
            storage_path=temp_dir / "learned_types.json",
            auto_save=False,
        )
    
    def test_registry_creation(self, temp_dir):
        """Test creating a registry."""
        registry = LearnedTypeRegistry(
            storage_path=temp_dir / "types.json",
        )
        assert registry is not None
    
    def test_record_type(self, temp_registry):
        """Test recording a new type."""
        temp_registry.record_type(
            type_name="MEDICAL_DEVICE",
            context="The patient was fitted with a pacemaker.",
            entity_name="pacemaker",
        )
        
        type_data = temp_registry.get_type("medical_device")
        
        assert type_data is not None
        assert type_data["count"] >= 1
    
    def test_record_type_with_count(self, temp_registry):
        """Test that type counts are tracked."""
        temp_registry.record_type("VEHICLE_MODEL", entity_name="Tesla Model 3")
        temp_registry.record_type("VEHICLE_MODEL", entity_name="Ford F-150")
        temp_registry.record_type("VEHICLE_MODEL", entity_name="Tesla Model S")
        
        type_data = temp_registry.get_type("vehicle_model")
        
        assert type_data["count"] == 3
    
    def test_get_learned_types(self, temp_registry):
        """Test getting learned types."""
        temp_registry.record_type("AIRCRAFT", entity_name="Boeing 747")
        temp_registry.record_type("AIRCRAFT", entity_name="Airbus A380")
        
        types = temp_registry.get_learned_types(min_count=1)
        
        assert len(types) >= 1
        # types is a list of dicts with "name" key
        type_names = [t["name"] if isinstance(t, dict) else t for t in types]
        assert "aircraft" in type_names
    
    def test_persistence(self, temp_dir):
        """Test registry persistence."""
        storage_path = temp_dir / "persist_types.json"
        
        # Create and populate
        reg1 = LearnedTypeRegistry(storage_path=storage_path, auto_save=True)
        reg1.record_type("TEST_TYPE", entity_name="example1")
        
        # Create new instance
        reg2 = LearnedTypeRegistry(storage_path=storage_path)
        
        type_data = reg2.get_type("test_type")
        assert type_data is not None


class TestLearnedRelationshipTypeRegistry:
    """Tests for LearnedRelationshipTypeRegistry."""
    
    @pytest.fixture
    def temp_registry(self, temp_dir):
        """Create a temporary registry."""
        return LearnedRelationshipTypeRegistry(
            storage_path=temp_dir / "learned_rel_types.json",
            auto_save=False,
        )
    
    def test_registry_creation(self, temp_dir):
        """Test creating a registry."""
        registry = LearnedRelationshipTypeRegistry(
            storage_path=temp_dir / "rel_types.json",
        )
        assert registry is not None
    
    def test_record_relationship_type(self, temp_registry):
        """Test recording a new relationship type."""
        temp_registry.record_type(
            type_name="MANUFACTURED_BY",
            context="iPhone manufactured by Apple",
            relationship_description="Product manufactured by Organization",
        )
        
        # Check the internal storage
        assert "manufactured_by" in temp_registry._types
    
    def test_types_storage(self, temp_registry):
        """Test that types are stored correctly."""
        temp_registry.record_type(
            type_name="LICENSED_TO",
            context="Technology licensed to company",
        )
        
        # Check internal storage
        assert len(temp_registry._types) >= 1
        assert "licensed_to" in temp_registry._types


class TestLearnedNormalizationPatterns:
    """Tests for LearnedNormalizationPatterns."""
    
    @pytest.fixture
    def temp_registry(self, temp_dir):
        """Create a temporary registry."""
        return LearnedNormalizationPatterns(
            storage_path=temp_dir / "normalization.json",
            auto_save=False,
        )
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = LearnedNormalizationPatterns()
        assert registry is not None
    
    def test_base_titles_included(self, temp_registry):
        """Test that base titles are present."""
        # The class has BASE_TITLES
        assert len(LearnedNormalizationPatterns.BASE_TITLES) > 0
        assert "mr." in LearnedNormalizationPatterns.BASE_TITLES or "mr" in [t.lower() for t in LearnedNormalizationPatterns.BASE_TITLES]
    
    def test_base_suffixes_included(self, temp_registry):
        """Test that base suffixes are present."""
        # The class has BASE_SUFFIXES
        assert len(LearnedNormalizationPatterns.BASE_SUFFIXES) > 0


class TestLearnedStopWords:
    """Tests for LearnedStopWords."""
    
    @pytest.fixture
    def temp_registry(self, temp_dir):
        """Create a temporary registry."""
        return LearnedStopWords(
            storage_path=temp_dir / "stop_words.json",
            auto_save=False,
        )
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = LearnedStopWords()
        assert registry is not None
    
    def test_base_stop_words_included(self, temp_registry):
        """Test that base stop words are present."""
        # The class has BASE_STOP_WORDS
        assert len(LearnedStopWords.BASE_STOP_WORDS) > 0
        assert "the" in LearnedStopWords.BASE_STOP_WORDS
    
    def test_is_stop_word_base(self, temp_registry):
        """Test checking base stop words."""
        # "the" should be in base stop words
        assert "the" in LearnedStopWords.BASE_STOP_WORDS


class TestLearnedQueryPatterns:
    """Tests for LearnedQueryPatterns."""
    
    @pytest.fixture
    def temp_registry(self, temp_dir):
        """Create a temporary registry."""
        return LearnedQueryPatterns(
            storage_path=temp_dir / "query_patterns.json",
            auto_save=False,
        )
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = LearnedQueryPatterns()
        assert registry is not None


class TestLearnedHeaderThresholds:
    """Tests for LearnedHeaderThresholds."""
    
    @pytest.fixture
    def temp_registry(self, temp_dir):
        """Create a temporary registry."""
        return LearnedHeaderThresholds(
            storage_path=temp_dir / "header_thresholds.json",
            auto_save=False,
        )
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = LearnedHeaderThresholds()
        assert registry is not None
    
    def test_default_thresholds(self):
        """Test that default thresholds are present."""
        assert LearnedHeaderThresholds.DEFAULT_H1_MIN > 0
        assert LearnedHeaderThresholds.DEFAULT_H2_MIN > 0
        assert LearnedHeaderThresholds.DEFAULT_H3_MIN > 0


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
