"""Tests for Variable Store (pointer-based stitching)."""

import pytest
from rnsr.agent.variable_store import VariableStore, generate_pointer_name


class TestGeneratePointerName:
    """Tests for pointer name generation."""
    
    def test_basic_generation(self):
        pointer = generate_pointer_name("Introduction")
        assert pointer.startswith("$")
        assert "INTRODUCTION" in pointer
    
    def test_multi_word(self):
        pointer = generate_pointer_name("Payment Terms and Conditions")
        assert pointer.startswith("$")
        # Should truncate/simplify
        assert len(pointer) < 50
    
    def test_special_characters(self):
        pointer = generate_pointer_name("Section 1.2: Overview")
        assert pointer.startswith("$")
    
    def test_empty_string(self):
        pointer = generate_pointer_name("")
        assert pointer.startswith("$")


class TestVariableStore:
    """Tests for VariableStore."""
    
    def test_assign_and_resolve(self):
        store = VariableStore()
        store.assign("$INTRO", "This is the introduction.", "node_1")
        
        content = store.resolve("$INTRO")
        assert content == "This is the introduction."
    
    def test_resolve_nonexistent(self):
        store = VariableStore()
        content = store.resolve("$NONEXISTENT")
        assert content is None
    
    def test_multiple_variables(self):
        store = VariableStore()
        store.assign("$VAR1", "Content 1", "node_1")
        store.assign("$VAR2", "Content 2", "node_2")
        store.assign("$VAR3", "Content 3", "node_3")
        
        assert store.resolve("$VAR1") == "Content 1"
        assert store.resolve("$VAR2") == "Content 2"
        assert store.resolve("$VAR3") == "Content 3"
    
    def test_list_pointers(self):
        store = VariableStore()
        store.assign("$ALPHA", "a", "n1")
        store.assign("$BETA", "b", "n2")
        
        pointers = store.list_pointers()
        assert "$ALPHA" in pointers
        assert "$BETA" in pointers
    
    def test_overwrite_variable(self):
        store = VariableStore()
        store.assign("$VAR", "Original", "node_1")
        store.assign("$VAR", "Updated", "node_1")
        
        assert store.resolve("$VAR") == "Updated"
    
    def test_get_metadata(self):
        store = VariableStore()
        store.assign("$INTRO", "Content", "node_123")
        
        metadata = store.get_metadata("$INTRO")
        assert metadata is not None
        assert metadata.source_node_id == "node_123"
    
    def test_clear(self):
        store = VariableStore()
        store.assign("$ALPHA", "a", "n1")
        store.assign("$BETA", "b", "n2")
        
        count = store.clear()
        
        assert count == 2
        assert store.resolve("$ALPHA") is None
        assert store.resolve("$BETA") is None
        assert len(store.list_pointers()) == 0
    
    def test_count(self):
        store = VariableStore()
        assert store.count() == 0
        
        store.assign("$ALPHA", "a", "n1")
        assert store.count() == 1
        
        store.assign("$BETA", "b", "n2")
        assert store.count() == 2
    
    def test_exists(self):
        store = VariableStore()
        store.assign("$EXISTS", "content", "node")
        
        assert store.exists("$EXISTS") is True
        assert store.exists("$NOTEXISTS") is False
    
    def test_delete(self):
        store = VariableStore()
        store.assign("$TODELETE", "content", "node")
        
        assert store.exists("$TODELETE") is True
        
        result = store.delete("$TODELETE")
        assert result is True
        assert store.exists("$TODELETE") is False
