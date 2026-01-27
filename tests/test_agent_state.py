"""Tests for Agent State and Graph."""

import pytest
from rnsr.agent.graph import (
    AgentState,
    create_initial_state,
    add_trace_entry,
)
from rnsr.agent.variable_store import generate_pointer_name


class TestCreateInitialState:
    """Tests for initial state creation."""
    
    def test_basic_creation(self):
        state = create_initial_state("What are the payment terms?", "root")
        
        assert state["question"] == "What are the payment terms?"
        assert state["sub_questions"] == []
        assert state["pending_questions"] == []
        assert state["visited_nodes"] == []
        assert state["variables"] == []
        assert state["iteration"] == 0
    
    def test_with_root_node(self):
        state = create_initial_state(
            "What is the refund policy?",
            root_node_id="doc_root",
        )
        
        assert state["current_node_id"] == "doc_root"
        assert state["navigation_path"] == ["doc_root"]
    
    def test_custom_max_iterations(self):
        state = create_initial_state(
            "Complex question",
            root_node_id="root",
            max_iterations=50,
        )
        
        assert state["max_iterations"] == 50
    
    def test_default_values(self):
        state = create_initial_state("Simple question", "root")
        
        assert state["answer"] is None
        assert state["confidence"] == 0.0
        assert state["context"] == ""
        assert state["trace"] == []


class TestAddTraceEntry:
    """Tests for trace logging."""
    
    def test_add_trace(self):
        state = create_initial_state("Test", "root")
        
        add_trace_entry(state, "navigation", "Moved to node X")
        
        assert len(state["trace"]) == 1
        assert state["trace"][0]["node_type"] == "navigation"
        assert state["trace"][0]["action"] == "Moved to node X"
    
    def test_multiple_traces(self):
        state = create_initial_state("Test", "root")
        
        add_trace_entry(state, "decomposition", "Split into 3 questions")
        add_trace_entry(state, "navigation", "Starting at root")
        add_trace_entry(state, "variable_stitching", "Fetched content")
        
        assert len(state["trace"]) == 3
    
    def test_trace_with_metadata(self):
        state = create_initial_state("Test", "root")
        
        add_trace_entry(
            state,
            "variable_stitching",
            "Stored $PAYMENT_TERMS",
            {"pointer": "$PAYMENT_TERMS", "chars": 500},
        )
        
        entry = state["trace"][0]
        assert entry["details"]["pointer"] == "$PAYMENT_TERMS"
        assert entry["details"]["chars"] == 500


class TestGeneratePointerName:
    """Tests for pointer name generation."""
    
    def test_simple_name(self):
        pointer = generate_pointer_name("Introduction")
        assert pointer.startswith("$")
        assert "INTRODUCTION" in pointer
    
    def test_multi_word(self):
        pointer = generate_pointer_name("Payment Terms")
        assert "$" in pointer
        assert "PAYMENT" in pointer.upper() or "TERMS" in pointer.upper()
    
    def test_with_numbers(self):
        pointer = generate_pointer_name("Section 3.2.1")
        assert pointer.startswith("$")
    
    def test_max_length(self):
        long_name = "This is a very long section title that goes on and on"
        pointer = generate_pointer_name(long_name)
        assert len(pointer) <= 50


class TestAgentStateTyping:
    """Type safety tests for AgentState."""
    
    def test_all_required_fields_present(self):
        state = create_initial_state("Test question", "root")
        
        # All TypedDict fields should be present
        required_fields = [
            "question",
            "sub_questions",
            "pending_questions",
            "current_sub_question",
            "current_node_id",
            "visited_nodes",
            "navigation_path",
            # Tree of Thoughts (ToT) fields
            "scored_candidates",
            "backtrack_stack",
            "dead_ends",
            "top_k",
            # Variable stitching
            "variables",
            "context",
            "answer",
            "confidence",
            "trace",
            "iteration",
            "max_iterations",
        ]
        
        for field in required_fields:
            assert field in state, f"Missing field: {field}"
    
    def test_tot_fields_initialized(self):
        """Test that Tree of Thoughts fields are properly initialized."""
        state = create_initial_state("Test question", "root")
        
        assert state["scored_candidates"] == []
        assert state["backtrack_stack"] == []
        assert state["dead_ends"] == []
        assert state["top_k"] == 3  # Default value
    
    def test_custom_top_k(self):
        """Test custom top_k for ToT selection."""
        state = create_initial_state("Test question", "root", top_k=5)
        
        assert state["top_k"] == 5
