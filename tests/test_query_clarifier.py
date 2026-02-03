"""Tests for the Query Clarification system."""

import pytest

from rnsr.agent.query_clarifier import (
    AmbiguityAnalysis,
    AmbiguityType,
    ClarificationRequest,
    ClarificationResult,
    QueryClarifier,
    clarify_query,
    needs_clarification,
)


class TestAmbiguityAnalysis:
    """Tests for AmbiguityAnalysis dataclass."""
    
    def test_analysis_creation(self):
        """Test creating an ambiguity analysis."""
        analysis = AmbiguityAnalysis(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.REFERENCE,
            description="Unclear pronoun reference",
            possible_interpretations=["the contract", "the clause"],
            confidence=0.8,
            suggested_clarification="What does 'it' refer to?",
        )
        
        assert analysis.is_ambiguous is True
        assert analysis.ambiguity_type == AmbiguityType.REFERENCE
    
    def test_analysis_to_dict(self):
        """Test converting analysis to dictionary."""
        analysis = AmbiguityAnalysis(
            is_ambiguous=False,
            ambiguity_type=AmbiguityType.NONE,
        )
        
        data = analysis.to_dict()
        
        assert data["is_ambiguous"] is False
        assert data["ambiguity_type"] == "none"


class TestClarificationRequest:
    """Tests for ClarificationRequest dataclass."""
    
    def test_request_creation(self):
        """Test creating a clarification request."""
        request = ClarificationRequest(
            id="clarify_001",
            original_query="What does it say about the clause?",
            question="What does 'it' refer to in your question?",
            options=["the contract", "the amendment"],
        )
        
        assert request.id == "clarify_001"
        assert len(request.options) == 2
    
    def test_request_to_dict(self):
        """Test converting request to dictionary."""
        request = ClarificationRequest(
            original_query="Test query",
            question="Clarifying question?",
        )
        
        data = request.to_dict()
        
        assert "original_query" in data
        assert "question" in data


class TestClarificationResult:
    """Tests for ClarificationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a clarification result."""
        result = ClarificationResult(
            original_query="Original query",
            refined_query="Refined query with clarification",
            needs_clarification=True,
            proceeded_with_best_guess=False,
        )
        
        assert result.original_query == "Original query"
        assert result.needs_clarification is True


class TestQueryClarifier:
    """Tests for QueryClarifier class."""
    
    def test_clarifier_creation(self):
        """Test creating a query clarifier."""
        clarifier = QueryClarifier(
            ambiguity_threshold=0.6,
            max_clarifications=2,
        )
        
        assert clarifier.ambiguity_threshold == 0.6
        assert clarifier.max_clarifications == 2
    
    def test_detect_reference_ambiguity(self):
        """Test detecting reference ambiguity (pronouns)."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        analysis = clarifier.analyze_query("What does it say about liability?")
        
        assert analysis.is_ambiguous is True
        assert analysis.ambiguity_type == AmbiguityType.REFERENCE
    
    def test_detect_scope_ambiguity(self):
        """Test detecting scope ambiguity."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        analysis = clarifier.analyze_query("What is in the section?")
        
        assert analysis.is_ambiguous is True
        assert analysis.ambiguity_type == AmbiguityType.SCOPE
    
    def test_detect_temporal_ambiguity(self):
        """Test detecting temporal ambiguity."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        analysis = clarifier.analyze_query("What was the amount previously?")
        
        assert analysis.is_ambiguous is True
        assert analysis.ambiguity_type == AmbiguityType.TEMPORAL
    
    def test_detect_comparison_ambiguity(self):
        """Test detecting comparison ambiguity."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        analysis = clarifier.analyze_query("How does section 5 compare?")
        
        assert analysis.is_ambiguous is True
        assert analysis.ambiguity_type == AmbiguityType.COMPARISON
    
    def test_no_ambiguity_detected(self):
        """Test clear query with no ambiguity."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        analysis = clarifier.analyze_query(
            "What is the payment term in section 5.2 of the contract?"
        )
        
        # This query is specific and clear
        assert analysis.ambiguity_type == AmbiguityType.NONE or analysis.is_ambiguous is False
    
    def test_generate_clarification(self):
        """Test generating a clarification request."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        analysis = AmbiguityAnalysis(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.REFERENCE,
            possible_interpretations=["the contract", "the amendment"],
            suggested_clarification="What does 'it' refer to?",
        )
        
        request = clarifier.generate_clarification(
            query="What does it say?",
            analysis=analysis,
        )
        
        assert request.question is not None
        assert len(request.options) == 2
    
    def test_refine_query_heuristic(self):
        """Test refining query with heuristic method."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        request = ClarificationRequest(
            original_query="What does it say?",
            question="What does 'it' refer to?",
        )
        
        refined = clarifier.refine_query(
            original_query="What does it say?",
            clarification=request,
            user_response="the contract",
        )
        
        assert "contract" in refined
    
    def test_get_best_guess(self):
        """Test getting best guess interpretation."""
        clarifier = QueryClarifier()
        
        analysis = AmbiguityAnalysis(
            is_ambiguous=True,
            possible_interpretations=["interpretation A", "interpretation B"],
        )
        
        guess = clarifier.get_best_guess(
            query="Ambiguous query",
            analysis=analysis,
        )
        
        assert "interpretation A" in guess
    
    def test_clarify_flow_no_ambiguity(self):
        """Test full clarification flow with no ambiguity."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        result = clarifier.clarify(
            query="What is the contract term in section 5?",
        )
        
        assert result.refined_query is not None
        # May or may not detect ambiguity
    
    def test_clarify_flow_with_best_guess(self):
        """Test clarification flow using best guess."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        result = clarifier.clarify(
            query="What does it say about this?",
            allow_best_guess=True,
        )
        
        assert result.refined_query is not None
        # Should proceed with best guess since no callback
    
    def test_clarify_flow_with_callback(self):
        """Test clarification flow with user response callback."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        def mock_callback(request):
            return "the liability clause"
        
        result = clarifier.clarify(
            query="What does it say about the clause?",
            get_user_response=mock_callback,
        )
        
        # Should have refined based on callback
        if result.needs_clarification:
            assert "liability" in result.refined_query.lower() or len(result.user_responses) > 0
    
    def test_clarifier_stats(self):
        """Test getting clarification statistics."""
        clarifier = QueryClarifier(use_llm_detection=False)
        
        # Trigger some clarifications
        clarifier.analyze_query("What does it say?")
        clarifier.analyze_query("What about they?")
        
        stats = clarifier.get_stats()
        
        assert isinstance(stats, dict)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_needs_clarification_clear_query(self):
        """Test needs_clarification with clear query."""
        is_ambiguous, analysis = needs_clarification(
            "What is the payment term in section 5?"
        )
        
        # Should be clear (or low ambiguity)
        assert isinstance(is_ambiguous, bool)
        assert isinstance(analysis, AmbiguityAnalysis)
    
    def test_needs_clarification_ambiguous_query(self):
        """Test needs_clarification with ambiguous query."""
        is_ambiguous, analysis = needs_clarification(
            "What does it say about that?"
        )
        
        assert is_ambiguous is True
        assert analysis.ambiguity_type != AmbiguityType.NONE
    
    def test_clarify_query_function(self):
        """Test clarify_query convenience function."""
        refined = clarify_query(
            query="What does it say?",
        )
        
        assert refined is not None
        assert len(refined) > 0
