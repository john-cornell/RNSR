"""Tests for the Self-Reflection Loop system."""

import pytest

from rnsr.agent.self_reflection import (
    CritiqueResult,
    Issue,
    IssueSeverity,
    IssueType,
    ReflectionIteration,
    ReflectionResult,
    SelfReflectionEngine,
)


class TestIssue:
    """Tests for Issue dataclass."""
    
    def test_issue_creation(self):
        """Test creating an issue."""
        issue = Issue(
            type=IssueType.ACCURACY,
            description="The answer contradicts the evidence.",
            severity=IssueSeverity.HIGH,
        )
        
        assert issue.type == IssueType.ACCURACY
        assert issue.severity == IssueSeverity.HIGH
    
    def test_issue_to_dict(self):
        """Test converting issue to dictionary."""
        issue = Issue(
            type=IssueType.COMPLETENESS,
            description="Missing information.",
            severity=IssueSeverity.MEDIUM,
        )
        
        data = issue.to_dict()
        
        assert data["type"] == "completeness"
        assert data["severity"] == "medium"


class TestCritiqueResult:
    """Tests for CritiqueResult dataclass."""
    
    def test_critique_result_creation(self):
        """Test creating a critique result."""
        issues = [
            Issue(IssueType.ACCURACY, "Inaccurate claim", IssueSeverity.HIGH),
        ]
        
        result = CritiqueResult(
            has_issues=True,
            issues=issues,
            confidence=0.8,
            suggested_improvements=["Verify against source"],
            should_retry=True,
        )
        
        assert result.has_issues is True
        assert len(result.issues) == 1
        assert result.should_retry is True
    
    def test_critique_result_no_issues(self):
        """Test critique result with no issues."""
        result = CritiqueResult(
            has_issues=False,
            issues=[],
            confidence=0.95,
            should_retry=False,
        )
        
        assert result.has_issues is False
        assert result.issues == []


class TestReflectionResult:
    """Tests for ReflectionResult dataclass."""
    
    def test_reflection_result_creation(self):
        """Test creating a reflection result."""
        result = ReflectionResult(
            original_answer="Original answer",
            final_answer="Improved answer",
            question="What is the answer?",
            total_iterations=2,
            improved=True,
            final_confidence=0.9,
        )
        
        assert result.original_answer == "Original answer"
        assert result.final_answer == "Improved answer"
        assert result.improved is True
    
    def test_reflection_result_to_dict(self):
        """Test converting reflection result to dictionary."""
        result = ReflectionResult(
            original_answer="Original",
            final_answer="Final",
            question="Question?",
        )
        
        data = result.to_dict()
        
        assert data["original_answer"] == "Original"
        assert data["final_answer"] == "Final"


class TestSelfReflectionEngine:
    """Tests for SelfReflectionEngine class."""
    
    def test_engine_creation(self):
        """Test creating a reflection engine."""
        engine = SelfReflectionEngine(
            max_iterations=3,
            min_confidence_threshold=0.8,
        )
        
        assert engine.max_iterations == 3
        assert engine.min_confidence_threshold == 0.8
    
    def test_engine_without_llm_returns_original(self):
        """Test that engine returns original answer when no LLM configured."""
        engine = SelfReflectionEngine(llm_fn=None)
        
        result = engine.reflect(
            answer="Original answer",
            question="What is the question?",
        )
        
        assert result.final_answer == "Original answer"
        assert result.improved is False
    
    def test_engine_with_mock_llm_no_issues(self):
        """Test reflection with mock LLM that finds no issues."""
        def mock_llm(prompt):
            return '{"has_issues": false, "issues": [], "confidence_in_critique": 0.9, "should_retry": false}'
        
        engine = SelfReflectionEngine(llm_fn=mock_llm)
        
        result = engine.reflect(
            answer="The answer is correct.",
            question="What is the answer?",
        )
        
        assert result.final_answer == "The answer is correct."
        assert result.total_iterations == 1
    
    def test_engine_with_mock_llm_finds_issues(self):
        """Test reflection with mock LLM that finds issues."""
        call_count = 0
        
        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            
            if "Critically evaluate" in prompt:
                # Critique prompt
                if call_count == 1:
                    return '''{"has_issues": true, "issues": [{"type": "accuracy", "description": "Inaccurate", "severity": "medium"}], "confidence_in_critique": 0.7, "should_retry": true, "suggested_improvements": ["Fix it"]}'''
                else:
                    return '{"has_issues": false, "issues": [], "confidence_in_critique": 0.9, "should_retry": false}'
            elif "improving" in prompt.lower() or "improved" in prompt.lower():
                # Refinement prompt
                return "Improved answer with corrections."
            elif "Compare" in prompt:
                # Verification prompt
                return '{"better_answer": "B", "confidence": 0.8, "reasoning": "B is better"}'
            
            return '{"has_issues": false}'
        
        engine = SelfReflectionEngine(
            llm_fn=mock_llm,
            max_iterations=3,
        )
        
        result = engine.reflect(
            answer="Original answer",
            question="What is the question?",
        )
        
        assert result.total_iterations >= 1
    
    def test_engine_respects_max_iterations(self):
        """Test that engine respects max_iterations limit."""
        iteration_count = 0
        
        def mock_llm(prompt):
            nonlocal iteration_count
            iteration_count += 1
            # Always report issues to force max iterations
            return '{"has_issues": true, "issues": [{"type": "accuracy", "description": "Issue", "severity": "low"}], "confidence_in_critique": 0.5, "should_retry": true}'
        
        engine = SelfReflectionEngine(
            llm_fn=mock_llm,
            max_iterations=2,
            enable_verification=False,
        )
        
        result = engine.reflect(
            answer="Answer",
            question="Question?",
        )
        
        assert result.total_iterations <= 2
    
    def test_parse_critique_no_issues_found(self):
        """Test parsing 'NO ISSUES FOUND' response."""
        engine = SelfReflectionEngine()
        
        result = engine._parse_critique("Based on my analysis: NO ISSUES FOUND with this answer.")
        
        assert result.has_issues is False
        assert result.confidence >= 0.9
    
    def test_parse_critique_json_response(self):
        """Test parsing JSON critique response."""
        engine = SelfReflectionEngine()
        
        response = '''
        Here is my analysis:
        ```json
        {
            "has_issues": true,
            "issues": [
                {"type": "accuracy", "description": "Inaccurate claim", "severity": "high"}
            ],
            "confidence_in_critique": 0.85,
            "should_retry": true,
            "suggested_improvements": ["Verify the claim"]
        }
        ```
        '''
        
        result = engine._parse_critique(response)
        
        assert result.has_issues is True
        assert len(result.issues) == 1
        assert result.issues[0].type == IssueType.ACCURACY
        assert result.should_retry is True
    
    def test_issue_stats_tracking(self):
        """Test that engine tracks issue statistics."""
        def mock_llm(prompt):
            if "Critically evaluate" in prompt:
                return '{"has_issues": true, "issues": [{"type": "accuracy", "description": "Issue", "severity": "high"}], "confidence_in_critique": 0.9, "should_retry": false}'
            return "Improved"
        
        engine = SelfReflectionEngine(llm_fn=mock_llm, max_iterations=1)
        
        engine.reflect("Answer 1", "Question 1")
        engine.reflect("Answer 2", "Question 2")
        
        stats = engine.get_issue_stats()
        
        assert "accuracy" in stats
        assert stats["accuracy"] >= 2
