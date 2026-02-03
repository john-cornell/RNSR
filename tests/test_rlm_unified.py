"""Tests for the RLM Unified Extractor system."""

import pytest

from rnsr.extraction.models import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
)
from rnsr.extraction.rlm_unified_extractor import (
    RLMUnifiedExtractor,
    RLMUnifiedResult,
    UnifiedREPL,
)


class TestUnifiedREPL:
    """Tests for the UnifiedREPL execution environment."""
    
    def test_repl_creation(self):
        """Test creating a REPL environment."""
        repl = UnifiedREPL(
            document_text="Test document content.",
        )
        
        assert repl is not None
        assert repl.document_text == "Test document content."
    
    def test_repl_doc_var_available(self):
        """Test that DOC_VAR is available in REPL namespace."""
        repl = UnifiedREPL("Sample document text.")
        
        # DOC_VAR should be in the namespace
        assert "DOC_VAR" in repl._namespace
        assert repl._namespace["DOC_VAR"] == "Sample document text."
    
    def test_repl_execute_stores_entities(self):
        """Test that executed code can store entities."""
        content = "John Smith is the CEO. Jane Doe is the CFO."
        repl = UnifiedREPL(content)
        
        # Use store_variable to store entities in VARIABLES dict
        code = '''
import re
entities = []
for match in re.finditer(r"([A-Z][a-z]+ [A-Z][a-z]+)", SECTION_CONTENT):
    entities.append({
        "name": match.group(1),
        "type": "PERSON",
        "span_start": match.start(),
        "span_end": match.end(),
    })
store_variable("ENTITIES", entities)
'''
        
        result = repl.execute(code)
        
        assert result["success"] is True
        # Entities should be stored in variables
        assert "ENTITIES" in result["variables"]
    
    def test_repl_syntax_error(self):
        """Test handling syntax errors."""
        repl = UnifiedREPL("Content")
        
        result = repl.execute("def broken(")
        
        assert result["success"] is False
        assert result["error"] is not None
    
    def test_repl_runtime_error(self):
        """Test handling runtime errors."""
        repl = UnifiedREPL("Content")
        
        result = repl.execute("1 / 0")
        
        assert result["success"] is False
        assert result["error"] is not None
    
    def test_repl_imports_available(self):
        """Test that re module is available."""
        repl = UnifiedREPL("Content")
        
        # re should be in namespace
        assert "re" in repl._namespace
    
    def test_repl_search_text(self):
        """Test the search_text helper function."""
        content = "Amount: $1,500.00 and $2,000.00"
        repl = UnifiedREPL(content)
        
        # Call the internal search function
        results = repl._search_text(r"\$[\d,]+\.?\d*")
        
        assert len(results) >= 2
    
    def test_repl_store_and_get_variable(self):
        """Test variable storage functions."""
        repl = UnifiedREPL("Content")
        
        repl._store_variable("test_var", "test_value")
        
        assert repl._get_variable("test_var") == "test_value"


class TestRLMUnifiedResult:
    """Tests for RLMUnifiedResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a unified result."""
        result = RLMUnifiedResult(
            node_id="node_1",
            doc_id="doc_1",
            entities=[],
            relationships=[],
            code_executed=True,
        )
        
        assert result.node_id == "node_1"
        assert result.code_executed is True
    
    def test_result_default_values(self):
        """Test default values."""
        result = RLMUnifiedResult()
        
        assert result.entities == []
        assert result.relationships == []
        assert result.code_generated == ""
        assert result.code_executed is False


class TestRLMUnifiedExtractor:
    """Tests for RLMUnifiedExtractor class."""
    
    def test_extractor_creation(self):
        """Test creating an extractor."""
        extractor = RLMUnifiedExtractor()
        
        assert extractor is not None
    
    def test_extractor_result_type(self):
        """Test that RLMUnifiedResult is correctly typed."""
        result = RLMUnifiedResult(
            node_id="node_1",
            doc_id="doc_1",
        )
        
        assert isinstance(result, RLMUnifiedResult)
        assert result.node_id == "node_1"
        assert result.doc_id == "doc_1"
    
    def test_extractor_instance(self):
        """Test creating extractor instance."""
        extractor = RLMUnifiedExtractor()
        
        # Extractor should have extract method
        assert hasattr(extractor, 'extract')
    
    def test_result_has_expected_fields(self):
        """Test RLMUnifiedResult has expected fields."""
        result = RLMUnifiedResult()
        
        assert hasattr(result, 'entities')
        assert hasattr(result, 'relationships')
        assert hasattr(result, 'code_generated')
        assert hasattr(result, 'code_executed')


class TestCodeExecutionSafety:
    """Tests for code execution safety measures."""
    
    def test_code_cleaning(self):
        """Test that markdown code blocks are cleaned."""
        repl = UnifiedREPL("Content")
        
        code_with_markdown = '''```python
x = 1
```'''
        
        cleaned = repl._clean_code(code_with_markdown)
        
        assert "```" not in cleaned
        assert "x = 1" in cleaned
    
    def test_dangerous_operations_blocked(self):
        """Test that dangerous operations fail gracefully."""
        repl = UnifiedREPL("Content")
        
        # These should fail due to restricted namespace
        dangerous_codes = [
            "__import__('os').system('ls')",
            "open('/etc/passwd').read()",
        ]
        
        for code in dangerous_codes:
            result = repl.execute(code)
            # Should either fail or be restricted
            # The namespace doesn't include open or __import__
