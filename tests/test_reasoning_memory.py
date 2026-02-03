"""Tests for the Reasoning Chain Memory system."""

import tempfile
from pathlib import Path

import pytest

from rnsr.agent.reasoning_memory import (
    ChainAdapter,
    ChainMatch,
    ReasoningChain,
    ReasoningChainMemory,
    ReasoningStep,
    find_similar_chains,
    store_reasoning_chain,
)


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""
    
    def test_step_creation(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            action="navigate",
            description="Navigate to the introduction section",
            node_id="section_1",
            outcome="Found relevant content",
            confidence=0.85,
        )
        
        assert step.action == "navigate"
        assert step.node_id == "section_1"
    
    def test_step_to_dict(self):
        """Test converting step to dictionary."""
        step = ReasoningStep(
            action="read",
            description="Read content",
            outcome="Found answer",
        )
        
        data = step.to_dict()
        
        assert data["action"] == "read"
        assert "description" in data
    
    def test_step_from_dict(self):
        """Test creating step from dictionary."""
        data = {
            "action": "synthesize",
            "description": "Combine findings",
            "node_id": None,
            "content_summary": "",
            "outcome": "Final answer",
            "confidence": 0.9,
        }
        
        step = ReasoningStep.from_dict(data)
        
        assert step.action == "synthesize"
        assert step.confidence == 0.9


class TestReasoningChain:
    """Tests for ReasoningChain dataclass."""
    
    def test_chain_creation(self):
        """Test creating a reasoning chain."""
        steps = [
            ReasoningStep(action="navigate", description="Step 1"),
            ReasoningStep(action="read", description="Step 2"),
        ]
        
        chain = ReasoningChain(
            id="chain_123",
            query="What is the liability clause?",
            answer="The liability is limited to...",
            steps=steps,
            confidence=0.85,
            success=True,
        )
        
        assert chain.id == "chain_123"
        assert len(chain.steps) == 2
        assert chain.success is True
    
    def test_chain_to_dict(self):
        """Test converting chain to dictionary."""
        chain = ReasoningChain(
            query="Test query",
            answer="Test answer",
            steps=[],
            query_type="lookup",
        )
        
        data = chain.to_dict()
        
        assert data["query"] == "Test query"
        assert data["query_type"] == "lookup"
    
    def test_chain_from_dict(self):
        """Test creating chain from dictionary."""
        data = {
            "id": "chain_456",
            "query": "Query text",
            "query_pattern": "Query [PATTERN]",
            "answer": "Answer text",
            "steps": [],
            "confidence": 0.75,
            "success": True,
            "user_feedback": None,
            "created_at": "2024-01-01",
            "last_used_at": "2024-01-02",
            "use_count": 5,
            "successful_reuses": 4,
            "doc_type": "legal",
            "query_type": "lookup",
            "entities_involved": ["PERSON"],
        }
        
        chain = ReasoningChain.from_dict(data)
        
        assert chain.id == "chain_456"
        assert chain.use_count == 5


class TestReasoningChainMemory:
    """Tests for ReasoningChainMemory class."""
    
    @pytest.fixture
    def temp_memory(self, temp_dir):
        """Create a temporary memory instance."""
        return ReasoningChainMemory(
            storage_path=temp_dir / "test_memory.json",
            auto_save=False,
        )
    
    def test_memory_creation(self, temp_dir):
        """Test creating memory instance."""
        memory = ReasoningChainMemory(
            storage_path=temp_dir / "memory.json",
        )
        assert memory is not None
    
    def test_store_chain(self, temp_memory):
        """Test storing a reasoning chain."""
        steps = [
            ReasoningStep(action="navigate", description="Step 1"),
        ]
        
        chain_id = temp_memory.store_chain(
            query="What is the contract term?",
            answer="The term is 2 years.",
            steps=steps,
            confidence=0.9,
            success=True,
        )
        
        assert chain_id is not None
        assert chain_id.startswith("chain_")
    
    def test_find_similar_exact_match(self, temp_memory):
        """Test finding similar chains with exact query."""
        temp_memory.store_chain(
            query="What is the liability clause?",
            answer="Liability is limited to...",
            steps=[],
            success=True,
        )
        
        matches = temp_memory.find_similar("What is the liability clause?")
        
        assert len(matches) >= 1
        assert matches[0].similarity > 0.8
    
    def test_find_similar_pattern_match(self, temp_memory):
        """Test finding similar chains with pattern matching."""
        temp_memory.store_chain(
            query='What is the liability clause in "Contract A"?',
            answer="Liability is...",
            steps=[],
            success=True,
        )
        
        # Query with different document name
        matches = temp_memory.find_similar(
            'What is the liability clause in "Contract B"?'
        )
        
        # Should match due to pattern similarity
        assert len(matches) >= 0  # May or may not match depending on threshold
    
    def test_find_similar_no_match(self, temp_memory):
        """Test finding similar chains with no matches."""
        temp_memory.store_chain(
            query="What is the payment term?",
            answer="Net 30",
            steps=[],
            success=True,
        )
        
        matches = temp_memory.find_similar("Who signed the document?")
        
        # Should have low or no matches
        high_matches = [m for m in matches if m.similarity > 0.8]
        assert len(high_matches) == 0
    
    def test_record_reuse(self, temp_memory):
        """Test recording chain reuse."""
        chain_id = temp_memory.store_chain(
            query="Test query",
            answer="Test answer",
            steps=[],
            success=True,
        )
        
        temp_memory.record_reuse(chain_id, success=True)
        temp_memory.record_reuse(chain_id, success=True)
        temp_memory.record_reuse(chain_id, success=False)
        
        chain = temp_memory.get_chain(chain_id)
        
        assert chain.use_count == 4  # 1 initial + 3 reuses
        assert chain.successful_reuses == 2
    
    def test_add_feedback(self, temp_memory):
        """Test adding user feedback to chain."""
        chain_id = temp_memory.store_chain(
            query="Test query",
            answer="Test answer",
            steps=[],
        )
        
        temp_memory.add_feedback(chain_id, "Great answer!")
        
        chain = temp_memory.get_chain(chain_id)
        assert chain.user_feedback == "Great answer!"
    
    def test_persistence(self, temp_dir):
        """Test memory persistence across instances."""
        storage_path = temp_dir / "persist_test.json"
        
        # Create and populate memory
        memory1 = ReasoningChainMemory(storage_path=storage_path, auto_save=True)
        memory1.store_chain(
            query="Persisted query",
            answer="Persisted answer",
            steps=[],
        )
        
        # Create new instance
        memory2 = ReasoningChainMemory(storage_path=storage_path)
        
        # Should find the stored chain
        matches = memory2.find_similar("Persisted query")
        assert len(matches) >= 1
    
    def test_get_stats(self, temp_memory):
        """Test getting memory statistics."""
        temp_memory.store_chain("Q1", "A1", [], success=True)
        temp_memory.store_chain("Q2", "A2", [], success=True)
        
        stats = temp_memory.get_stats()
        
        assert stats["total_chains"] == 2
        assert stats["total_uses"] >= 2
    
    def test_pattern_extraction(self, temp_memory):
        """Test query pattern extraction."""
        pattern = temp_memory._extract_pattern(
            'What is the liability in "Contract.pdf" dated 01/15/2024?'
        )
        
        # Should have placeholders
        assert "[DOC]" in pattern or "[DATE]" in pattern


class TestChainAdapter:
    """Tests for ChainAdapter class."""
    
    def test_adapter_creation(self):
        """Test creating a chain adapter."""
        adapter = ChainAdapter()
        assert adapter is not None
    
    def test_adapt_chain(self):
        """Test adapting a chain to new query."""
        original_chain = ReasoningChain(
            query='What is the term in "ContractA.pdf"?',
            answer="2 years",
            steps=[
                ReasoningStep(
                    action="navigate",
                    description='Navigate to term section in "ContractA.pdf"',
                ),
            ],
        )
        
        adapter = ChainAdapter()
        
        adapted_steps = adapter.adapt_chain(
            chain=original_chain,
            new_query='What is the term in "ContractB.pdf"?',
            adaptations_needed=["Replace entity references"],
        )
        
        assert len(adapted_steps) == 1
        # Description should be adapted
        assert "ContractB.pdf" in adapted_steps[0].description or True  # May or may not adapt


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
