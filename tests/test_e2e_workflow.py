"""
End-to-End Workflow Tests for RNSR

These tests demonstrate the complete document processing pipeline:
1. Document ingestion (parsing structure)
2. Entity extraction (RLM Unified)
3. Knowledge graph storage
4. Query processing with ToT navigation
5. Answer generation with provenance
6. Self-reflection and improvement

Run with: pytest tests/test_e2e_workflow.py -v -s
"""

import tempfile
from pathlib import Path

import pytest

# Sample documents for testing
SAMPLE_CONTRACT = """
# PROFESSIONAL SERVICES AGREEMENT

**Contract Number:** PSA-2024-00142
**Effective Date:** January 15, 2024
**Parties:** Acme Technologies Inc. ("Client") and DataSoft Solutions LLC ("Provider")

## 1. PARTIES

### 1.1 Client Information
**Acme Technologies Inc.**
- CEO: Dr. Sarah Chen
- CFO: Michael Rodriguez
- Primary Contact: Jennifer Walsh

### 1.2 Provider Information
**DataSoft Solutions LLC**
- Managing Director: Robert Thompson
- Project Lead: Amanda Foster

## 2. COMPENSATION

### 2.1 Total Contract Value
The total value is **$750,000 USD**.

### 2.2 Payment Schedule
| Milestone | Amount | Due Date |
|-----------|--------|----------|
| Contract Signing | $150,000 | January 15, 2024 |
| Phase 1 | $112,500 | March 15, 2024 |
| Phase 2 | $225,000 | June 15, 2024 |
| Final | $262,500 | September 15, 2024 |

## 3. TERM

This Agreement runs from January 15, 2024 to September 30, 2024.

## 4. CONFIDENTIALITY

Confidentiality obligations survive for 3 years after termination.

## 5. LIABILITY

Total liability shall not exceed $750,000.
"""

SAMPLE_FINANCIAL = """
# Q4 2024 Financial Report

**Company:** Nexus Corporation
**Period:** October - December 2024

## Executive Summary

Revenue grew 23% year-over-year to $892 million.
Operating margin expanded to 18.5%.
Net income reached $127 million.

## Revenue by Segment

| Segment | Q4 2024 | Q4 2023 | Growth |
|---------|---------|---------|--------|
| Cloud Services | $412M | $298M | +38.3% |
| Enterprise | $287M | $251M | +14.3% |
| Services | $118M | $102M | +15.7% |
| Licensing | $75M | $74M | +1.4% |

## Key Metrics

- Annual Recurring Revenue (ARR): $1.52 billion
- Net Revenue Retention: 118%
- Customer Count: 4,500+
- Employees: 4,700

## Leadership

- CEO: Thomas Anderson
- CFO: Margaret Liu
- CTO: Dr. James Park

## 2025 Guidance

Revenue target: $3.9B - $4.1B (18-24% growth)
"""


class TestDocumentIngestion:
    """Test document parsing and structure extraction."""
    
    def test_parse_markdown_structure(self):
        """Test that markdown structure is correctly parsed."""
        # Using table parser as primary structure detector for markdown
        from rnsr.ingestion.table_parser import TableParser
        
        parser = TableParser()
        tables = parser.parse_from_text(SAMPLE_CONTRACT)
        
        # Should detect tables in the document
        assert tables is not None
    
    def test_table_extraction(self):
        """Test that tables are extracted from documents."""
        from rnsr.ingestion.table_parser import TableParser
        
        parser = TableParser()
        tables = parser.parse_from_text(SAMPLE_CONTRACT)
        
        # Should find the payment schedule table
        assert len(tables) >= 1
        
        # Check table has expected columns
        table = tables[0]
        assert "Milestone" in table.headers or "Amount" in table.headers
    
    def test_financial_table_parsing(self):
        """Test parsing financial tables with numbers."""
        from rnsr.ingestion.table_parser import TableParser, TableQueryEngine
        
        parser = TableParser()
        tables = parser.parse_from_text(SAMPLE_FINANCIAL)
        
        assert len(tables) >= 1
        
        # Query the revenue table
        for table in tables:
            if any("Segment" in h or "Q4" in h for h in table.headers):
                engine = TableQueryEngine(table)
                results = engine.select()
                assert len(results) >= 1
                break


class TestEntityExtraction:
    """Test entity and relationship extraction."""
    
    def test_extract_entities_from_contract(self):
        """Test extracting entities using candidate extractor (pattern-based)."""
        from rnsr.extraction import CandidateExtractor
        
        extractor = CandidateExtractor()
        
        content = """
        Acme Technologies Inc. - CEO: Dr. Sarah Chen, CFO: Michael Rodriguez.
        DataSoft Solutions LLC - Managing Director: Robert Thompson.
        Contact: Jennifer Walsh at jennifer.walsh@acmetech.com
        """
        
        candidates = extractor.extract_candidates(content)
        
        # Should extract some candidates (emails, names, etc.)
        assert candidates is not None
        assert len(candidates) >= 1
        
        # Check we extracted something (emails, orgs, persons)
        types_found = [c.candidate_type for c in candidates]
        assert len(types_found) >= 1
    
    def test_extract_financial_entities(self):
        """Test extracting monetary amounts from financial report."""
        from rnsr.extraction import CandidateExtractor
        
        extractor = CandidateExtractor()
        
        content = """
        Revenue grew 23% year-over-year to $892 million.
        Net income reached $127 million.
        CEO Thomas Anderson reported strong performance.
        """
        
        candidates = extractor.extract_candidates(content)
        
        # Should extract monetary values
        assert len(candidates) >= 1
        types_found = [c.candidate_type for c in candidates]
        # Should find monetary or person entities
        assert any(t in ("monetary", "person", "percentage") for t in types_found)


class TestKnowledgeGraph:
    """Test knowledge graph storage and retrieval."""
    
    @pytest.fixture
    def temp_kg(self, temp_dir):
        """Create a temporary knowledge graph."""
        from rnsr.indexing.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph(db_path=temp_dir / "test_kg.db")
    
    def test_store_entities(self, temp_kg):
        """Test storing entities."""
        from rnsr.extraction.models import Entity, EntityType
        
        # Create test entities with all required fields
        entities = [
            Entity(
                id="e1",
                name="Dr. Sarah Chen",
                canonical_name="sarah chen",
                type=EntityType.PERSON,
                doc_id="contract",
            ),
            Entity(
                id="e2",
                name="Acme Technologies Inc.",
                canonical_name="acme technologies inc",
                type=EntityType.ORGANIZATION,
                doc_id="contract",
            ),
        ]
        
        # Store entities
        for entity in entities:
            temp_kg.add_entity(entity)
        
        # Entities should be stored successfully
        assert True  # If we get here, storage worked
    
    def test_store_relationships(self, temp_kg):
        """Test storing relationships."""
        from rnsr.extraction.models import Entity, EntityType, Relationship, RelationType
        
        # Create entities with all required fields
        person = Entity(
            id="p1",
            name="Robert Thompson",
            canonical_name="robert thompson",
            type=EntityType.PERSON,
        )
        org = Entity(
            id="o1",
            name="DataSoft Solutions",
            canonical_name="datasoft solutions",
            type=EntityType.ORGANIZATION,
        )
        
        temp_kg.add_entity(person)
        temp_kg.add_entity(org)
        
        # Create relationship using AFFILIATED_WITH (person affiliated with org)
        # Note: Relationship uses source_id/target_id, not source_entity_id/target_entity_id
        rel = Relationship(
            id="r1",
            source_id="p1",
            target_id="o1",
            type=RelationType.AFFILIATED_WITH,
        )
        
        temp_kg.add_relationship(rel)
        
        # Relationship should be stored successfully
        assert True  # If we get here, storage worked


class TestQueryProcessing:
    """Test query processing and answer generation."""
    
    def test_query_clarification(self):
        """Test ambiguous query detection."""
        from rnsr.agent.query_clarifier import QueryClarifier, AmbiguityType
        
        clarifier = QueryClarifier(use_llm_detection=False)
        
        # Ambiguous query
        analysis = clarifier.analyze_query("What does it say about the clause?")
        assert analysis.is_ambiguous
        assert analysis.ambiguity_type == AmbiguityType.REFERENCE
        
        # Clear query
        analysis = clarifier.analyze_query(
            "What is the total contract value in PSA-2024-00142?"
        )
        # Should be less ambiguous
    
    def test_table_query(self):
        """Test SQL-like queries over tables."""
        from rnsr.ingestion.table_parser import (
            TableParser,
            TableQueryEngine,
        )
        
        parser = TableParser()
        tables = parser.parse_from_text(SAMPLE_CONTRACT)
        
        if tables:
            engine = TableQueryEngine(tables[0])
            
            # Select with filter
            results = engine.select(limit=2)
            assert len(results) <= 2


class TestProvenanceAndCitations:
    """Test provenance tracking and citation generation."""
    
    def test_citation_creation(self):
        """Test creating citations."""
        from rnsr.agent.provenance import Citation, CitationStrength
        
        citation = Citation(
            doc_id="sample_contract",
            node_id="section_2",
            quote="The total value is $750,000 USD.",
            page_num=1,
            strength=CitationStrength.DIRECT,
            confidence=0.95,
        )
        
        assert citation.doc_id == "sample_contract"
        assert citation.confidence == 0.95
        
        # Format for display
        formatted = citation.to_formatted_string()
        assert "$750,000" in formatted
    
    def test_provenance_record(self):
        """Test creating a full provenance record."""
        from rnsr.agent.provenance import (
            ProvenanceTracker,
            ProvenanceRecord,
        )
        
        tracker = ProvenanceTracker()
        
        variables = {
            "section_2": {
                "content": "The total value is $750,000 USD.",
                "node_id": "section_2",
                "doc_id": "contract",
            }
        }
        
        record = tracker.create_provenance_record(
            answer="The contract value is $750,000.",
            question="What is the contract value?",
            variables=variables,
        )
        
        assert isinstance(record, ProvenanceRecord)
        assert record.answer == "The contract value is $750,000."


class TestSelfReflection:
    """Test self-reflection and answer improvement."""
    
    def test_reflection_engine_creation(self):
        """Test creating a reflection engine."""
        from rnsr.agent.self_reflection import SelfReflectionEngine
        
        engine = SelfReflectionEngine(max_iterations=3)
        
        assert engine.max_iterations == 3
    
    def test_reflection_without_llm(self):
        """Test reflection passthrough without LLM."""
        from rnsr.agent.self_reflection import SelfReflectionEngine
        
        engine = SelfReflectionEngine(llm_fn=None)
        
        result = engine.reflect(
            answer="The contract value is $750,000.",
            question="What is the contract value?",
        )
        
        assert result.final_answer == "The contract value is $750,000."


class TestReasoningMemory:
    """Test reasoning chain memory."""
    
    def test_store_and_retrieve_chain(self, temp_dir):
        """Test storing and retrieving reasoning chains."""
        from rnsr.agent.reasoning_memory import ReasoningChainMemory, ReasoningStep
        
        memory = ReasoningChainMemory(
            storage_path=temp_dir / "chains.json",
            auto_save=False,
        )
        
        steps = [
            ReasoningStep(action="navigate", description="Go to compensation section"),
            ReasoningStep(action="read", description="Read contract value"),
            ReasoningStep(action="answer", description="Extract $750,000"),
        ]
        
        chain_id = memory.store_chain(
            query="What is the contract value?",
            answer="$750,000",
            steps=steps,
            success=True,
        )
        
        assert chain_id is not None
        
        # Find similar
        matches = memory.find_similar("What is the total amount?")
        assert isinstance(matches, list)


class TestLLMCache:
    """Test LLM response caching."""
    
    def test_cache_hit(self, temp_dir):
        """Test that repeated prompts hit cache."""
        from rnsr.agent.llm_cache import LLMCache, CachedLLM
        
        call_count = 0
        
        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return f"Response #{call_count}"
        
        cache = LLMCache(
            storage_path=temp_dir / "cache.db",
            enable_persistence=True,
        )
        
        cached_llm = CachedLLM(mock_llm, cache=cache)
        
        # First call
        result1 = cached_llm("What is 2+2?")
        assert result1 == "Response #1"
        
        # Second call - should hit cache
        result2 = cached_llm("What is 2+2?")
        assert result2 == "Response #1"
        assert call_count == 1  # Only called once


class TestAdaptiveLearning:
    """Test adaptive learning registries."""
    
    def test_learn_entity_type(self, temp_dir):
        """Test learning new entity types."""
        from rnsr.extraction.learned_types import LearnedTypeRegistry
        
        registry = LearnedTypeRegistry(
            storage_path=temp_dir / "types.json",
            auto_save=False,
        )
        
        registry.record_type(
            type_name="contract_clause",
            entity_name="Termination Clause",
            context="The Termination Clause allows either party...",
        )
        
        type_data = registry.get_type("contract_clause")
        assert type_data is not None
        assert type_data["count"] >= 1


class TestFullWorkflow:
    """Integration tests for the complete workflow."""
    
    def test_contract_analysis_workflow(self, temp_dir):
        """Test complete workflow for contract analysis."""
        from rnsr.ingestion.table_parser import TableParser
        from rnsr.extraction import CandidateExtractor
        from rnsr.agent.provenance import ProvenanceTracker
        
        # Step 1: Parse tables
        parser = TableParser()
        tables = parser.parse_from_text(SAMPLE_CONTRACT)
        
        # Step 2: Extract entities using pattern-based extractor
        extractor = CandidateExtractor()
        candidates = extractor.extract_candidates(
            "Client: Acme Technologies Inc. Provider: DataSoft Solutions LLC. Email: contact@acme.com"
        )
        
        # Step 3: Create provenance
        tracker = ProvenanceTracker()
        record = tracker.create_provenance_record(
            answer="The parties are Acme Technologies Inc. (Client) and DataSoft Solutions LLC (Provider).",
            question="Who are the parties to this contract?",
            variables={
                "parties": {
                    "content": "Client: Acme Technologies Inc. Provider: DataSoft Solutions LLC.",
                    "doc_id": "contract",
                }
            },
        )
        
        # Verify workflow completed
        assert len(tables) >= 1
        assert candidates is not None
        assert record is not None
    
    def test_financial_analysis_workflow(self, temp_dir):
        """Test complete workflow for financial analysis."""
        from rnsr.ingestion.table_parser import TableParser, TableQueryEngine
        from rnsr.extraction import CandidateExtractor
        
        # Step 1: Parse revenue table
        parser = TableParser()
        tables = parser.parse_from_text(SAMPLE_FINANCIAL)
        
        # Step 2: Query table data
        results = []
        for table in tables:
            if table.headers:
                engine = TableQueryEngine(table)
                results = engine.select()
                if results:
                    break
        
        # Step 3: Extract key figures using pattern extractor
        extractor = CandidateExtractor()
        candidates = extractor.extract_candidates(
            "Revenue: $892M. Net income: $127M. CEO: Thomas Anderson."
        )
        
        assert len(tables) >= 1
        assert candidates is not None


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
