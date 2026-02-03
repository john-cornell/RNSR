# RNSR - Recursive Neural-Symbolic Retriever

A state-of-the-art document retrieval system that preserves hierarchical structure for superior RAG performance. Combines PageIndex, Recursive Language Models (RLM), Knowledge Graphs, and Tree of Thoughts navigation.

## Overview

RNSR combines neural and symbolic approaches to achieve accurate document understanding:

- **Font Histogram Algorithm** - Automatically detects document hierarchy from font sizes (no training required)
- **Skeleton Index Pattern** - Lightweight summaries with KV store for efficient retrieval
- **Tree-of-Thoughts Navigation** - LLM reasons about document structure to find answers
- **RLM Unified Extraction** - LLM writes extraction code, grounded in actual text
- **Knowledge Graph** - Entity and relationship storage for cross-document linking
- **Self-Reflection Loop** - Iterative answer improvement through self-critique
- **Adaptive Learning** - System learns from your document workload over time

## Key Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Extraction** | Preserves document structure (sections, subsections, paragraphs) |
| **RLM Unified Extractor** | LLM writes extraction code + ToT validation (grounded, no hallucination) |
| **Provenance System** | Every answer traces back to exact document citations |
| **LLM Response Cache** | Semantic-aware caching for 10x cost/speed improvement |
| **Self-Reflection** | Iterative self-correction improves answer quality |
| **Reasoning Memory** | Learns successful query patterns for faster future queries |
| **Query Clarification** | Detects ambiguous queries and asks clarifying questions |
| **Table/Chart Parsing** | SQL-like queries over tables, chart trend analysis |
| **Adaptive Learning** | 6 registries that learn from usage and persist to disk |
| **Multi-Document Detection** | Automatically splits bundled PDFs |
| **Vision Mode** | OCR-free analysis for scanned documents and charts |

## Installation

```bash
# Clone the repository
git clone https://github.com/theeufj/RNSR.git
cd RNSR

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all LLM providers
pip install -e ".[all]"

# Or install with specific provider
pip install -e ".[openai]"      # OpenAI only
pip install -e ".[anthropic]"   # Anthropic only
pip install -e ".[gemini]"      # Google Gemini only
```

## Quick Start

### 1. Set up API keys

Create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your API keys
```

```env
# Choose your preferred LLM provider
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or
GOOGLE_API_KEY=AI...

# Optional: Override default models
LLM_PROVIDER=anthropic
SUMMARY_MODEL=claude-sonnet-4-5
```

### 2. Use the Python API

```python
from rnsr import RNSRClient

# Simple one-line Q&A
client = RNSRClient()
answer = client.ask("contract.pdf", "What are the payment terms?")
print(answer)

# Advanced navigation with verification and self-reflection
result = client.ask_advanced(
    "complex_report.pdf",
    "Compare liability clauses in sections 5 and 8",
    enable_verification=True,
    enable_self_reflection=True,
    max_recursion_depth=3,
)
```

### 3. Run the Demo UI

```bash
python demo.py
# Open http://localhost:7860 in your browser
```

## New Features

### Provenance System

Every answer includes traceable citations:

```python
from rnsr.agent import ProvenanceTracker, format_citations_for_display

tracker = ProvenanceTracker(kv_store=kv_store, skeleton=skeleton)
record = tracker.create_provenance_record(
    answer="The payment terms are net 30.",
    question="What are the payment terms?",
    variables=navigation_variables,
)

print(f"Confidence: {record.aggregate_confidence:.0%}")
print(format_citations_for_display(record.citations))
# Output:
# **Sources:**
# 1. [contract.pdf] Section: Payment Terms, Page 5: "Payment shall be due within 30 days..."
```

### LLM Response Caching

Automatic caching reduces costs and latency:

```python
from rnsr.agent import wrap_llm_with_cache, get_global_cache

# Wrap any LLM function with caching
cached_llm = wrap_llm_with_cache(llm.complete, ttl_seconds=3600)

# Use cached LLM - repeated prompts hit cache
response = cached_llm("What is 2+2?")  # Calls LLM
response = cached_llm("What is 2+2?")  # Returns cached (instant)

# Check cache stats
print(get_global_cache().get_stats())
# {'entries': 150, 'hits': 89, 'hit_rate': 0.59}
```

### Self-Reflection Loop

Answers are automatically critiqued and improved:

```python
from rnsr.agent import SelfReflectionEngine, reflect_on_answer

# Quick one-liner
result = reflect_on_answer(
    answer="The contract expires in 2024.",
    question="When does the contract expire?",
    evidence="Contract dated 2023, 2-year term...",
)

print(f"Improved: {result.improved}")
print(f"Final answer: {result.final_answer}")
print(f"Iterations: {result.total_iterations}")
```

### Reasoning Chain Memory

The system learns from successful queries:

```python
from rnsr.agent import get_reasoning_memory, find_similar_chains

# Find similar past queries
matches = find_similar_chains("What is the liability cap?")
for match in matches:
    print(f"Similar query: {match.chain.query}")
    print(f"Similarity: {match.similarity:.0%}")
    print(f"Past answer: {match.chain.answer}")
```

### Table Parsing

Extract and query tables from documents:

```python
from rnsr.ingestion import TableParser, TableQueryEngine

parser = TableParser()
tables = parser.parse_from_text(document_text)

# SQL-like queries
engine = TableQueryEngine(tables[0])
results = engine.select(
    columns=["Name", "Amount"],
    where={"Status": "Active"},
    order_by="Amount",
)

# Aggregations
total = engine.aggregate("Amount", "sum")
```

### Query Clarification

Handle ambiguous queries gracefully:

```python
from rnsr.agent import QueryClarifier, needs_clarification

# Check if query needs clarification
is_ambiguous, analysis = needs_clarification(
    "What does it say about the clause?"
)

if is_ambiguous:
    print(f"Ambiguity: {analysis.ambiguity_type}")
    print(f"Clarifying question: {analysis.suggested_clarification}")
    # "What does 'it' refer to in your question?"
```

## Adaptive Learning

RNSR learns from your document workload. All learned data persists in `~/.rnsr/`:

```
~/.rnsr/
├── learned_entity_types.json       # New entity types discovered
├── learned_relationship_types.json # New relationship types
├── learned_normalization.json      # Title/suffix patterns
├── learned_stop_words.json         # Domain-specific stop words
├── learned_header_thresholds.json  # Document-type font thresholds
├── learned_query_patterns.json     # Successful query patterns
├── reasoning_chains.json           # Successful reasoning chains
└── llm_cache.db                    # LLM response cache
```

The more you use RNSR, the better it gets at understanding your domain.

## How It Works

### Document Ingestion Pipeline

```
PDF → Font Analysis → Header Classification → Tree Building → Skeleton Index
         ↓                    ↓                    ↓              ↓
   Detect font sizes   Classify H1/H2/H3    Build hierarchy   Create summaries
                                                  ↓
                                        Multi-doc detection
                                        (page number resets)
```

### Query Processing

```
Question → Clarify → Pre-Filter → Tree Navigation → Answer → Self-Reflect → Verify
              ↓           ↓              ↓             ↓           ↓           ↓
        Ask if ambig  Keyword scan  ToT reasoning  Synthesize  Critique   Fact-check
                                         ↓                        ↓
                                  Sub-LLM recursion        Improve answer
                                  (complex queries)        (if issues)
```

### Entity Extraction (RLM Unified)

```
Document → LLM writes code → Execute on DOC_VAR → ToT validation → Cross-validate
              ↓                     ↓                   ↓               ↓
     Generates regex/Python   Grounded results   Probability scores  Entity↔Relationship
                                    ↓
                            All tied to exact text spans
```

## Architecture

```
rnsr/
├── agent/                   # Query processing
│   ├── rlm_navigator.py     # Main navigation agent
│   ├── provenance.py        # Citation tracking (NEW)
│   ├── llm_cache.py         # Response caching (NEW)
│   ├── self_reflection.py   # Answer improvement (NEW)
│   ├── reasoning_memory.py  # Chain memory (NEW)
│   ├── query_clarifier.py   # Ambiguity handling (NEW)
│   ├── graph.py             # LangGraph workflow
│   └── variable_store.py    # Context management
├── extraction/              # Entity/relationship extraction
│   ├── rlm_unified_extractor.py  # Best extractor (NEW)
│   ├── learned_types.py     # Adaptive type learning
│   ├── entity_linker.py     # Cross-document linking
│   └── models.py            # Entity/Relationship models
├── indexing/                # Index construction
│   ├── skeleton_index.py    # Summary generation
│   ├── knowledge_graph.py   # Entity/relationship storage
│   ├── kv_store.py          # SQLite/in-memory storage
│   └── semantic_search.py   # Optional vector search
├── ingestion/               # Document processing
│   ├── pipeline.py          # Main ingestion orchestrator
│   ├── font_histogram.py    # Font-based structure detection
│   ├── header_classifier.py # H1/H2/H3 classification
│   ├── table_parser.py      # Table extraction (NEW)
│   ├── chart_parser.py      # Chart interpretation (NEW)
│   └── tree_builder.py      # Hierarchical tree construction
├── llm.py                   # Multi-provider LLM abstraction
├── client.py                # High-level API
└── models.py                # Data structures
```

## API Reference

### High-Level API

```python
from rnsr import RNSRClient

client = RNSRClient(
    llm_provider="anthropic",  # or "openai", "gemini"
    llm_model="claude-sonnet-4-5"
)

# Simple query
answer = client.ask("document.pdf", "What is the main topic?")

# Vision mode (for scanned docs)
answer = client.ask_vision("scanned.pdf", "What does the chart show?")
```

### Low-Level API

```python
from rnsr import (
    ingest_document,
    build_skeleton_index,
    run_rlm_navigator,
    SQLiteKVStore
)
from rnsr.extraction import RLMUnifiedExtractor
from rnsr.agent import ProvenanceTracker, SelfReflectionEngine

# Step 1: Ingest document
result = ingest_document("document.pdf")
print(f"Extracted {result.tree.total_nodes} nodes")

# Step 2: Build index
kv_store = SQLiteKVStore("./data/index.db")
skeleton = build_skeleton_index(result.tree, kv_store)

# Step 3: Extract entities (grounded, no hallucination)
extractor = RLMUnifiedExtractor()
extraction = extractor.extract(
    node_id="section_1",
    doc_id="document",
    header="Introduction",
    content="..."
)

# Step 4: Query with provenance
answer = run_rlm_navigator(
    question="What are the key findings?",
    skeleton=skeleton,
    kv_store=kv_store
)

# Step 5: Get citations
tracker = ProvenanceTracker(kv_store=kv_store)
record = tracker.create_provenance_record(answer, question, variables)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Primary LLM provider | `auto` (detect from keys) |
| `SUMMARY_MODEL` | Model for summarization | Provider default |
| `AGENT_MODEL` | Model for navigation | Provider default |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `KV_STORE_PATH` | SQLite database path | `./data/kv_store.db` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `RNSR_LLM_CACHE_PATH` | Custom cache location | `~/.rnsr/llm_cache.db` |
| `RNSR_REASONING_MEMORY_PATH` | Custom memory location | `~/.rnsr/reasoning_chains.json` |

### Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | `gpt-5.2`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4o-mini` |
| **Anthropic** | `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5` |
| **Gemini** | `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` |

## Benchmarks

RNSR is designed for complex document understanding tasks:

- **Multi-document PDFs** - Automatically detects and separates bundled documents
- **Hierarchical queries** - "Compare section 3.2 with section 5.1"
- **Cross-reference questions** - "What does the appendix say about the claim in section 2?"
- **Entity extraction** - Grounded extraction with ToT validation (no hallucination)
- **Table queries** - "What is the total for Q4 2024?"

## Sample Documents

RNSR includes sample documents for testing and demonstration:

### Synthetic Documents (`samples/`)

| File | Type | Features Demonstrated |
|------|------|----------------------|
| `sample_contract.md` | Legal Contract | Entities (people, orgs), relationships, payment tables, legal terms |
| `sample_financial_report.md` | Financial Report | Financial tables, metrics, executive names, quarterly data |
| `sample_research_paper.md` | Academic Paper | Citations, hierarchical sections, technical content, tables |

### Real Test Documents (`rnsr/test-documents/`)

Legal documents from the Djokovic visa case (public court records) for testing with actual PDFs:
- Affidavits and court applications
- Legal submissions and orders
- Interview transcripts

### Using Sample Documents

```python
from pathlib import Path
from rnsr.ingestion import TableParser
from rnsr.extraction import CandidateExtractor

# Parse a sample document
sample = Path("samples/sample_contract.md").read_text()

# Extract tables
parser = TableParser()
tables = parser.parse_from_text(sample)
print(f"Found {len(tables)} tables")

# Extract entities
extractor = CandidateExtractor()
candidates = extractor.extract_candidates(sample)
print(f"Found {len(candidates)} entity candidates")
```

## Testing

### Test Suite Overview

RNSR has comprehensive test coverage with **281+ tests**:

```bash
# Run all tests
pytest tests/ -v

# Run specific feature tests
pytest tests/test_provenance.py tests/test_llm_cache.py -v

# Run end-to-end workflow tests
pytest tests/test_e2e_workflow.py -v

# Run with coverage
pytest tests/ --cov=rnsr --cov-report=html
```

### Test Categories

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_e2e_workflow.py` | 18 | Full pipeline: ingestion → extraction → KG → query → provenance |
| `test_provenance.py` | 17 | Citations, contradictions, provenance records |
| `test_llm_cache.py` | 17 | Cache get/set, TTL, persistence |
| `test_self_reflection.py` | 13 | Critique, refinement, iteration limits |
| `test_reasoning_memory.py` | 15 | Chain storage, similarity matching |
| `test_query_clarifier.py` | 19 | Ambiguity detection, clarification |
| `test_table_parser.py` | 26 | Markdown/ASCII tables, SQL-like queries |
| `test_chart_parser.py` | 16 | Chart detection, trend analysis |
| `test_rlm_unified.py` | 13 | REPL execution, code cleaning |
| `test_learned_types.py` | 13 | Adaptive learning registries |

### End-to-End Workflow Tests

The `test_e2e_workflow.py` demonstrates the complete pipeline:

```python
# Tests cover:
# 1. Document Ingestion - Parse structure and tables
# 2. Entity Extraction - Pattern-based grounded extraction  
# 3. Knowledge Graph - Store entities and relationships
# 4. Query Processing - Ambiguity detection, table queries
# 5. Provenance - Citations and evidence tracking
# 6. Self-Reflection - Answer improvement loop
# 7. Reasoning Memory - Learn from successful queries
# 8. LLM Cache - Response caching
# 9. Adaptive Learning - Type discovery
# 10. Full Workflow - Contract and financial analysis
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check .

# Type checking
mypy rnsr/
```

## Requirements

- Python 3.10+
- At least one LLM API key (OpenAI, Anthropic, or Gemini)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Research

RNSR is inspired by:
- [Hybrid Document Retrieval System Design](Research/Hybrid%20Document%20Retrieval%20System%20Design.pdf) - Core architecture and design principles
- [PageIndex (VectifyAI)](https://github.com/VectifyAI/PageIndex) - Vectorless reasoning-based tree search
- [Recursive Language Models](https://arxiv.org/html/2512.24601v1) - REPL environment with recursive sub-LLM calls
- Tree of Thoughts - LLM-based decision making with probabilities
- Self-Refine / Reflexion - Iterative self-correction patterns
