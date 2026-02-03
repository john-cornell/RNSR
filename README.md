# RNSR - Recursive Neural-Symbolic Retriever

A state-of-the-art document retrieval system that preserves hierarchical structure for superior RAG performance.

## Overview

RNSR combines neural and symbolic approaches to achieve accurate document understanding:

- **Font Histogram Algorithm** - Automatically detects document hierarchy from font sizes (no training required)
- **Skeleton Index Pattern** - Lightweight summaries with KV store for efficient retrieval
- **Tree-of-Thoughts Navigation** - LLM reasons about document structure to find answers
- **Multi-Document Detection** - Automatically splits bundled PDFs into constituent documents
- **Multi-Provider LLM Support** - Works with OpenAI, Anthropic, and Google Gemini

## Key Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Extraction** | Preserves document structure (sections, subsections, paragraphs) |
| **Page Number Reset Detection** | Identifies document boundaries in multi-document PDFs |
| **LLM Boundary Validation** | Uses AI to verify detected document splits |
| **Recursive Sub-LLM Calls** | Breaks complex questions into sub-queries |
| **Answer Verification** | Validates answers against source context |
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

# Advanced navigation with verification
result = client.ask_advanced(
    "complex_report.pdf",
    "Compare liability clauses in sections 5 and 8",
    enable_verification=True,
    max_recursion_depth=3,
)
```

### 3. Run the Demo UI

```bash
python demo.py
# Open http://localhost:7860 in your browser
```

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
Question → Pre-Filter → Tree Navigation → Context Retrieval → Answer Generation
              ↓              ↓                   ↓                   ↓
         Keyword scan   ToT reasoning     Fetch from KV store   LLM synthesis
                             ↓
                     Sub-LLM recursion
                     (for complex queries)
```

## Architecture

```
rnsr/
├── ingestion/           # Document processing
│   ├── pipeline.py      # Main ingestion orchestrator
│   ├── font_histogram.py    # Font-based structure detection
│   ├── header_classifier.py # H1/H2/H3 classification
│   ├── tree_builder.py      # Hierarchical tree construction
│   └── document_boundary.py # Multi-document detection
├── indexing/            # Index construction
│   ├── skeleton_index.py    # Summary generation
│   ├── kv_store.py          # SQLite/in-memory storage
│   └── semantic_search.py   # Optional vector search
├── agent/               # Query processing
│   ├── rlm_navigator.py     # Main navigation agent
│   ├── graph.py             # LangGraph workflow
│   └── variable_store.py    # Context management
├── llm.py               # Multi-provider LLM abstraction
├── client.py            # High-level API
└── models.py            # Data structures
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

# Step 1: Ingest document
result = ingest_document("document.pdf")
print(f"Extracted {result.tree.total_nodes} nodes")

# Step 2: Build index
kv_store = SQLiteKVStore("./data/index.db")
skeleton = build_skeleton_index(result.tree, kv_store)

# Step 3: Query
answer = run_rlm_navigator(
    question="What are the key findings?",
    skeleton=skeleton,
    kv_store=kv_store
)
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

### Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| **Anthropic** | `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4-5` |
| **Gemini** | `gemini-2.5-flash`, `gemini-2.5-pro` |

## Benchmarks

RNSR is designed for complex document understanding tasks:

- **Multi-document PDFs** - Automatically detects and separates bundled documents
- **Hierarchical queries** - "Compare section 3.2 with section 5.1"
- **Cross-reference questions** - "What does the appendix say about the claim in section 2?"

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

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
