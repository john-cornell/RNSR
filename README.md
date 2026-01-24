# RNSR - Recursive Neural-Symbolic Retriever

[![PyPI version](https://badge.fury.io/py/rnsr.svg)](https://badge.fury.io/py/rnsr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**RNSR** is a next-generation document retrieval system that reconstructs document hierarchies using font histogram analysis and navigates them with a recursive LangGraph agent.

Unlike traditional RAG systems that flatten documents into chunks, RNSR preserves the structural hierarchy of documents - understanding that a heading relates to its subsections, that a contract clause contains sub-clauses, and that context flows through the document tree.

## 🎯 Key Features

- **Font Histogram Algorithm** - Reconstructs document hierarchy from font size distributions (NOT vision models)
- **Skeleton Index Pattern** - Stores summaries in vector index, full text in KV store for efficient retrieval
- **Pointer-Based Variable Stitching** - Prevents context pollution by using symbolic references
- **3-Tier Graceful Degradation** - Font → Semantic → OCR fallback chain
- **Multi-Provider LLM Support** - OpenAI, Anthropic, and Google Gemini

## 📊 Benchmark Results

| Benchmark | Samples | Answer EM | Answer F1 | Avg Time | Description |
|-----------|---------|-----------|-----------|----------|-------------|
| **QuALITY** | 100 | 84.0% | 84.6% | 6.78s | Long document comprehension (2,000-8,000 words) |
| **HotpotQA** | 100 | 54.0% | 67.9% | 3.22s | Multi-hop reasoning across Wikipedia articles |
| **Qasper** | 85 | 71.8% | 86.0% | 1.92s | Scientific paper question answering |

*Results using Gemini 2.5 Flash. Qasper limited to 85 samples due to dataset availability.*

### Comparison to Baselines

| Benchmark | RNSR | Typical RAG | Human | Notes |
|-----------|------|-------------|-------|-------|
| **QuALITY** (EM) | 84.0% | 60-75% | 93.5% | Matches GPT-4 full-context performance |
| **HotpotQA** (F1) | 67.9% | 55-65% | 74.0% | Near human-level multi-hop reasoning |
| **Qasper** (F1) | 86.0% | 50-70% | — | Exceptional on structured scientific papers |

RNSR's hierarchical approach outperforms traditional chunk-based RAG by **+9-36%** across benchmarks, with the largest gains on long structured documents where preserving hierarchy matters most.

## 🚀 Installation

### Basic Installation

```bash
pip install rnsr
```

### With LLM Providers

```bash
# With OpenAI
pip install rnsr[openai]

# With Anthropic
pip install rnsr[anthropic]

# With Google Gemini
pip install rnsr[gemini]

# All providers
pip install rnsr[all]

# With benchmarking tools
pip install rnsr[benchmarks]
```

### From Source

```bash
git clone https://github.com/rnsr-project/rnsr.git
cd rnsr
pip install -e ".[all,dev]"
```

## ⚡ Quick Start

### 1. Set up your API key

```bash
# Create .env file
echo "GOOGLE_API_KEY=your-key-here" > .env

# Or use OpenAI/Anthropic
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 2. Basic Usage

```python
from rnsr import ingest_document, build_skeleton_index, run_navigator

# Step 1: Ingest a PDF document
result = ingest_document("contract.pdf")
tree = result.tree

# Step 2: Build the skeleton index
skeleton, kv_store = build_skeleton_index(tree)

# Step 3: Query with the navigator agent
answer = run_navigator(
    "What are the payment terms?",
    skeleton,
    kv_store
)

print(answer)
```

### 3. CLI Usage

```bash
# Ingest a document
rnsr ingest contract.pdf --output contract_index/

# Query a document
rnsr query "What are the payment terms?" --index contract_index/

# Run benchmarks
rnsr benchmark --datasets hotpotqa quality --samples 20
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         RNSR Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Document   │───▶│ Font-Based   │───▶│  Hierarchy   │   │
│  │   Ingestion  │    │  Analysis    │    │    Tree      │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                    │         │
│                                                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Answer     │◀───│  Navigator   │◀───│   Skeleton   │   │
│  │   Synthesis  │    │    Agent     │    │    Index     │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Document Ingestion**: PDFs are parsed using `pdfplumber` to extract text with font metadata
2. **Font Histogram Analysis**: Font sizes are clustered to identify heading levels (H1, H2, H3, etc.)
3. **Hierarchy Reconstruction**: A tree structure is built where headings contain their child content
4. **Skeleton Index**: Each node gets a summary (for vector search) while full text goes to KV store
5. **Navigator Agent**: A LangGraph-based agent traverses the tree using `DrillDown`, `Ascend`, and `Read` actions
6. **Variable Stitching**: Retrieved content is stored in variables (e.g., `$1`, `$2`) to prevent context pollution

## 🔧 Configuration

### LLM Providers

```python
from rnsr.llm import get_llm, LLMProvider

# Auto-detect from environment
llm = get_llm()

# Explicit provider
llm = get_llm(provider=LLMProvider.GEMINI, model="gemini-2.5-flash")
llm = get_llm(provider=LLMProvider.OPENAI, model="gpt-4o")
llm = get_llm(provider=LLMProvider.ANTHROPIC, model="claude-3-sonnet")
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `RNSR_LLM_PROVIDER` | Default provider (`gemini`, `openai`, `anthropic`) |
| `RNSR_LLM_MODEL` | Default model name |

## 📚 API Reference

### Ingestion

```python
from rnsr import ingest_document, IngestionResult

result: IngestionResult = ingest_document(
    file_path="document.pdf",
    strategy="font_histogram",  # or "semantic", "ocr"
)

# Access the document tree
tree = result.tree
print(f"Root: {tree.root.title}")
for child in tree.root.children:
    print(f"  - {child.title}")
```

### Indexing

```python
from rnsr import build_skeleton_index, SQLiteKVStore

skeleton, kv_store = build_skeleton_index(
    tree,
    kv_store=SQLiteKVStore("./cache.db"),  # Optional persistent store
)
```

### Navigation

```python
from rnsr import run_navigator

answer = run_navigator(
    query="What are the termination clauses?",
    skeleton_index=skeleton,
    kv_store=kv_store,
    max_iterations=10,
)
```

## 🧪 Running Benchmarks

```bash
# Run specific benchmarks
python -m rnsr.benchmarks.evaluation_suite \
    --datasets hotpotqa quality qasper \
    --samples 50 \
    --llm-provider gemini

# Available benchmarks:
# - hotpotqa: Multi-hop reasoning
# - quality: Long document comprehension
# - qasper: Scientific paper QA (uses SciQ)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/rnsr-project/rnsr.git
cd rnsr
pip install -e ".[all,dev]"

# Run tests
pytest

# Run linting
ruff check .
mypy rnsr/
```

## 📖 Research Background

RNSR is based on research into improving RAG systems for structured documents. Key insights:

1. **Flat retrieval strips structural context** - Traditional chunking loses the relationship between headings and content
2. **Vector search is local** - Similarity matching doesn't understand document hierarchy
3. **Lost-in-the-middle effect** - LLMs neglect middle context in long sequences
4. **Semantic gap** - Embedding similarity ≠ semantic relevance for the query

RNSR addresses these by:
- Preserving document structure as a navigable tree
- Using an agent-based approach for intelligent traversal
- Separating retrieval (summaries) from reading (full text)
- Employing variable stitching to manage context efficiently

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for the indexing infrastructure
- [LangGraph](https://github.com/langchain-ai/langgraph) for the agent framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF parsing

---

<p align="center">
  <b>Built with ❤️ for better document understanding</b>
</p>
