# RNSR Benchmarking Suite

This module provides comprehensive benchmarking to validate RNSR's claims against standard RAG baselines and established academic benchmarks.

## Latest Benchmark Results (Gemini 2.5 Flash)

| Benchmark | Answer EM | Answer F1 | Type | Notes |
|-----------|-----------|-----------|------|-------|
| **QuALITY** | 90% | 88.5% | Long document | Best for RNSR - tests full document comprehension |
| **SciQ** | 71.4% | 91.7% | Scientific reasoning | QASPER alternative (scientific QA) |
| **HotpotQA** | 60% | 80.4% | Multi-hop | Matches human baseline (60% EM) |

*Results on 10 samples per benchmark using Gemini 2.5 Flash*

## Why Benchmarking Matters

As stated in the research paper, RNSR aims to solve fundamental limitations of standard RAG:

1. **Flat retrieval strips structural context** - chunks lose their hierarchical position
2. **Vector search is local** - no global awareness of document structure  
3. **Lost-in-the-middle effect** - LLMs neglect middle context in long sequences
4. **Semantic gap** - similarity matching ≠ semantic understanding

We validate these claims by comparing against:
- Naive chunking baselines (512, 1024 token chunks)
- Semantic chunking approaches
- Standard academic benchmarks

## Standard Benchmarks Supported

### Long Document Benchmarks (Best for RNSR)

#### QASPER (Scientific Paper QA) ⭐ Recommended
- **Paper**: Dasigi et al., NAACL 2021
- **URL**: https://allenai.org/data/qasper
- **Why ideal for RNSR**: Full scientific papers with sections, figures, and structure
- **Tests**: Hierarchical navigation, section-based reasoning, evidence extraction
- **Metrics**: Answer F1, Evidence F1

#### QuALITY (Long Document QA) ⭐ Recommended
- **Paper**: Pang et al., NAACL 2022
- **URL**: https://github.com/nyu-mll/quality
- **Why ideal for RNSR**: Long articles (2,000-8,000 words) requiring full document comprehension
- **Tests**: Long-range understanding, multiple-choice reasoning
- **Metrics**: Accuracy, Answer EM

#### NarrativeQA (Very Long Documents)
- **Paper**: Kočiský et al., TACL 2018
- **URL**: https://github.com/deepmind/narrativeqa
- **Why relevant**: Full books and scripts (10k-100k+ words)
- **Tests**: Extreme long-range comprehension
- **Metrics**: Answer F1, ROUGE-L

### Multi-hop Reasoning Benchmarks

#### HotpotQA (Multi-hop QA)
- **Paper**: Yang et al., EMNLP 2018
- **URL**: https://hotpotqa.github.io/
- **Why relevant**: Tests multi-hop reasoning across multiple paragraphs
- **Metrics**: Answer EM, Answer F1, Support EM, Support F1, Joint EM/F1

#### MuSiQue (Compositional Multi-hop)
- **Paper**: Trivedi et al., TACL 2022
- **URL**: https://github.com/StonyBrookNLP/musique
- **Why relevant**: 2-4 hop questions composed from single-hop - tests recursive reasoning
- **Metrics**: Answer F1, Support F1

### Retrieval Benchmarks

#### BEIR (Information Retrieval)
- **Paper**: Thakur et al., NeurIPS 2021
- **URL**: https://github.com/beir-cellar/beir
- **Why relevant**: 17+ diverse IR tasks for zero-shot retrieval evaluation
- **Metrics**: NDCG@10, MAP, Recall@100, Precision@10

### RAG Evaluation

#### RAGAS (RAG Metrics)
- **Library**: https://github.com/explodinggradients/ragas
- **Why relevant**: Standard metrics for RAG pipeline evaluation
- **Metrics**:
  - **Faithfulness**: Is the answer grounded in context?
  - **Answer Relevancy**: Does answer address the question?
  - **Context Precision**: Are retrieved contexts relevant?
  - **Context Recall**: Are all relevant contexts retrieved?

## Quick Start

### Install Dependencies

```bash
pip install datasets beir ragas sentence-transformers
```

### Run Evaluation Suite

```bash
# RECOMMENDED: Test on QASPER (scientific papers) - best for RNSR
python -m rnsr.benchmarks.evaluation_suite --datasets qasper --samples 50 --llm-provider gemini

# Test on QuALITY (long documents)
python -m rnsr.benchmarks.evaluation_suite --datasets quality --samples 50 --llm-provider gemini

# Multi-hop reasoning (short contexts)
python -m rnsr.benchmarks.evaluation_suite --datasets hotpotqa --samples 100 --llm-provider gemini

# Compare against multiple baselines
python -m rnsr.benchmarks.evaluation_suite \
  --datasets hotpotqa musique_ans \
  --baselines naive_chunk_512 naive_chunk_1024 semantic_chunk \
  --samples 200

# Full benchmark with BEIR
python -m rnsr.benchmarks.evaluation_suite \
  --datasets hotpotqa musique_ans beir_nfcorpus beir_scifact \
  --samples 100 \
  --output benchmark_results
```

### Use Programmatically

```python
from rnsr.benchmarks import (
    EvaluationSuite,
    EvaluationConfig,
    BenchmarkLoader,
    compare_rnsr_vs_baseline,
)

# Configure evaluation
config = EvaluationConfig(
    datasets=["hotpotqa", "musique_ans"],
    max_samples=100,
    baselines=["naive_chunk_512"],
    run_ragas=True,
)

# Run evaluation
suite = EvaluationSuite(config)
report = suite.run()

# Print results
report.print_summary()

# Access detailed metrics
for dataset, results in report.dataset_results.items():
    print(f"{dataset}: F1={results['rnsr_metrics']['answer_f1']:.3f}")
```

## Metrics Explained

### Multi-hop QA Metrics (HotpotQA, MuSiQue)

| Metric | Description | Why RNSR should improve |
|--------|-------------|-------------------------|
| **Answer EM** | Exact string match | Precise navigation to correct node |
| **Answer F1** | Token-level overlap | Context preservation through hierarchy |
| **Support EM** | Exact supporting facts match | Tree traversal captures reasoning chain |
| **Support F1** | Supporting facts overlap | Explainable path through tree |
| **Joint F1** | Answer × Support F1 | End-to-end multi-hop reasoning |

### Retrieval Metrics (BEIR)

| Metric | Description | Why RNSR should improve |
|--------|-------------|-------------------------|
| **NDCG@10** | Ranking quality | Hierarchical context improves ranking |
| **Precision@10** | Relevance of top-10 | Tree navigation reduces false positives |
| **Recall@100** | Coverage | Structured search finds more relevant content |

### RAGAS Metrics

| Metric | Description | Why RNSR should improve |
|--------|-------------|-------------------------|
| **Faithfulness** | Grounded in context | Variables store exact text, no hallucination |
| **Context Precision** | Retrieved relevance | Tree traversal is targeted, not broad |
| **Context Recall** | All relevant retrieved | Hierarchical search covers full structure |
| **Answer Relevancy** | Addresses question | Better context = better answers |

## Expected Results

Based on the research paper claims, we expect RNSR to show improvements in:

1. **Multi-hop reasoning (HotpotQA)**
   - Expected: +10-20% on Joint F1 vs naive chunking
   - Reason: Tree traversal preserves reasoning chains

2. **Complex documents**
   - Expected: +15-25% on Support F1
   - Reason: Hierarchical indexing maintains context

3. **Query latency (large documents)**
   - Expected: O(log n) vs O(n) scaling
   - Reason: Tree depth vs scanning all chunks

4. **Explainability**
   - Expected: Full trace of navigation path
   - Reason: Agent records every node visited

## Baseline Implementations

### NaiveChunkRAG

Standard chunking approach:
```python
from rnsr.benchmarks import NaiveChunkRAG

baseline = NaiveChunkRAG(
    chunk_size=512,
    chunk_overlap=50,
    top_k=5,
)
result = baseline.query("What is the main finding?", pdf_path)
```

### SemanticChunkRAG

Embedding-based boundary detection:
```python
from rnsr.benchmarks import SemanticChunkRAG

baseline = SemanticChunkRAG(
    similarity_threshold=0.7,
    top_k=5,
)
result = baseline.query("Compare sections A and B", pdf_path)
```

## Generating Comparison Reports

```python
from rnsr.benchmarks import compare_rnsr_vs_baseline

comparison = compare_rnsr_vs_baseline(
    rnsr_results={"answer_f1": 0.72, "support_f1": 0.68},
    baseline_results={"answer_f1": 0.58, "support_f1": 0.52},
    dataset_name="HotpotQA",
    baseline_name="naive_chunk_512",
)

print(comparison.summary())
# Output:
# ============================================================
# Comparison: RNSR vs naive_chunk_512
# Dataset: HotpotQA
# ============================================================
# 
# Metric                    RNSR    Baseline        Δ         %
# -----------------------------------------------------------------
# answer_f1                0.720      0.580    +0.140    +24.1%
# support_f1               0.680      0.520    +0.160    +30.8%
# =================================================================
```

## Adding Custom Benchmarks

```python
from rnsr.benchmarks import BenchmarkDataset, BenchmarkQuestion

# Create custom dataset
questions = [
    BenchmarkQuestion(
        id="q1",
        question="What is the revenue for Q4?",
        answer="$2.5 billion",
        supporting_facts=["Financial Summary", "Q4 Results"],
        context=["Revenue for Q4 2025 was $2.5 billion..."],
        reasoning_type="single-hop",
    ),
    # ... more questions
]

dataset = BenchmarkDataset(
    name="FinancialQA",
    description="Financial document QA benchmark",
    questions=questions,
    metrics=["answer_f1", "support_f1"],
    source_url="internal",
)

# Run evaluation
from rnsr.benchmarks import EvaluationSuite, EvaluationConfig

suite = EvaluationSuite(EvaluationConfig())
predictions, metrics = suite.evaluate_rnsr_on_dataset(dataset)
```

## Citation

If using these benchmarks, please cite the original papers:

```bibtex
@inproceedings{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and others},
  booktitle={EMNLP},
  year={2018}
}

@article{trivedi2022musique,
  title={MuSiQue: Multi-hop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and others},
  journal={TACL},
  year={2022}
}

@inproceedings{thakur2021beir,
  title={BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of IR Models},
  author={Thakur, Nandan and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2021}
}
```

## Module Structure

```
rnsr/benchmarks/
├── __init__.py              # Exports
├── performance.py           # Timing, memory benchmarks
├── quality.py               # Precision, recall, F1
├── runner.py                # Orchestration
├── standard_benchmarks.py   # HotpotQA, MuSiQue, BEIR, RAGAS
├── evaluation_suite.py      # Complete evaluation runner
└── README.md                # This file
```
