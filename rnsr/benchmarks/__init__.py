"""
RNSR Benchmarking Suite

Measures:
1. Performance: Ingestion speed, query latency, memory usage
2. Quality: Retrieval accuracy, answer relevance (requires ground truth)
3. Comparison: RNSR vs baseline chunking approaches
4. Standard Benchmarks: HotpotQA, MuSiQue, BEIR, RAGAS

Standard RAG Benchmarks:
- HotpotQA: Multi-hop question answering (EMNLP 2018)
- MuSiQue: Compositional multi-hop QA (TACL 2022)
- BEIR: Information retrieval benchmark (NeurIPS 2021)
- RAGAS: RAG evaluation metrics (faithfulness, relevance, etc.)
"""

from rnsr.benchmarks.performance import (
    PerformanceBenchmark,
    BenchmarkResult,
    run_ingestion_benchmark,
    run_query_benchmark,
)
from rnsr.benchmarks.quality import (
    QualityBenchmark,
    QualityMetrics,
    evaluate_retrieval,
)
from rnsr.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    run_full_benchmark,
)
from rnsr.benchmarks.standard_benchmarks import (
    # Baselines
    NaiveChunkRAG,
    SemanticChunkRAG,
    BaselineResult,
    # Benchmark datasets
    BenchmarkLoader,
    BenchmarkDataset,
    BenchmarkQuestion,
    # RAGAS metrics
    RAGASEvaluator,
    RAGASMetrics,
    # Multi-hop metrics
    MultiHopMetrics,
    evaluate_multihop,
    # Comparison
    compare_rnsr_vs_baseline,
)
from rnsr.benchmarks.evaluation_suite import (
    EvaluationSuite,
    EvaluationConfig,
    EvaluationReport,
    RNSRBenchmarkAdapter,
)

__all__ = [
    # Performance
    "PerformanceBenchmark",
    "BenchmarkResult",
    "run_ingestion_benchmark",
    "run_query_benchmark",
    # Quality
    "QualityBenchmark",
    "QualityMetrics",
    "evaluate_retrieval",
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "run_full_benchmark",
    # Standard Benchmarks
    "NaiveChunkRAG",
    "SemanticChunkRAG",
    "BaselineResult",
    "BenchmarkLoader",
    "BenchmarkDataset",
    "BenchmarkQuestion",
    "RAGASEvaluator",
    "RAGASMetrics",
    "MultiHopMetrics",
    "evaluate_multihop",
    "compare_rnsr_vs_baseline",
    # Evaluation Suite
    "EvaluationSuite",
    "EvaluationConfig",
    "EvaluationReport",
    "RNSRBenchmarkAdapter",
]
