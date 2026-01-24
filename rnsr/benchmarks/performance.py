"""
Performance Benchmarks

Measures:
- Ingestion time (PDF → Tree)
- Indexing time (Tree → Skeleton + KV)
- Query latency (Question → Answer)
- Memory usage
- Throughput (pages/sec, queries/sec)
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    
    name: str
    duration_seconds: float
    memory_peak_mb: float
    memory_current_mb: float
    throughput: float | None = None
    throughput_unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        return self.duration_seconds * 1000
    
    def __str__(self) -> str:
        parts = [
            f"{self.name}:",
            f"  Time: {self.duration_ms:.2f}ms",
            f"  Memory Peak: {self.memory_peak_mb:.2f}MB",
        ]
        if self.throughput:
            parts.append(f"  Throughput: {self.throughput:.2f} {self.throughput_unit}")
        return "\n".join(parts)


@dataclass
class PerformanceBenchmark:
    """Collection of performance benchmark results."""
    
    ingestion: BenchmarkResult | None = None
    indexing: BenchmarkResult | None = None
    query_cold: BenchmarkResult | None = None
    query_warm: BenchmarkResult | None = None
    total_time_seconds: float = 0.0
    
    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "ingestion_ms": self.ingestion.duration_ms if self.ingestion else None,
            "indexing_ms": self.indexing.duration_ms if self.indexing else None,
            "query_cold_ms": self.query_cold.duration_ms if self.query_cold else None,
            "query_warm_ms": self.query_warm.duration_ms if self.query_warm else None,
            "total_time_seconds": self.total_time_seconds,
            "peak_memory_mb": max(
                r.memory_peak_mb for r in [self.ingestion, self.indexing, self.query_cold]
                if r is not None
            ) if any([self.ingestion, self.indexing, self.query_cold]) else 0,
        }
    
    def __str__(self) -> str:
        lines = ["=== Performance Benchmark Results ==="]
        for result in [self.ingestion, self.indexing, self.query_cold, self.query_warm]:
            if result:
                lines.append(str(result))
        lines.append(f"\nTotal Time: {self.total_time_seconds:.2f}s")
        return "\n".join(lines)


def _measure_execution(
    func: Callable[[], Any],
    name: str,
    warmup_runs: int = 0,
) -> tuple[Any, BenchmarkResult]:
    """
    Execute a function and measure time and memory.
    
    Args:
        func: Function to execute
        name: Name for the benchmark
        warmup_runs: Number of warmup runs before measurement
        
    Returns:
        Tuple of (function result, BenchmarkResult)
    """
    # Warmup
    for _ in range(warmup_runs):
        func()
        gc.collect()
    
    # Force garbage collection before measurement
    gc.collect()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Time the execution
    start_time = time.perf_counter()
    result = func()
    end_time = time.perf_counter()
    
    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    benchmark_result = BenchmarkResult(
        name=name,
        duration_seconds=end_time - start_time,
        memory_peak_mb=peak / 1024 / 1024,
        memory_current_mb=current / 1024 / 1024,
    )
    
    return result, benchmark_result


def run_ingestion_benchmark(
    pdf_path: Path | str,
    iterations: int = 1,
) -> BenchmarkResult:
    """
    Benchmark PDF ingestion.
    
    Args:
        pdf_path: Path to PDF file
        iterations: Number of iterations to average
        
    Returns:
        BenchmarkResult with timing and memory stats
    """
    from rnsr.ingestion import ingest_document
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Get file size
    file_size_mb = pdf_path.stat().st_size / 1024 / 1024
    
    results: list[BenchmarkResult] = []
    final_result = None
    
    for i in range(iterations):
        logger.info("ingestion_benchmark_iteration", iteration=i + 1, total=iterations)
        
        def run_ingestion():
            return ingest_document(pdf_path)
        
        ingestion_result, benchmark = _measure_execution(
            run_ingestion,
            f"Ingestion (iter {i + 1})",
        )
        
        # Calculate throughput (pages/sec)
        if hasattr(ingestion_result, 'tree') and ingestion_result.tree:
            page_count = ingestion_result.tree.total_nodes
            benchmark.throughput = page_count / benchmark.duration_seconds
            benchmark.throughput_unit = "nodes/sec"
        
        benchmark.metadata = {
            "file": pdf_path.name,
            "file_size_mb": file_size_mb,
            "tier_used": ingestion_result.tier_used if hasattr(ingestion_result, 'tier_used') else None,
        }
        
        results.append(benchmark)
        final_result = ingestion_result
    
    # Average results
    avg_duration = sum(r.duration_seconds for r in results) / len(results)
    max_memory = max(r.memory_peak_mb for r in results)
    avg_throughput = sum(r.throughput or 0 for r in results) / len(results)
    
    return BenchmarkResult(
        name="Ingestion",
        duration_seconds=avg_duration,
        memory_peak_mb=max_memory,
        memory_current_mb=results[-1].memory_current_mb,
        throughput=avg_throughput if avg_throughput > 0 else None,
        throughput_unit="nodes/sec",
        metadata={
            "iterations": iterations,
            "file": pdf_path.name,
            "file_size_mb": file_size_mb,
        },
    )


def run_query_benchmark(
    questions: list[str],
    skeleton: dict,
    kv_store: Any,
    warmup: bool = True,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Benchmark query execution.
    
    Args:
        questions: List of questions to benchmark
        skeleton: Skeleton index
        kv_store: KV store with content
        warmup: Whether to do warmup run
        
    Returns:
        Tuple of (cold_start_result, warm_result)
    """
    from rnsr.agent import run_navigator
    
    if not questions:
        raise ValueError("At least one question required")
    
    # Cold start benchmark (first query)
    def run_cold_query():
        return run_navigator(
            question=questions[0],
            skeleton=skeleton,
            kv_store=kv_store,
            max_iterations=10,
        )
    
    _, cold_result = _measure_execution(run_cold_query, "Query (Cold Start)")
    cold_result.metadata = {"question": questions[0][:50]}
    
    # Warm benchmark (average of remaining queries)
    warm_times: list[float] = []
    warm_memories: list[float] = []
    
    for q in questions[1:] if len(questions) > 1 else questions:
        def run_warm_query():
            return run_navigator(
                question=q,
                skeleton=skeleton,
                kv_store=kv_store,
                max_iterations=10,
            )
        
        _, warm_bench = _measure_execution(run_warm_query, "Query (Warm)")
        warm_times.append(warm_bench.duration_seconds)
        warm_memories.append(warm_bench.memory_peak_mb)
    
    warm_result = BenchmarkResult(
        name="Query (Warm)",
        duration_seconds=sum(warm_times) / len(warm_times) if warm_times else 0,
        memory_peak_mb=max(warm_memories) if warm_memories else 0,
        memory_current_mb=warm_memories[-1] if warm_memories else 0,
        throughput=len(warm_times) / sum(warm_times) if sum(warm_times) > 0 else None,
        throughput_unit="queries/sec",
        metadata={"query_count": len(warm_times)},
    )
    
    return cold_result, warm_result


def run_end_to_end_benchmark(
    pdf_path: Path | str,
    questions: list[str],
) -> PerformanceBenchmark:
    """
    Run complete end-to-end benchmark.
    
    Args:
        pdf_path: Path to PDF file
        questions: List of test questions
        
    Returns:
        PerformanceBenchmark with all results
    """
    from rnsr.indexing import build_skeleton_index
    from rnsr.ingestion import ingest_document
    
    pdf_path = Path(pdf_path)
    total_start = time.perf_counter()
    
    benchmark = PerformanceBenchmark()
    
    # 1. Ingestion benchmark
    logger.info("benchmark_ingestion_start", file=pdf_path.name)
    
    def do_ingest():
        return ingest_document(pdf_path)
    
    ingestion_result, benchmark.ingestion = _measure_execution(do_ingest, "Ingestion")
    benchmark.ingestion.metadata = {"file": pdf_path.name}
    
    # 2. Indexing benchmark
    logger.info("benchmark_indexing_start")
    
    def do_index():
        return build_skeleton_index(ingestion_result.tree)
    
    (skeleton, kv_store), benchmark.indexing = _measure_execution(do_index, "Indexing")
    benchmark.indexing.metadata = {"node_count": len(skeleton)}
    
    # 3. Query benchmarks
    if questions:
        logger.info("benchmark_query_start", question_count=len(questions))
        benchmark.query_cold, benchmark.query_warm = run_query_benchmark(
            questions, skeleton, kv_store
        )
    
    benchmark.total_time_seconds = time.perf_counter() - total_start
    
    logger.info("benchmark_complete", total_seconds=benchmark.total_time_seconds)
    
    return benchmark
