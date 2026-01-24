"""
Benchmark Runner

Orchestrates performance and quality benchmarks,
compares RNSR against baseline approaches.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog

from rnsr.benchmarks.performance import (
    BenchmarkResult,
    PerformanceBenchmark,
    run_end_to_end_benchmark,
)
from rnsr.benchmarks.quality import (
    QualityBenchmark,
    QualityMetrics,
    TestCase,
    evaluate_retrieval,
)

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Files to benchmark
    pdf_paths: list[Path] = field(default_factory=list)
    
    # Test questions
    questions: list[str] = field(default_factory=list)
    
    # Quality test cases (optional)
    quality_test_cases: list[TestCase] = field(default_factory=list)
    
    # Settings
    iterations: int = 1
    warmup: bool = True
    compute_quality: bool = True
    compare_baseline: bool = False
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    
    @classmethod
    def from_json(cls, path: Path | str) -> "BenchmarkConfig":
        """Load config from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            pdf_paths=[Path(p) for p in data.get("pdf_paths", [])],
            questions=data.get("questions", []),
            quality_test_cases=[
                TestCase(
                    question=tc["question"],
                    expected_nodes=tc.get("expected_nodes", []),
                    expected_answer=tc.get("expected_answer", ""),
                    keywords=tc.get("keywords", []),
                )
                for tc in data.get("quality_test_cases", [])
            ],
            iterations=data.get("iterations", 1),
            warmup=data.get("warmup", True),
            compute_quality=data.get("compute_quality", True),
            compare_baseline=data.get("compare_baseline", False),
            output_dir=Path(data.get("output_dir", "benchmark_results")),
        )


@dataclass
class ComparisonResult:
    """Comparison between RNSR and baseline."""
    
    method: Literal["rnsr", "baseline_chunk", "baseline_semantic"]
    performance: PerformanceBenchmark | None = None
    quality: QualityMetrics | None = None
    
    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "performance": self.performance.summary() if self.performance else None,
            "quality": self.quality.summary() if self.quality else None,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: BenchmarkConfig | None = None
    
    # Results per file
    file_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Aggregated results
    rnsr_performance: PerformanceBenchmark | None = None
    rnsr_quality: QualityMetrics | None = None
    
    # Comparison (if enabled)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    
    def to_json(self, path: Path | str) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": self.timestamp,
            "file_results": self.file_results,
            "rnsr_performance": self.rnsr_performance.summary() if self.rnsr_performance else None,
            "rnsr_quality": self.rnsr_quality.summary() if self.rnsr_quality else None,
            "comparisons": [c.summary() for c in self.comparisons],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info("benchmark_report_saved", path=str(path))
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("RNSR BENCHMARK REPORT")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Files benchmarked: {len(self.file_results)}")
        
        if self.rnsr_performance:
            print("\n--- Performance ---")
            print(self.rnsr_performance)
        
        if self.rnsr_quality:
            print("\n--- Quality ---")
            print(self.rnsr_quality)
        
        if self.comparisons:
            print("\n--- Comparison ---")
            for comp in self.comparisons:
                print(f"\n{comp.method}:")
                if comp.performance:
                    perf = comp.performance.summary()
                    print(f"  Query Latency: {perf.get('query_warm_ms', 'N/A')}ms")
                if comp.quality:
                    qual = comp.quality.summary()
                    print(f"  F1 Score: {qual.get('f1', 'N/A'):.3f}")
        
        print("\n" + "=" * 60)


class BenchmarkRunner:
    """Runs benchmarks according to configuration."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.report = BenchmarkReport(config=config)
    
    def run(self) -> BenchmarkReport:
        """Run all configured benchmarks."""
        logger.info("benchmark_runner_start", file_count=len(self.config.pdf_paths))
        
        all_performance: list[PerformanceBenchmark] = []
        all_quality: list[QualityMetrics] = []
        
        for pdf_path in self.config.pdf_paths:
            if not pdf_path.exists():
                logger.warning("pdf_not_found", path=str(pdf_path))
                continue
            
            logger.info("benchmarking_file", file=pdf_path.name)
            
            # Performance benchmark
            perf = run_end_to_end_benchmark(
                pdf_path,
                self.config.questions,
            )
            all_performance.append(perf)
            
            # Quality benchmark (if test cases provided)
            quality = None
            if self.config.compute_quality and self.config.quality_test_cases:
                from rnsr.indexing import build_skeleton_index
                from rnsr.ingestion import ingest_document
                
                result = ingest_document(pdf_path)
                skeleton, kv_store = build_skeleton_index(result.tree)
                
                quality = evaluate_retrieval(
                    skeleton,
                    kv_store,
                    self.config.quality_test_cases,
                )
                all_quality.append(quality)
            
            # Store per-file results
            self.report.file_results[pdf_path.name] = {
                "performance": perf.summary(),
                "quality": quality.summary() if quality else None,
            }
        
        # Aggregate results
        if all_performance:
            self.report.rnsr_performance = self._aggregate_performance(all_performance)
        
        if all_quality:
            self.report.rnsr_quality = self._aggregate_quality(all_quality)
        
        # Run baseline comparison if enabled
        if self.config.compare_baseline:
            self._run_baseline_comparison()
        
        logger.info("benchmark_runner_complete")
        
        return self.report
    
    def _aggregate_performance(
        self, 
        results: list[PerformanceBenchmark],
    ) -> PerformanceBenchmark:
        """Aggregate multiple performance results."""
        # For now, just return the last one
        # TODO: Implement proper averaging
        return results[-1] if results else PerformanceBenchmark()
    
    def _aggregate_quality(
        self,
        results: list[QualityMetrics],
    ) -> QualityMetrics:
        """Aggregate multiple quality results."""
        combined = QualityMetrics()
        for r in results:
            combined.retrieval_metrics.extend(r.retrieval_metrics)
            combined.answer_metrics.extend(r.answer_metrics)
        return combined
    
    def _run_baseline_comparison(self) -> None:
        """Run baseline chunking approach for comparison."""
        logger.info("running_baseline_comparison")
        
        # TODO: Implement baseline comparison
        # This would use simple fixed-size chunking instead of RNSR
        
        baseline = ComparisonResult(
            method="baseline_chunk",
            performance=None,  # Would run baseline benchmark
            quality=None,  # Would evaluate baseline quality
        )
        self.report.comparisons.append(baseline)


def run_full_benchmark(
    pdf_paths: list[Path | str],
    questions: list[str] | None = None,
    output_dir: Path | str = "benchmark_results",
) -> BenchmarkReport:
    """
    Convenience function to run a full benchmark.
    
    Args:
        pdf_paths: List of PDF files to benchmark
        questions: Test questions (optional)
        output_dir: Directory for results
        
    Returns:
        BenchmarkReport with all results
    """
    config = BenchmarkConfig(
        pdf_paths=[Path(p) for p in pdf_paths],
        questions=questions or [
            "What is this document about?",
            "What are the main sections?",
            "What are the key findings?",
        ],
        output_dir=Path(output_dir),
    )
    
    runner = BenchmarkRunner(config)
    report = runner.run()
    
    # Save results
    output_path = config.output_dir / f"benchmark_{int(time.time())}.json"
    report.to_json(output_path)
    report.print_summary()
    
    return report
