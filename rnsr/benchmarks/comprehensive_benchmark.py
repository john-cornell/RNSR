"""
Comprehensive Benchmark Suite

Evaluates RNSR against different configurations and approaches:
1. Standard Navigator (ToT-based)
2. RLM Navigator (pre-filtering + deep recursion + verification)
3. Vision Navigator (OCR-free page image analysis)
4. Hybrid Navigator (text + vision)

Benchmarks inspired by:
- FinanceBench: Financial document QA (PageIndex achieved 98.7%)
- OOLONG: Long context aggregation tasks
- Custom Multi-hop: Complex relational queries

Usage:
    from rnsr.benchmarks.comprehensive_benchmark import run_comprehensive_benchmark
    
    results = run_comprehensive_benchmark(
        pdf_paths=["contract.pdf", "report.pdf"],
        benchmark_type="all",  # or "financebench", "custom", etc.
    )
    results.print_report()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Benchmark Test Cases
# =============================================================================


@dataclass
class BenchmarkTestCase:
    """A single test case for benchmarking."""
    
    id: str
    question: str
    expected_answer: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    requires_multi_hop: bool = False
    requires_aggregation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    
    test_case_id: str
    method: str
    answer: str
    is_correct: bool
    confidence: float
    latency_ms: float
    tokens_used: int = 0
    iterations: int = 0
    error: str | None = None
    trace: list[dict] = field(default_factory=list)


@dataclass
class MethodResults:
    """Aggregated results for a single method."""
    
    method: str
    total_tests: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)
    
    def compute_stats(self) -> None:
        """Compute aggregate statistics."""
        if not self.results:
            return
        
        self.total_tests = len(self.results)
        self.correct = sum(1 for r in self.results if r.is_correct)
        self.accuracy = self.correct / self.total_tests if self.total_tests > 0 else 0
        
        latencies = [r.latency_ms for r in self.results if r.latency_ms > 0]
        self.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
        
        confidences = [r.confidence for r in self.results]
        self.avg_confidence = sum(confidences) / len(confidences) if confidences else 0


@dataclass
class ComprehensiveBenchmarkReport:
    """Complete benchmark report with all methods."""
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    benchmark_type: str = "comprehensive"
    document_count: int = 0
    test_case_count: int = 0
    
    # Results by method
    standard_navigator: MethodResults | None = None
    rlm_navigator: MethodResults | None = None
    vision_navigator: MethodResults | None = None
    hybrid_navigator: MethodResults | None = None
    
    # Comparison summary
    best_method: str = ""
    best_accuracy: float = 0.0
    
    def print_report(self) -> None:
        """Print a formatted report."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE RNSR BENCHMARK REPORT")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Documents: {self.document_count}")
        print(f"Test Cases: {self.test_case_count}")
        print()
        
        methods = [
            ("Standard Navigator", self.standard_navigator),
            ("RLM Navigator", self.rlm_navigator),
            ("Vision Navigator", self.vision_navigator),
            ("Hybrid Navigator", self.hybrid_navigator),
        ]
        
        print("-" * 70)
        print(f"{'Method':<25} {'Accuracy':<12} {'Avg Latency':<15} {'Confidence':<12}")
        print("-" * 70)
        
        for name, result in methods:
            if result:
                acc = f"{result.accuracy:.1%}"
                lat = f"{result.avg_latency_ms:.0f}ms"
                conf = f"{result.avg_confidence:.2f}"
                print(f"{name:<25} {acc:<12} {lat:<15} {conf:<12}")
        
        print("-" * 70)
        print(f"\nBest Method: {self.best_method} ({self.best_accuracy:.1%} accuracy)")
        print("=" * 70)
    
    def to_json(self, path: Path | str) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": self.timestamp,
            "benchmark_type": self.benchmark_type,
            "document_count": self.document_count,
            "test_case_count": self.test_case_count,
            "best_method": self.best_method,
            "best_accuracy": self.best_accuracy,
            "methods": {},
        }
        
        for name, result in [
            ("standard_navigator", self.standard_navigator),
            ("rlm_navigator", self.rlm_navigator),
            ("vision_navigator", self.vision_navigator),
            ("hybrid_navigator", self.hybrid_navigator),
        ]:
            if result:
                data["methods"][name] = {
                    "total_tests": result.total_tests,
                    "correct": result.correct,
                    "accuracy": result.accuracy,
                    "avg_latency_ms": result.avg_latency_ms,
                    "avg_confidence": result.avg_confidence,
                }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info("benchmark_report_saved", path=str(path))


# =============================================================================
# Standard Benchmark Test Suites
# =============================================================================


def get_financebench_cases() -> list[BenchmarkTestCase]:
    """
    FinanceBench-style test cases.
    
    PageIndex achieved 98.7% accuracy on FinanceBench.
    These are financial document QA questions requiring multi-hop reasoning.
    """
    return [
        BenchmarkTestCase(
            id="fin_001",
            question="What was the total revenue in the most recent fiscal year?",
            expected_keywords=["revenue", "fiscal", "year", "total"],
            category="financial",
            difficulty="medium",
        ),
        BenchmarkTestCase(
            id="fin_002",
            question="What is the year-over-year growth in net income?",
            expected_keywords=["growth", "net income", "year-over-year"],
            category="financial",
            difficulty="hard",
            requires_multi_hop=True,
        ),
        BenchmarkTestCase(
            id="fin_003",
            question="What are the key risk factors mentioned in the report?",
            expected_keywords=["risk", "factors"],
            category="financial",
            difficulty="medium",
            requires_aggregation=True,
        ),
        BenchmarkTestCase(
            id="fin_004",
            question="Compare the gross margin between Q1 and Q4.",
            expected_keywords=["gross margin", "Q1", "Q4", "compare"],
            category="financial",
            difficulty="hard",
            requires_multi_hop=True,
        ),
        BenchmarkTestCase(
            id="fin_005",
            question="What is the company's debt-to-equity ratio?",
            expected_keywords=["debt", "equity", "ratio"],
            category="financial",
            difficulty="medium",
        ),
    ]


def get_oolong_style_cases() -> list[BenchmarkTestCase]:
    """
    OOLONG-style test cases.
    
    OOLONG tests long context reasoning and aggregation.
    These require processing many parts of a document.
    """
    return [
        BenchmarkTestCase(
            id="ool_001",
            question="List all the parties mentioned in this agreement.",
            expected_keywords=["parties", "agreement"],
            category="aggregation",
            difficulty="hard",
            requires_aggregation=True,
        ),
        BenchmarkTestCase(
            id="ool_002",
            question="How many sections are there in total?",
            expected_keywords=["sections", "total"],
            category="structure",
            difficulty="easy",
        ),
        BenchmarkTestCase(
            id="ool_003",
            question="What are all the obligations of Party A?",
            expected_keywords=["obligations", "party"],
            category="aggregation",
            difficulty="hard",
            requires_aggregation=True,
        ),
        BenchmarkTestCase(
            id="ool_004",
            question="Summarize the key terms across all sections.",
            expected_keywords=["key terms", "summarize", "sections"],
            category="aggregation",
            difficulty="hard",
            requires_aggregation=True,
        ),
    ]


def get_multi_hop_cases() -> list[BenchmarkTestCase]:
    """
    Multi-hop reasoning test cases.
    
    These require connecting information from multiple sections.
    """
    return [
        BenchmarkTestCase(
            id="mh_001",
            question="If the termination clause is triggered, what penalties apply according to the payment terms?",
            expected_keywords=["termination", "penalties", "payment"],
            category="multi_hop",
            difficulty="hard",
            requires_multi_hop=True,
        ),
        BenchmarkTestCase(
            id="mh_002",
            question="How do the warranties in Section 5 relate to the limitations in Section 8?",
            expected_keywords=["warranties", "limitations", "section"],
            category="multi_hop",
            difficulty="hard",
            requires_multi_hop=True,
        ),
        BenchmarkTestCase(
            id="mh_003",
            question="What happens to the IP rights if the contract is terminated for cause?",
            expected_keywords=["IP", "intellectual property", "terminated", "cause"],
            category="multi_hop",
            difficulty="hard",
            requires_multi_hop=True,
        ),
    ]


# =============================================================================
# Benchmark Runner
# =============================================================================


class ComprehensiveBenchmarkRunner:
    """
    Runs comprehensive benchmarks across all navigator types.
    """
    
    def __init__(
        self,
        pdf_paths: list[Path | str],
        test_cases: list[BenchmarkTestCase] | None = None,
        methods: list[str] | None = None,
    ):
        self.pdf_paths = [Path(p) for p in pdf_paths]
        self.test_cases = test_cases or self._get_default_cases()
        self.methods = methods or ["standard", "rlm", "vision", "hybrid"]
        
        # Indexes (built on first use)
        self._indexes: dict[str, tuple[dict, Any]] = {}
    
    def _get_default_cases(self) -> list[BenchmarkTestCase]:
        """Get default test cases."""
        cases = []
        cases.extend(get_financebench_cases())
        cases.extend(get_oolong_style_cases())
        cases.extend(get_multi_hop_cases())
        return cases
    
    def _get_or_build_index(self, pdf_path: Path) -> tuple[dict, Any]:
        """Get or build index for a PDF."""
        key = str(pdf_path)
        if key in self._indexes:
            return self._indexes[key]
        
        from rnsr import ingest_document, build_skeleton_index
        
        result = ingest_document(pdf_path)
        skeleton, kv_store = build_skeleton_index(result.tree)
        self._indexes[key] = (skeleton, kv_store)
        return skeleton, kv_store
    
    def run(self) -> ComprehensiveBenchmarkReport:
        """Run all benchmarks."""
        report = ComprehensiveBenchmarkReport(
            document_count=len(self.pdf_paths),
            test_case_count=len(self.test_cases),
        )
        
        logger.info(
            "comprehensive_benchmark_start",
            documents=len(self.pdf_paths),
            test_cases=len(self.test_cases),
            methods=self.methods,
        )
        
        # Run each method
        if "standard" in self.methods:
            report.standard_navigator = self._run_method(
                "standard",
                self._run_standard_navigator,
            )
        
        if "rlm" in self.methods:
            report.rlm_navigator = self._run_method(
                "rlm",
                self._run_rlm_navigator,
            )
        
        if "vision" in self.methods:
            report.vision_navigator = self._run_method(
                "vision",
                self._run_vision_navigator,
            )
        
        if "hybrid" in self.methods:
            report.hybrid_navigator = self._run_method(
                "hybrid",
                self._run_hybrid_navigator,
            )
        
        # Determine best method
        best_accuracy = 0.0
        best_method = ""
        for name, result in [
            ("Standard Navigator", report.standard_navigator),
            ("RLM Navigator", report.rlm_navigator),
            ("Vision Navigator", report.vision_navigator),
            ("Hybrid Navigator", report.hybrid_navigator),
        ]:
            if result and result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
                best_method = name
        
        report.best_method = best_method
        report.best_accuracy = best_accuracy
        
        logger.info("comprehensive_benchmark_complete", best_method=best_method)
        
        return report
    
    def _run_method(
        self,
        method_name: str,
        runner_fn: Callable,
    ) -> MethodResults:
        """Run a single method across all test cases."""
        results = MethodResults(method=method_name)
        
        logger.info("running_method", method=method_name)
        
        for pdf_path in self.pdf_paths:
            if not pdf_path.exists():
                logger.warning("pdf_not_found", path=str(pdf_path))
                continue
            
            for test_case in self.test_cases:
                try:
                    result = runner_fn(pdf_path, test_case)
                    results.results.append(result)
                except Exception as e:
                    logger.warning(
                        "test_case_failed",
                        method=method_name,
                        test_case=test_case.id,
                        error=str(e),
                    )
                    results.results.append(BenchmarkResult(
                        test_case_id=test_case.id,
                        method=method_name,
                        answer="",
                        is_correct=False,
                        confidence=0.0,
                        latency_ms=0,
                        error=str(e),
                    ))
        
        results.compute_stats()
        return results
    
    def _run_standard_navigator(
        self,
        pdf_path: Path,
        test_case: BenchmarkTestCase,
    ) -> BenchmarkResult:
        """Run standard ToT-based navigator."""
        from rnsr.agent import run_navigator
        
        skeleton, kv_store = self._get_or_build_index(pdf_path)
        
        start_time = time.time()
        result = run_navigator(
            test_case.question,
            skeleton,
            kv_store,
            metadata=test_case.metadata,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0.0)
        
        is_correct = self._evaluate_answer(answer, test_case)
        
        return BenchmarkResult(
            test_case_id=test_case.id,
            method="standard",
            answer=answer,
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
            iterations=len(result.get("trace", [])),
            trace=result.get("trace", []),
        )
    
    def _run_rlm_navigator(
        self,
        pdf_path: Path,
        test_case: BenchmarkTestCase,
    ) -> BenchmarkResult:
        """Run RLM navigator with full features."""
        from rnsr.agent.rlm_navigator import RLMConfig, run_rlm_navigator
        
        skeleton, kv_store = self._get_or_build_index(pdf_path)
        
        config = RLMConfig(
            enable_pre_filtering=True,
            enable_verification=True,
            max_recursion_depth=3,
        )
        
        start_time = time.time()
        result = run_rlm_navigator(
            test_case.question,
            skeleton,
            kv_store,
            config=config,
            metadata=test_case.metadata,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0.0)
        
        is_correct = self._evaluate_answer(answer, test_case)
        
        return BenchmarkResult(
            test_case_id=test_case.id,
            method="rlm",
            answer=answer,
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
            iterations=result.get("iteration", 0),
            trace=result.get("trace", []),
        )
    
    def _run_vision_navigator(
        self,
        pdf_path: Path,
        test_case: BenchmarkTestCase,
    ) -> BenchmarkResult:
        """Run vision-based navigator."""
        from rnsr.ingestion.vision_retrieval import create_vision_navigator
        
        navigator = create_vision_navigator(pdf_path)
        
        start_time = time.time()
        result = navigator.navigate(
            test_case.question,
            metadata=test_case.metadata,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0.0)
        
        is_correct = self._evaluate_answer(answer, test_case)
        
        return BenchmarkResult(
            test_case_id=test_case.id,
            method="vision",
            answer=answer,
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
            trace=result.get("trace", []),
        )
    
    def _run_hybrid_navigator(
        self,
        pdf_path: Path,
        test_case: BenchmarkTestCase,
    ) -> BenchmarkResult:
        """Run hybrid text+vision navigator."""
        from rnsr.ingestion.vision_retrieval import create_hybrid_navigator
        
        skeleton, kv_store = self._get_or_build_index(pdf_path)
        navigator = create_hybrid_navigator(pdf_path, skeleton, kv_store)
        
        start_time = time.time()
        result = navigator.navigate(
            test_case.question,
            metadata=test_case.metadata,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        answer = result.get("combined_answer", "")
        confidence = result.get("confidence", 0.0)
        
        is_correct = self._evaluate_answer(answer, test_case)
        
        return BenchmarkResult(
            test_case_id=test_case.id,
            method="hybrid",
            answer=answer,
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
        )
    
    def _evaluate_answer(
        self,
        answer: str,
        test_case: BenchmarkTestCase,
    ) -> bool:
        """Evaluate if an answer is correct."""
        if not answer:
            return False
        
        answer_lower = answer.lower()
        
        # Check expected answer if provided
        if test_case.expected_answer:
            if test_case.expected_answer.lower() in answer_lower:
                return True
        
        # Check keywords
        if test_case.expected_keywords:
            matches = sum(
                1 for kw in test_case.expected_keywords
                if kw.lower() in answer_lower
            )
            # Require at least half the keywords
            required = len(test_case.expected_keywords) // 2 + 1
            return matches >= required
        
        # Default: consider non-empty as potentially correct
        return len(answer) > 20


# =============================================================================
# Convenience Functions
# =============================================================================


def run_comprehensive_benchmark(
    pdf_paths: list[Path | str],
    benchmark_type: Literal["all", "financebench", "oolong", "multihop", "custom"] = "all",
    custom_test_cases: list[BenchmarkTestCase] | None = None,
    methods: list[str] | None = None,
    output_path: Path | str | None = None,
) -> ComprehensiveBenchmarkReport:
    """
    Run a comprehensive benchmark across all navigator types.
    
    Args:
        pdf_paths: List of PDF files to benchmark.
        benchmark_type: Type of benchmark to run.
        custom_test_cases: Custom test cases (for 'custom' type).
        methods: List of methods to benchmark ['standard', 'rlm', 'vision', 'hybrid'].
        output_path: Path to save JSON report.
        
    Returns:
        ComprehensiveBenchmarkReport with all results.
        
    Example:
        from rnsr.benchmarks.comprehensive_benchmark import run_comprehensive_benchmark
        
        results = run_comprehensive_benchmark(
            pdf_paths=["financial_report.pdf"],
            benchmark_type="financebench",
            methods=["standard", "rlm"],
        )
        results.print_report()
    """
    # Get test cases based on type
    if benchmark_type == "all":
        test_cases = []
        test_cases.extend(get_financebench_cases())
        test_cases.extend(get_oolong_style_cases())
        test_cases.extend(get_multi_hop_cases())
    elif benchmark_type == "financebench":
        test_cases = get_financebench_cases()
    elif benchmark_type == "oolong":
        test_cases = get_oolong_style_cases()
    elif benchmark_type == "multihop":
        test_cases = get_multi_hop_cases()
    elif benchmark_type == "custom":
        test_cases = custom_test_cases or []
    else:
        test_cases = []
    
    # Run benchmark
    runner = ComprehensiveBenchmarkRunner(
        pdf_paths=pdf_paths,
        test_cases=test_cases,
        methods=methods,
    )
    
    report = runner.run()
    report.benchmark_type = benchmark_type
    
    # Save if path provided
    if output_path:
        report.to_json(output_path)
    
    return report


def quick_benchmark(
    pdf_path: Path | str,
    question: str,
) -> dict[str, Any]:
    """
    Quick benchmark a single question across all methods.
    
    Args:
        pdf_path: Path to PDF file.
        question: Question to ask.
        
    Returns:
        Dict with results from each method.
    """
    test_case = BenchmarkTestCase(
        id="quick",
        question=question,
    )
    
    runner = ComprehensiveBenchmarkRunner(
        pdf_paths=[pdf_path],
        test_cases=[test_case],
    )
    
    report = runner.run()
    
    return {
        "standard": report.standard_navigator.results[0] if report.standard_navigator else None,
        "rlm": report.rlm_navigator.results[0] if report.rlm_navigator else None,
        "vision": report.vision_navigator.results[0] if report.vision_navigator else None,
        "hybrid": report.hybrid_navigator.results[0] if report.hybrid_navigator else None,
    }
