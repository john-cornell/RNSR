#!/usr/bin/env python3
"""
RNSR Generalization Benchmark Suite

Runs RNSR against multiple standard benchmarks to prove generalization
beyond FinanceBench. Produces a combined report with per-benchmark metrics.

Benchmarks:
1. FinanceBench  — Financial document QA (baseline: 100% accuracy)
2. MultiHiertt   — Multi-step hierarchical table reasoning (ACL 2022)
3. TAT-QA        — Tabular + textual financial QA (ACL 2021)
4. QASPER        — Scientific paper QA (NAACL 2021)
5. DocVQA        — Visual document understanding (WACV 2021)

Usage:
    # Run all benchmarks (10 samples each for quick validation)
    python run_all_benchmarks.py --max-samples 10

    # Run specific benchmark
    python run_all_benchmarks.py --benchmarks tatqa qasper --max-samples 50

    # Full run (no sample limit)
    python run_all_benchmarks.py --full
"""

from __future__ import annotations

import argparse
import json
import re
import time
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkScore:
    """Score for a single benchmark."""

    name: str
    total_questions: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_f1: float = 0.0
    avg_latency_s: float = 0.0
    errors: int = 0
    per_type_scores: dict[str, float] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CombinedReport:
    """Combined report across all benchmarks."""

    timestamp: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    max_samples: int | None = None
    benchmarks: list[BenchmarkScore] = field(default_factory=list)
    overall_accuracy: float = 0.0
    overall_f1: float = 0.0
    total_questions: int = 0
    total_correct: int = 0
    total_errors: int = 0
    runtime_s: float = 0.0

    def compute_overall(self) -> None:
        """Compute aggregate stats across all benchmarks."""
        self.total_questions = sum(b.total_questions for b in self.benchmarks)
        self.total_correct = sum(b.correct for b in self.benchmarks)
        self.total_errors = sum(b.errors for b in self.benchmarks)
        self.overall_accuracy = (
            self.total_correct / self.total_questions if self.total_questions > 0 else 0
        )
        f1_scores = [b.avg_f1 for b in self.benchmarks if b.total_questions > 0]
        self.overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    def print_report(self) -> None:
        """Print a formatted report to stdout."""
        print("\n" + "=" * 78)
        print("RNSR GENERALIZATION BENCHMARK REPORT")
        print("=" * 78)
        print(f"  Timestamp:    {self.timestamp}")
        print(f"  LLM:          {self.llm_provider} / {self.llm_model}")
        print(f"  Max samples:  {self.max_samples or 'unlimited'}")
        print(f"  Runtime:      {self.runtime_s:.1f}s")
        print("=" * 78)
        print()

        # Per-benchmark results
        print(f"{'Benchmark':<20} {'Questions':>10} {'Correct':>10} {'Accuracy':>10} {'F1':>10} {'Errors':>8}")
        print("-" * 78)

        for b in self.benchmarks:
            print(
                f"{b.name:<20} {b.total_questions:>10} {b.correct:>10} "
                f"{b.accuracy:>9.1%} {b.avg_f1:>9.3f} {b.errors:>8}"
            )

        print("-" * 78)
        print(
            f"{'OVERALL':<20} {self.total_questions:>10} {self.total_correct:>10} "
            f"{self.overall_accuracy:>9.1%} {self.overall_f1:>9.3f} {self.total_errors:>8}"
        )
        print("=" * 78)

        # Per-type breakdown if available
        for b in self.benchmarks:
            if b.per_type_scores:
                print(f"\n  {b.name} — Per-type breakdown:")
                for answer_type, score in sorted(b.per_type_scores.items()):
                    print(f"    {answer_type:<30} {score:.1%}")

    def save(self, path: str | Path) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        print(f"\nReport saved to: {path}")


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _is_correct(prediction: str, ground_truth: str, all_answers: list[str] | None = None) -> bool:
    """Check if prediction matches ground truth (flexible matching)."""
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()

    # Exact match
    if pred == gt:
        return True

    # Check all acceptable answers
    if all_answers:
        for ans in all_answers:
            if pred == ans.strip().lower():
                return True

    # Containment check (ground truth is contained in prediction)
    if gt and gt in pred:
        return True

    # Numeric matching (handles "1,234" vs "1234", "$1,234" vs "1234")
    pred_nums = re.findall(r"[\d,.]+", pred)
    gt_nums = re.findall(r"[\d,.]+", gt)
    if pred_nums and gt_nums:
        pred_clean = pred_nums[0].replace(",", "")
        gt_clean = gt_nums[0].replace(",", "")
        try:
            if abs(float(pred_clean) - float(gt_clean)) < 0.01:
                return True
        except ValueError:
            pass

    return False


def run_benchmark_suite(
    benchmarks: list[str] | None = None,
    max_samples: int | None = 10,
    llm_provider: str = "gemini",
    llm_model: str = "gemini-2.5-flash",
    output_path: str = "benchmark_results/generalization_report.json",
) -> CombinedReport:
    """
    Run the full benchmark suite.

    Args:
        benchmarks: List of benchmark names to run. None = all.
        max_samples: Max samples per benchmark. None = unlimited.
        llm_provider: LLM provider to use.
        llm_model: LLM model to use.
        output_path: Where to save the report.

    Returns:
        CombinedReport with all results.
    """
    from rnsr.benchmarks.evaluation_suite import RNSRBenchmarkAdapter

    all_benchmarks = ["financebench", "multihiertt", "tatqa", "qasper", "docvqa"]
    selected = benchmarks or all_benchmarks

    report = CombinedReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_samples=max_samples,
    )

    adapter = RNSRBenchmarkAdapter(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    start_total = time.perf_counter()

    for bench_name in selected:
        print(f"\n{'='*60}")
        print(f"  Running: {bench_name.upper()}")
        print(f"{'='*60}")

        try:
            dataset = _load_dataset(bench_name, max_samples)
        except Exception as e:
            logger.error(f"Failed to load {bench_name}", error=str(e))
            report.benchmarks.append(
                BenchmarkScore(name=bench_name, errors=1)
            )
            continue

        if not dataset.questions:
            print(f"  ⚠ No questions loaded for {bench_name}, skipping.")
            report.benchmarks.append(BenchmarkScore(name=bench_name))
            continue

        print(f"  Loaded {len(dataset.questions)} questions")

        score = BenchmarkScore(
            name=bench_name,
            total_questions=len(dataset.questions),
        )

        f1_scores: list[float] = []
        latencies: list[float] = []
        type_correct: dict[str, int] = {}
        type_total: dict[str, int] = {}

        for i, q in enumerate(dataset.questions):
            print(f"  [{i + 1}/{len(dataset.questions)}] {q.question[:80]}...")

            try:
                start = time.perf_counter()

                # Route to PDF or context-based answering
                if q.metadata.get("pdf_path") and Path(q.metadata["pdf_path"]).exists():
                    result = adapter.answer_from_pdf(
                        q.question,
                        Path(q.metadata["pdf_path"]),
                        metadata=q.metadata,
                    )
                else:
                    result = adapter.answer_from_context(
                        q.question,
                        q.context,
                        metadata=q.metadata,
                    )

                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

                # Evaluate
                all_answers = q.metadata.get("all_answers")
                correct = _is_correct(result.answer, q.answer, all_answers)
                f1 = _compute_f1(result.answer, q.answer)

                if correct:
                    score.correct += 1
                f1_scores.append(f1)

                # Per-type tracking
                answer_type = q.metadata.get("answer_type", q.reasoning_type)
                type_total[answer_type] = type_total.get(answer_type, 0) + 1
                if correct:
                    type_correct[answer_type] = type_correct.get(answer_type, 0) + 1

                score.details.append({
                    "id": q.id,
                    "question": q.question,
                    "expected": q.answer,
                    "predicted": result.answer,
                    "correct": correct,
                    "f1": f1,
                    "latency_s": elapsed,
                    "answer_type": answer_type,
                })

                status = "✓" if correct else "✗"
                print(f"    {status} F1={f1:.2f} ({elapsed:.1f}s)")

            except Exception as e:
                score.errors += 1
                logger.warning(
                    "Question failed",
                    question_id=q.id,
                    error=str(e),
                )
                print(f"    ✗ ERROR: {str(e)[:60]}")

        # Compute aggregates
        score.accuracy = score.correct / score.total_questions if score.total_questions else 0
        score.avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        score.avg_latency_s = sum(latencies) / len(latencies) if latencies else 0

        for t, total in type_total.items():
            score.per_type_scores[t] = type_correct.get(t, 0) / total

        report.benchmarks.append(score)

        print(f"\n  {bench_name}: {score.accuracy:.1%} accuracy, "
              f"F1={score.avg_f1:.3f}, {score.errors} errors")

    report.runtime_s = time.perf_counter() - start_total
    report.compute_overall()
    report.print_report()
    report.save(output_path)

    return report


def _load_dataset(name: str, max_samples: int | None = None) -> Any:
    """Load a benchmark dataset by name."""
    if name == "financebench":
        from rnsr.benchmarks.finance_bench import FinanceBenchLoader
        return FinanceBenchLoader.load(max_samples=max_samples)
    elif name == "multihiertt":
        from rnsr.benchmarks.multihiertt_bench import MultiHierttLoader
        return MultiHierttLoader.load(max_samples=max_samples)
    elif name == "tatqa":
        from rnsr.benchmarks.tatqa_bench import TATQALoader
        return TATQALoader.load(max_samples=max_samples)
    elif name == "qasper":
        from rnsr.benchmarks.qasper_bench import QASPERLoader
        return QASPERLoader.load(max_samples=max_samples)
    elif name == "docvqa":
        from rnsr.benchmarks.docvqa_bench import DocVQALoader
        return DocVQALoader.load(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNSR Generalization Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (10 samples per benchmark)
  python run_all_benchmarks.py

  # Run specific benchmarks
  python run_all_benchmarks.py --benchmarks tatqa qasper

  # Full run with all samples
  python run_all_benchmarks.py --full

  # Use OpenAI instead of Gemini
  python run_all_benchmarks.py --provider openai --model gpt-4o
        """,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["financebench", "multihiertt", "tatqa", "qasper", "docvqa"],
        help="Benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Max samples per benchmark (default: 10)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all samples (overrides --max-samples)",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "anthropic"],
        help="LLM provider (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="LLM model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/generalization_report.json",
        help="Output path for the report JSON",
    )

    args = parser.parse_args()

    max_samples = None if args.full else args.max_samples

    run_benchmark_suite(
        benchmarks=args.benchmarks,
        max_samples=max_samples,
        llm_provider=args.provider,
        llm_model=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
