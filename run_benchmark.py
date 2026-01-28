#!/usr/bin/env python3
"""
Run RNSR Comprehensive Benchmark
Compares all navigator types: standard, RLM, vision, hybrid
"""

from pathlib import Path
from rnsr.benchmarks import run_comprehensive_benchmark

# Use the included sample PDF
pdf_paths = [
    Path("rnsr/benchmarks/data/financebench/3c4c9e28_3M_2018_10K.pdf"),
]

print("=" * 70)
print("RNSR Comprehensive Benchmark")
print("=" * 70)
print(f"PDF: {pdf_paths[0]}")
print("Methods: standard, rlm, vision, hybrid")
print("Benchmark: financebench (5 test cases)")
print("=" * 70)
print()

# Run FinanceBench-style tests across all methods
results = run_comprehensive_benchmark(
    pdf_paths=pdf_paths,
    benchmark_type="financebench",
    methods=["standard", "rlm", "vision", "hybrid"],
    output_path="benchmark_results/report.json",
)

# Print formatted results table
results.print_report()

print("\nResults saved to: benchmark_results/report.json")
