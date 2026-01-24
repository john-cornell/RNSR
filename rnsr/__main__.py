"""
RNSR CLI - Command Line Interface

Usage:
    python -m rnsr ingest document.pdf
    python -m rnsr query "What are the payment terms?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

logger = structlog.get_logger(__name__)


def cmd_ingest(args):
    """Ingest a PDF document."""
    from rnsr.ingestion import ingest_document
    
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Ingesting: {pdf_path}")
    result = ingest_document(pdf_path)
    
    print(f"\n✓ Ingestion complete!")
    print(f"  Tier used: {result.tier_used} ({result.method})")
    print(f"  Total nodes: {result.tree.total_nodes}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result.tree.model_dump(), f, indent=2)
        print(f"\nTree saved to: {output_path}")
    
    return result


def cmd_index(args):
    """Build skeleton index from ingested document."""
    from rnsr.indexing import SQLiteKVStore, build_skeleton_index
    from rnsr.ingestion import ingest_document
    
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Ingest first
    print(f"Ingesting: {pdf_path}")
    result = ingest_document(pdf_path)
    
    # Build index
    db_path = args.db or f"{pdf_path.stem}_index.db"
    kv_store = SQLiteKVStore(db_path)
    skeleton, _ = build_skeleton_index(result.tree, kv_store)
    
    print(f"\n✓ Index built!")
    print(f"  Skeleton nodes: {len(skeleton)}")
    print(f"  KV entries: {kv_store.count()}")
    print(f"  Database: {db_path}")
    
    return skeleton, kv_store


def cmd_query(args):
    """Query a document."""
    from rnsr.agent import run_navigator
    from rnsr.indexing import SQLiteKVStore, build_skeleton_index
    from rnsr.ingestion import ingest_document
    
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Ingest
    print(f"Ingesting: {pdf_path}")
    result = ingest_document(pdf_path)
    
    # Build index
    skeleton, kv_store = build_skeleton_index(result.tree)
    
    # Run query
    print(f"\nQuery: {args.query}")
    print("-" * 40)
    
    answer = run_navigator(
        question=args.query,
        skeleton=skeleton,
        kv_store=kv_store,
        max_iterations=args.max_iter,
    )
    
    print(f"\nAnswer:")
    print(answer["answer"])
    print(f"\nConfidence: {answer['confidence']:.2f}")
    print(f"Nodes visited: {len(answer['nodes_visited'])}")
    print(f"Variables used: {len(answer['variables_used'])}")
    
    if args.trace:
        print(f"\nTrace:")
        for entry in answer["trace"]:
            print(f"  [{entry['node_type']}] {entry['action']}")


def cmd_benchmark(args):
    """Run benchmarks on the RNSR system."""
    from .benchmarks import BenchmarkRunner, BenchmarkConfig
    
    # Check files are provided
    if not args.config and not args.files:
        print("❌ Error: Provide --files or --config for benchmarking")
        return
    
    # Load config if provided
    if args.config:
        config = BenchmarkConfig.from_json(args.config)
    else:
        config = BenchmarkConfig(
            pdf_paths=[Path(f) for f in (args.files or [])],
            iterations=args.iterations,
            compute_quality=args.quality or args.all,
        )
    
    print("=" * 60)
    print("RNSR Benchmark Suite")
    print("=" * 60)
    print(f"Files: {len(config.pdf_paths)}")
    print(f"Iterations: {config.iterations}")
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    report = runner.run()
    
    # Print summary
    report.print_summary()
    
    # Save results
    output_dir = args.output or "benchmark_results"
    output_path = Path(output_dir)
    report_file = output_path / f"benchmark_report_{report.timestamp.replace(':', '-')}.json"
    report.to_json(report_file)
    
    print(f"\n📄 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RNSR - Recursive Neural-Symbolic Retriever"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF document")
    ingest_parser.add_argument("file", help="Path to PDF file")
    ingest_parser.add_argument("-o", "--output", help="Output JSON file for tree")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build skeleton index")
    index_parser.add_argument("file", help="Path to PDF file")
    index_parser.add_argument("--db", help="SQLite database path")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a document")
    query_parser.add_argument("file", help="Path to PDF file")
    query_parser.add_argument("query", help="Question to ask")
    query_parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    query_parser.add_argument("--trace", action="store_true", help="Show trace")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument(
        "--config", "-c",
        help="Path to benchmark config JSON file"
    )
    bench_parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="PDF files to benchmark"
    )
    bench_parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=3,
        help="Number of iterations per benchmark (default: 3)"
    )
    bench_parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    bench_parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Run performance benchmarks"
    )
    bench_parser.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Run quality benchmarks"
    )
    bench_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all benchmarks"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
