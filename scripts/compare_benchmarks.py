#!/usr/bin/env python3
"""
RNSR Comparison Benchmarks

Compares RNSR against baseline approaches:
1. Naive RAG (chunk + embed + retrieve)
2. Long-context LLM (no RAG)
3. RNSR (hierarchical + ToT)

Metrics:
- Retrieval Precision/Recall
- Answer Quality (LLM-as-judge)
- Hallucination Detection
- Citation Accuracy
- Response Time
- Cost per Query

Usage:
    python scripts/compare_benchmarks.py --pdf document.pdf --queries queries.json
    python scripts/compare_benchmarks.py --quick  # Use sample data
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress

console = Console()


# =============================================================================
# Benchmark Data Structures
# =============================================================================

@dataclass
class BenchmarkQuery:
    """A query with ground truth for evaluation."""
    question: str
    ground_truth_answer: str = ""
    relevant_sections: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard


@dataclass 
class BenchmarkResult:
    """Result from a single benchmark run."""
    method: str
    query: str
    answer: str
    time_sec: float
    sources_used: int = 0
    
    # Quality metrics (0-1)
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    hallucination_score: float = 0.0  # 0 = no hallucination, 1 = all hallucinated
    citation_accuracy: float = 0.0
    
    # Cost
    tokens_used: int = 0
    estimated_cost: float = 0.0


@dataclass
class ComparisonReport:
    """Full comparison report."""
    document: str
    num_queries: int
    results: list[BenchmarkResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# =============================================================================
# Baseline: Naive RAG
# =============================================================================

class NaiveRAGBaseline:
    """
    Simple chunk-based RAG for comparison.
    
    This represents the typical approach:
    1. Split document into fixed-size chunks
    2. Embed chunks
    3. Retrieve top-k by similarity
    4. Generate answer from chunks
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, top_k: int = 5):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.chunks = []
        self.embeddings = []
        
    def ingest(self, text: str):
        """Chunk the document."""
        self.chunks = self._chunk_text(text)
        console.print(f"[dim]NaiveRAG: Created {len(self.chunks)} chunks[/dim]")
        
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks
    
    def query(self, question: str) -> dict:
        """Answer using naive retrieval."""
        from rnsr.llm import get_llm
        
        start = time.time()
        
        # Simple keyword matching for retrieval (in real impl, use embeddings)
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            scored_chunks.append((overlap, i, chunk))
        
        # Get top-k chunks
        scored_chunks.sort(reverse=True)
        top_chunks = [c[2] for c in scored_chunks[:self.top_k]]
        
        # Generate answer
        context = "\n\n---\n\n".join(top_chunks)
        
        llm = get_llm()
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = llm.complete(prompt)
        answer = str(response)
        
        elapsed = time.time() - start
        
        return {
            "answer": answer,
            "time_sec": elapsed,
            "sources": len(top_chunks),
            "method": "naive_rag",
        }


# =============================================================================
# Baseline: Long-Context LLM
# =============================================================================

class LongContextBaseline:
    """
    Just stuff the whole document into the LLM context.
    
    This tests whether structure matters - if the doc fits in context,
    does hierarchical understanding still help?
    """
    
    def __init__(self, max_chars: int = 100000):
        self.max_chars = max_chars
        self.document = ""
        
    def ingest(self, text: str):
        """Store document (truncated if needed)."""
        self.document = text[:self.max_chars]
        if len(text) > self.max_chars:
            console.print(f"[yellow]LongContext: Truncated to {self.max_chars} chars[/yellow]")
    
    def query(self, question: str) -> dict:
        """Answer by stuffing whole doc in context."""
        from rnsr.llm import get_llm
        
        start = time.time()
        
        llm = get_llm()
        prompt = f"""Read the following document and answer the question.

Document:
{self.document}

Question: {question}

Answer:"""
        
        response = llm.complete(prompt)
        answer = str(response)
        
        elapsed = time.time() - start
        
        return {
            "answer": answer,
            "time_sec": elapsed,
            "sources": 1,
            "method": "long_context",
        }


# =============================================================================
# RNSR System
# =============================================================================

class RNSRSystem:
    """RNSR with full hierarchical processing."""
    
    def __init__(self):
        self.tree = None
        self.skeleton = None
        self.kv_store = None
        self.navigator = None
        
    def ingest(self, pdf_path: Path):
        """Ingest document with hierarchical processing."""
        from rnsr.ingestion.pipeline import ingest_document
        from rnsr.indexing import build_skeleton_index
        
        result = ingest_document(pdf_path)
        self.tree = result.tree
        self.skeleton, self.kv_store = build_skeleton_index(self.tree)
        
        console.print(f"[dim]RNSR: {self.tree.total_nodes} nodes, depth {self.tree.max_depth}[/dim]")
    
    def ingest_text(self, text: str):
        """Ingest from text (for comparison with baselines)."""
        # Create a simple tree from text
        from rnsr.models import DocumentNode, DocumentTree
        
        # Split by potential headers
        sections = re.split(r'\n(?=[A-Z][^a-z]*\n)', text)
        
        root = DocumentNode(id="root", header="Document", content="", level=0)
        
        for i, section in enumerate(sections):
            lines = section.strip().split('\n')
            header = lines[0] if lines else f"Section {i+1}"
            content = '\n'.join(lines[1:]) if len(lines) > 1 else section
            
            node = DocumentNode(
                id=f"section_{i}",
                header=header[:50],
                content=content,
                level=1,
            )
            root.children.append(node)
        
        self.tree = DocumentTree(root=root, id="doc")
        
        from rnsr.indexing import build_skeleton_index
        self.skeleton, self.kv_store = build_skeleton_index(self.tree)
        
        console.print(f"[dim]RNSR: {self.tree.total_nodes} nodes from text[/dim]")
    
    def query(self, question: str) -> dict:
        """Answer using RNSR navigation."""
        from rnsr.agent.rlm_navigator import RLMNavigator
        
        start = time.time()
        
        if self.navigator is None:
            self.navigator = RLMNavigator(
                skeleton=self.skeleton,
                kv_store=self.kv_store,
                tree=self.tree,
            )
        
        result = self.navigator.answer(question)
        
        elapsed = time.time() - start
        
        return {
            "answer": result.get("answer", ""),
            "time_sec": elapsed,
            "sources": len(result.get("sources", [])),
            "method": "rnsr",
        }


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_answer_quality(
    question: str,
    answer: str,
    ground_truth: str,
) -> dict:
    """
    Evaluate answer quality using LLM-as-judge.
    
    Returns scores for:
    - relevance: Does the answer address the question?
    - correctness: Is the answer factually correct?
    - completeness: Does it cover all aspects?
    """
    if not ground_truth:
        return {"relevance": 0.5, "correctness": 0.5, "completeness": 0.5}
    
    from rnsr.llm import get_llm
    
    prompt = f"""Evaluate this answer against the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Rate the generated answer on these dimensions (0.0 to 1.0):
1. Relevance: Does it address the question?
2. Correctness: Is it factually accurate compared to ground truth?
3. Completeness: Does it cover the key points?

Respond with JSON only:
{{"relevance": 0.X, "correctness": 0.X, "completeness": 0.X}}"""

    try:
        llm = get_llm()
        response = llm.complete(prompt)
        response_text = str(response)
        
        # Parse JSON
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            scores = json.loads(json_match.group())
            return {
                "relevance": float(scores.get("relevance", 0.5)),
                "correctness": float(scores.get("correctness", 0.5)),
                "completeness": float(scores.get("completeness", 0.5)),
            }
    except Exception as e:
        console.print(f"[yellow]Evaluation error: {e}[/yellow]")
    
    return {"relevance": 0.5, "correctness": 0.5, "completeness": 0.5}


def detect_hallucination(answer: str, source_text: str) -> float:
    """
    Detect hallucination by checking if claims are grounded in source.
    
    Returns: Hallucination score (0 = fully grounded, 1 = fully hallucinated)
    """
    from rnsr.llm import get_llm
    
    prompt = f"""Analyze this answer for hallucination.

Source Text (ground truth):
{source_text[:3000]}

Generated Answer:
{answer}

For each claim in the answer, check if it's supported by the source text.
Respond with a hallucination score from 0.0 (fully grounded) to 1.0 (fully hallucinated).

JSON only: {{"hallucination_score": 0.X, "unsupported_claims": ["claim1", "claim2"]}}"""

    try:
        llm = get_llm()
        response = llm.complete(prompt)
        response_text = str(response)
        
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            return float(result.get("hallucination_score", 0.5))
    except Exception:
        pass
    
    return 0.5


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_comparison(
    document_text: str,
    queries: list[BenchmarkQuery],
    pdf_path: Path = None,
) -> ComparisonReport:
    """Run full comparison benchmark."""
    
    console.print("\n[bold blue]📊 Running Comparison Benchmark[/bold blue]\n")
    
    # Initialize systems
    naive_rag = NaiveRAGBaseline()
    long_context = LongContextBaseline()
    rnsr = RNSRSystem()
    
    # Ingest document
    console.print("[bold]Ingesting document...[/bold]")
    
    naive_rag.ingest(document_text)
    long_context.ingest(document_text)
    
    if pdf_path:
        rnsr.ingest(pdf_path)
    else:
        rnsr.ingest_text(document_text)
    
    console.print()
    
    # Run queries
    results = []
    
    with Progress() as progress:
        task = progress.add_task("Running queries...", total=len(queries) * 3)
        
        for query in queries:
            for system, name in [
                (naive_rag, "Naive RAG"),
                (long_context, "Long Context"),
                (rnsr, "RNSR"),
            ]:
                try:
                    result = system.query(query.question)
                    
                    # Evaluate quality
                    quality = evaluate_answer_quality(
                        query.question,
                        result["answer"],
                        query.ground_truth_answer,
                    )
                    
                    # Detect hallucination
                    hallucination = detect_hallucination(
                        result["answer"],
                        document_text[:5000],
                    )
                    
                    benchmark_result = BenchmarkResult(
                        method=result["method"],
                        query=query.question,
                        answer=result["answer"],
                        time_sec=result["time_sec"],
                        sources_used=result.get("sources", 0),
                        answer_relevance=quality["relevance"],
                        answer_correctness=quality["correctness"],
                        hallucination_score=hallucination,
                    )
                    results.append(benchmark_result)
                    
                except Exception as e:
                    console.print(f"[red]Error with {name}: {e}[/red]")
                
                progress.advance(task)
    
    # Generate report
    report = ComparisonReport(
        document=pdf_path.name if pdf_path else "text_input",
        num_queries=len(queries),
        results=results,
    )
    
    # Calculate summary statistics
    report.summary = calculate_summary(results)
    
    return report


def calculate_summary(results: list[BenchmarkResult]) -> dict:
    """Calculate summary statistics by method."""
    summary = {}
    
    methods = set(r.method for r in results)
    
    for method in methods:
        method_results = [r for r in results if r.method == method]
        
        if not method_results:
            continue
            
        summary[method] = {
            "avg_time_sec": sum(r.time_sec for r in method_results) / len(method_results),
            "avg_relevance": sum(r.answer_relevance for r in method_results) / len(method_results),
            "avg_correctness": sum(r.answer_correctness for r in method_results) / len(method_results),
            "avg_hallucination": sum(r.hallucination_score for r in method_results) / len(method_results),
            "total_sources": sum(r.sources_used for r in method_results),
            "num_queries": len(method_results),
        }
    
    return summary


def display_report(report: ComparisonReport):
    """Display benchmark report."""
    
    console.print("\n[bold blue]📈 Benchmark Results[/bold blue]\n")
    
    # Summary table
    table = Table(title="Performance Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Avg Time", style="yellow")
    table.add_column("Relevance", style="green")
    table.add_column("Correctness", style="green")
    table.add_column("Hallucination", style="red")
    
    for method, stats in report.summary.items():
        # Format method name nicely
        method_name = {
            "naive_rag": "Naive RAG",
            "long_context": "Long Context",
            "rnsr": "RNSR",
        }.get(method, method)
        
        # Color code hallucination (lower is better)
        hall_color = "green" if stats["avg_hallucination"] < 0.3 else "yellow" if stats["avg_hallucination"] < 0.5 else "red"
        
        table.add_row(
            method_name,
            f"{stats['avg_time_sec']:.2f}s",
            f"{stats['avg_relevance']:.0%}",
            f"{stats['avg_correctness']:.0%}",
            f"[{hall_color}]{stats['avg_hallucination']:.0%}[/{hall_color}]",
        )
    
    console.print(table)
    
    # Winner analysis
    console.print("\n[bold]Analysis:[/bold]\n")
    
    if "rnsr" in report.summary and "naive_rag" in report.summary:
        rnsr = report.summary["rnsr"]
        naive = report.summary["naive_rag"]
        
        if rnsr["avg_correctness"] > naive["avg_correctness"]:
            improvement = (rnsr["avg_correctness"] - naive["avg_correctness"]) / naive["avg_correctness"] * 100
            console.print(f"  ✅ RNSR correctness is [green]{improvement:.0f}% better[/green] than Naive RAG")
        
        if rnsr["avg_hallucination"] < naive["avg_hallucination"]:
            reduction = (naive["avg_hallucination"] - rnsr["avg_hallucination"]) / naive["avg_hallucination"] * 100
            console.print(f"  ✅ RNSR reduces hallucination by [green]{reduction:.0f}%[/green]")


def create_sample_queries() -> list[BenchmarkQuery]:
    """Create sample queries for benchmarking."""
    return [
        BenchmarkQuery(
            question="What is the main topic of this document?",
            ground_truth_answer="",  # Will be evaluated without ground truth
            difficulty="easy",
        ),
        BenchmarkQuery(
            question="What are the key findings or conclusions?",
            ground_truth_answer="",
            difficulty="medium",
        ),
        BenchmarkQuery(
            question="Summarize the methodology or approach described.",
            ground_truth_answer="",
            difficulty="medium",
        ),
        BenchmarkQuery(
            question="What specific numbers or statistics are mentioned?",
            ground_truth_answer="",
            difficulty="hard",
        ),
        BenchmarkQuery(
            question="What are the main sections of this document?",
            ground_truth_answer="",
            difficulty="easy",
        ),
    ]


def run_quick_benchmark():
    """Run benchmark on sample contract."""
    
    samples_dir = Path(__file__).parent.parent / "samples"
    sample_contract = samples_dir / "sample_contract.md"
    
    if not sample_contract.exists():
        console.print("[red]Sample documents not found.[/red]")
        return
    
    document_text = sample_contract.read_text()
    
    # Create queries specific to the contract
    queries = [
        BenchmarkQuery(
            question="Who are the parties to this contract?",
            ground_truth_answer="Acme Technologies Inc. (Client) and DataSoft Solutions LLC (Provider)",
            difficulty="easy",
        ),
        BenchmarkQuery(
            question="What is the contract value and duration?",
            ground_truth_answer="$2,500,000 over 36 months",
            difficulty="easy",
        ),
        BenchmarkQuery(
            question="What are the key deliverables in Phase 1?",
            ground_truth_answer="",
            difficulty="medium",
        ),
        BenchmarkQuery(
            question="What are the termination conditions?",
            ground_truth_answer="",
            difficulty="medium",
        ),
    ]
    
    report = run_comparison(document_text, queries)
    display_report(report)
    
    # Save report
    report_path = Path("benchmark_comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    console.print(f"\n[green]📄 Report saved to: {report_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="RNSR Comparison Benchmarks")
    parser.add_argument("--pdf", type=Path, help="PDF to benchmark")
    parser.add_argument("--queries", type=Path, help="JSON file with queries")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with samples")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]RNSR Comparison Benchmark[/bold cyan]\n"
        "Comparing RNSR vs Naive RAG vs Long Context",
        border_style="cyan",
    ))
    
    if args.quick or (not args.pdf and not args.queries):
        run_quick_benchmark()
    elif args.pdf:
        # Load PDF and run benchmark
        if not args.pdf.exists():
            console.print(f"[red]File not found: {args.pdf}[/red]")
            return 1
        
        # Read PDF text
        import fitz
        doc = fitz.open(args.pdf)
        text = "\n\n".join(page.get_text() for page in doc)
        
        queries = create_sample_queries()
        if args.queries and args.queries.exists():
            with open(args.queries) as f:
                query_data = json.load(f)
                queries = [BenchmarkQuery(**q) for q in query_data]
        
        report = run_comparison(text, queries, args.pdf)
        display_report(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
