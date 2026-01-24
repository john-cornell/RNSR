"""
Quality Benchmarks

Measures retrieval and answer quality:
- Precision/Recall for retrieved nodes
- Answer relevance (semantic similarity)
- Context coverage
- Faithfulness (answer grounded in retrieved content)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval evaluation."""
    
    question: str
    expected_nodes: list[str]  # Ground truth node IDs
    retrieved_nodes: list[str]  # Actually retrieved node IDs
    
    @property
    def precision(self) -> float:
        """Fraction of retrieved nodes that are relevant."""
        if not self.retrieved_nodes:
            return 0.0
        relevant = set(self.expected_nodes) & set(self.retrieved_nodes)
        return len(relevant) / len(self.retrieved_nodes)
    
    @property
    def recall(self) -> float:
        """Fraction of relevant nodes that were retrieved."""
        if not self.expected_nodes:
            return 1.0  # No expected = perfect recall
        relevant = set(self.expected_nodes) & set(self.retrieved_nodes)
        return len(relevant) / len(self.expected_nodes)
    
    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @property
    def hit_rate(self) -> float:
        """Whether at least one relevant node was retrieved."""
        return 1.0 if set(self.expected_nodes) & set(self.retrieved_nodes) else 0.0


@dataclass
class AnswerMetrics:
    """Metrics for answer quality."""
    
    question: str
    expected_answer: str
    generated_answer: str
    semantic_similarity: float = 0.0  # Cosine similarity of embeddings
    rouge_l: float = 0.0  # ROUGE-L score
    contains_expected: bool = False  # Whether key info is present
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "semantic_similarity": self.semantic_similarity,
            "rouge_l": self.rouge_l,
            "contains_expected": self.contains_expected,
        }


@dataclass
class QualityMetrics:
    """Aggregated quality metrics across all test cases."""
    
    retrieval_metrics: list[RetrievalMetrics] = field(default_factory=list)
    answer_metrics: list[AnswerMetrics] = field(default_factory=list)
    
    @property
    def avg_precision(self) -> float:
        if not self.retrieval_metrics:
            return 0.0
        return sum(m.precision for m in self.retrieval_metrics) / len(self.retrieval_metrics)
    
    @property
    def avg_recall(self) -> float:
        if not self.retrieval_metrics:
            return 0.0
        return sum(m.recall for m in self.retrieval_metrics) / len(self.retrieval_metrics)
    
    @property
    def avg_f1(self) -> float:
        if not self.retrieval_metrics:
            return 0.0
        return sum(m.f1 for m in self.retrieval_metrics) / len(self.retrieval_metrics)
    
    @property
    def hit_rate(self) -> float:
        if not self.retrieval_metrics:
            return 0.0
        return sum(m.hit_rate for m in self.retrieval_metrics) / len(self.retrieval_metrics)
    
    @property
    def avg_semantic_similarity(self) -> float:
        if not self.answer_metrics:
            return 0.0
        return sum(m.semantic_similarity for m in self.answer_metrics) / len(self.answer_metrics)
    
    def summary(self) -> dict[str, float]:
        return {
            "precision": self.avg_precision,
            "recall": self.avg_recall,
            "f1": self.avg_f1,
            "hit_rate": self.hit_rate,
            "semantic_similarity": self.avg_semantic_similarity,
            "test_cases": len(self.retrieval_metrics),
        }
    
    def __str__(self) -> str:
        s = self.summary()
        return (
            f"=== Quality Metrics ===\n"
            f"Precision: {s['precision']:.3f}\n"
            f"Recall: {s['recall']:.3f}\n"
            f"F1 Score: {s['f1']:.3f}\n"
            f"Hit Rate: {s['hit_rate']:.3f}\n"
            f"Semantic Similarity: {s['semantic_similarity']:.3f}\n"
            f"Test Cases: {s['test_cases']}"
        )


@dataclass
class TestCase:
    """A single test case for quality evaluation."""
    
    question: str
    expected_nodes: list[str] = field(default_factory=list)
    expected_answer: str = ""
    keywords: list[str] = field(default_factory=list)  # Keywords that should appear


@dataclass 
class QualityBenchmark:
    """Quality benchmark configuration and results."""
    
    test_cases: list[TestCase] = field(default_factory=list)
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    
    @classmethod
    def from_json(cls, path: Path | str) -> "QualityBenchmark":
        """Load test cases from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        test_cases = [
            TestCase(
                question=tc["question"],
                expected_nodes=tc.get("expected_nodes", []),
                expected_answer=tc.get("expected_answer", ""),
                keywords=tc.get("keywords", []),
            )
            for tc in data.get("test_cases", [])
        ]
        
        return cls(test_cases=test_cases)
    
    def to_json(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        data = {
            "test_cases": [
                {
                    "question": tc.question,
                    "expected_nodes": tc.expected_nodes,
                    "expected_answer": tc.expected_answer,
                    "keywords": tc.keywords,
                }
                for tc in self.test_cases
            ],
            "metrics": self.metrics.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using embeddings.
    
    Returns cosine similarity in range [0, 1].
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([text1, text2])
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    except ImportError:
        logger.warning("sentence-transformers not installed, returning 0")
        return 0.0


def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-L score (longest common subsequence).
    
    Returns F1-based ROUGE-L score.
    """
    def lcs_length(x: str, y: str) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    # Tokenize (simple word-level)
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    lcs = lcs_length(" ".join(ref_tokens), " ".join(cand_tokens))
    
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def evaluate_retrieval(
    skeleton: dict,
    kv_store: Any,
    test_cases: list[TestCase],
    compute_answer_quality: bool = True,
) -> QualityMetrics:
    """
    Evaluate retrieval quality against test cases.
    
    Args:
        skeleton: Skeleton index
        kv_store: KV store with content
        test_cases: List of test cases with ground truth
        compute_answer_quality: Whether to compute answer metrics (slower)
        
    Returns:
        QualityMetrics with all evaluation results
    """
    from rnsr.agent import run_navigator
    
    metrics = QualityMetrics()
    
    for i, tc in enumerate(test_cases):
        logger.info("evaluating_test_case", index=i + 1, total=len(test_cases))
        
        # Run the navigator
        result = run_navigator(
            question=tc.question,
            skeleton=skeleton,
            kv_store=kv_store,
            max_iterations=20,
        )
        
        # Extract retrieved nodes
        retrieved_nodes = result.get("nodes_visited", [])
        
        # Retrieval metrics
        retrieval = RetrievalMetrics(
            question=tc.question,
            expected_nodes=tc.expected_nodes,
            retrieved_nodes=retrieved_nodes,
        )
        metrics.retrieval_metrics.append(retrieval)
        
        # Answer quality metrics
        if compute_answer_quality and tc.expected_answer:
            generated = result.get("answer", "")
            
            answer = AnswerMetrics(
                question=tc.question,
                expected_answer=tc.expected_answer,
                generated_answer=generated,
                semantic_similarity=compute_semantic_similarity(
                    tc.expected_answer, generated
                ),
                rouge_l=compute_rouge_l(tc.expected_answer, generated),
                contains_expected=all(
                    kw.lower() in generated.lower() 
                    for kw in tc.keywords
                ) if tc.keywords else True,
            )
            metrics.answer_metrics.append(answer)
    
    return metrics
