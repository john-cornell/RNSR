"""
Standard RAG Benchmarks for RNSR Evaluation

This module provides integration with established RAG and retrieval benchmarks
to validate RNSR's claims of improved document parsing and traversal.

Key Benchmarks:
1. RAGAS - Standard RAG evaluation metrics (faithfulness, relevance, etc.)
2. BEIR - Information retrieval benchmark (17+ datasets)
3. HotpotQA - Multi-hop question answering
4. MuSiQue - Multi-hop questions via single-hop composition

These benchmarks help demonstrate RNSR's advantages:
- Hierarchical tree traversal vs flat chunk retrieval
- Multi-hop reasoning capabilities
- Context preservation in complex documents
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Baseline RAG Systems for Comparison
# =============================================================================

@dataclass
class BaselineResult:
    """Result from a baseline RAG system."""
    
    answer: str
    retrieved_chunks: list[str]
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaselineRAG(ABC):
    """Abstract base class for baseline RAG implementations."""
    
    @abstractmethod
    def query(self, question: str, document_path: Path) -> BaselineResult:
        """Answer a question using the baseline method."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this baseline."""
        pass


class NaiveChunkRAG(BaselineRAG):
    """
    Naive chunking baseline - the standard RAG approach.
    
    Chunks document into fixed-size segments, embeds them,
    retrieves top-k by similarity, and generates answer.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
    
    def name(self) -> str:
        return f"naive_chunk_{self.chunk_size}"
    
    def query(self, question: str, document_path: Path) -> BaselineResult:
        """Query using naive chunking."""
        import fitz  # type: ignore[import-not-found]  # PyMuPDF
        
        start_total = time.perf_counter()
        
        # Extract text
        doc = fitz.open(document_path)
        full_text = ""
        for page in doc:
            text = page.get_text()
            if isinstance(text, str):
                full_text += text
        doc.close()
        
        # Naive chunking
        chunks = []
        for i in range(0, len(full_text), self.chunk_size - self.chunk_overlap):
            chunk = full_text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        # Embed and retrieve (simplified - would use actual embeddings)
        start_retrieval = time.perf_counter()
        
        # For now, use simple keyword matching as proxy
        # In production, use actual embeddings
        question_words = set(question.lower().split())
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(question_words & chunk_words) / max(len(question_words), 1)
            scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        retrieved = [c for _, c in scored_chunks[:self.top_k]]
        
        retrieval_time = time.perf_counter() - start_retrieval
        
        # Generate answer (placeholder - would use LLM)
        start_generation = time.perf_counter()
        context = "\n\n".join(retrieved)
        answer = f"[Baseline answer based on {len(retrieved)} chunks]"
        generation_time = time.perf_counter() - start_generation
        
        total_time = time.perf_counter() - start_total
        
        return BaselineResult(
            answer=answer,
            retrieved_chunks=retrieved,
            retrieval_time_s=retrieval_time,
            generation_time_s=generation_time,
            total_time_s=total_time,
            method=self.name(),
            metadata={
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
            }
        )


class SemanticChunkRAG(BaselineRAG):
    """
    Semantic chunking baseline - splits on semantic boundaries.
    
    Uses sentence embeddings to detect topic shifts and
    creates more coherent chunks than naive splitting.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        top_k: int = 5,
    ):
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
    
    def name(self) -> str:
        return "semantic_chunk"
    
    def query(self, question: str, document_path: Path) -> BaselineResult:
        """Query using semantic chunking."""
        # Placeholder implementation
        start_total = time.perf_counter()
        
        # Would implement semantic boundary detection here
        # For now, return placeholder result
        
        return BaselineResult(
            answer="[Semantic baseline placeholder]",
            retrieved_chunks=[],
            retrieval_time_s=0.0,
            generation_time_s=0.0,
            total_time_s=time.perf_counter() - start_total,
            method=self.name(),
        )


# =============================================================================
# Standard Benchmark Datasets
# =============================================================================

@dataclass
class BenchmarkQuestion:
    """A question from a standard benchmark."""
    
    id: str
    question: str
    answer: str
    supporting_facts: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    reasoning_type: str = "single-hop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """A standard benchmark dataset."""
    
    name: str
    description: str
    questions: list[BenchmarkQuestion]
    metrics: list[str]
    source_url: str
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def sample(self, n: int) -> list[BenchmarkQuestion]:
        """Get a random sample of questions."""
        import random
        return random.sample(self.questions, min(n, len(self.questions)))


class BenchmarkLoader:
    """Load standard benchmark datasets."""
    
    @staticmethod
    def load_hotpotqa(
        split: Literal["train", "dev_distractor", "dev_fullwiki"] = "dev_distractor",
        max_samples: int | None = None,
    ) -> BenchmarkDataset:
        """
        Load HotpotQA dataset for multi-hop QA evaluation.
        
        HotpotQA features:
        - Natural multi-hop questions
        - Strong supervision for supporting facts
        - Explainable reasoning chains
        
        Download: http://curtis.ml.cmu.edu/datasets/hotpot/
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
            
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            
            questions = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                questions.append(BenchmarkQuestion(
                    id=item["id"],
                    question=item["question"],
                    answer=item["answer"],
                    supporting_facts=item.get("supporting_facts", {}).get("title", []),
                    context=[
                        " ".join(sentences) 
                        for sentences in item.get("context", {}).get("sentences", [])
                    ],
                    reasoning_type="multi-hop",
                    metadata={
                        "type": item.get("type", "unknown"),
                        "level": item.get("level", "unknown"),
                    }
                ))
            
            return BenchmarkDataset(
                name="HotpotQA",
                description="Multi-hop question answering with supporting facts",
                questions=questions,
                metrics=["answer_em", "answer_f1", "support_em", "support_f1"],
                source_url="https://hotpotqa.github.io/",
            )
            
        except ImportError:
            logger.warning("datasets library not installed, returning empty dataset")
            return BenchmarkDataset(
                name="HotpotQA",
                description="Multi-hop QA (not loaded - install 'datasets')",
                questions=[],
                metrics=["answer_em", "answer_f1", "support_em", "support_f1"],
                source_url="https://hotpotqa.github.io/",
            )
    
    @staticmethod
    def load_musique(
        variant: Literal["ans", "full"] = "ans",
        max_samples: int | None = None,
    ) -> BenchmarkDataset:
        """
        Load MuSiQue dataset for compositional multi-hop QA.
        
        MuSiQue features:
        - Questions composed from single-hop questions
        - Harder disconnected reasoning required
        - 2-4 hop questions
        
        Download: https://github.com/StonyBrookNLP/musique
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
            
            dataset = load_dataset(
                "dgslibiern/musique_ans" if variant == "ans" else "dgslibiern/musique_full",
                split="validation"
            )
            
            questions = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                questions.append(BenchmarkQuestion(
                    id=item.get("id", str(i)),
                    question=item["question"],
                    answer=item.get("answer", ""),
                    supporting_facts=[],
                    context=item.get("paragraphs", []),
                    reasoning_type="multi-hop-compositional",
                    metadata={
                        "answerable": item.get("answerable", True),
                    }
                ))
            
            return BenchmarkDataset(
                name=f"MuSiQue-{variant.upper()}",
                description="Compositional multi-hop questions",
                questions=questions,
                metrics=["answer_f1", "support_f1"],
                source_url="https://github.com/StonyBrookNLP/musique",
            )
            
        except ImportError:
            logger.warning("datasets library not installed")
            return BenchmarkDataset(
                name=f"MuSiQue-{variant.upper()}",
                description="MuSiQue (not loaded - install 'datasets')",
                questions=[],
                metrics=["answer_f1", "support_f1"],
                source_url="https://github.com/StonyBrookNLP/musique",
            )
    
    @staticmethod
    def load_beir_dataset(
        dataset_name: str = "nfcorpus",
        max_samples: int | None = None,
    ) -> BenchmarkDataset:
        """
        Load a BEIR benchmark dataset for retrieval evaluation.
        
        Available datasets:
        - msmarco, trec-covid, nfcorpus, bioasq, nq, hotpotqa
        - fiqa, arguana, webis-touche2020, cqadupstack, quora
        - dbpedia-entity, scidocs, fever, climate-fever, scifact
        
        See: https://github.com/beir-cellar/beir
        """
        try:
            from beir import util  # type: ignore[import-not-found]
            from beir.datasets.data_loader import GenericDataLoader  # type: ignore[import-not-found]
            
            data_path = util.download_and_unzip(
                f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
                "benchmark_data"
            )
            
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
            
            questions = []
            for i, (qid, query) in enumerate(queries.items()):
                if max_samples and i >= max_samples:
                    break
                
                relevant_docs = qrels.get(qid, {})
                context = [corpus[doc_id]["text"] for doc_id in relevant_docs if doc_id in corpus]
                
                questions.append(BenchmarkQuestion(
                    id=qid,
                    question=query,
                    answer="",  # BEIR is retrieval-focused, not QA
                    context=context[:5],
                    reasoning_type="retrieval",
                    metadata={"relevance_scores": relevant_docs}
                ))
            
            return BenchmarkDataset(
                name=f"BEIR-{dataset_name}",
                description=f"BEIR retrieval benchmark: {dataset_name}",
                questions=questions,
                metrics=["ndcg@10", "map", "recall@100", "precision@10"],
                source_url="https://github.com/beir-cellar/beir",
            )
            
        except ImportError:
            logger.warning("beir library not installed")
            return BenchmarkDataset(
                name=f"BEIR-{dataset_name}",
                description=f"BEIR (not loaded - install 'beir')",
                questions=[],
                metrics=["ndcg@10", "map", "recall@100"],
                source_url="https://github.com/beir-cellar/beir",
            )

    @staticmethod
    def load_qasper(
        max_samples: int | None = None,
    ) -> BenchmarkDataset:
        """
        Load QASPER-style scientific paper QA.
        
        Note: Original QASPER dataset uses deprecated format.
        Using SciQ as a scientific reasoning alternative.
        
        For true QASPER testing, download from:
        https://allenai.org/data/qasper
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
            
            # Use SciQ as scientific QA alternative (QASPER is deprecated)
            dataset = load_dataset("allenai/sciq", split="validation")
            
            questions = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                # SciQ has question, correct_answer, support (context)
                support = item.get("support", "")
                question = item.get("question", "")
                answer = item.get("correct_answer", "")
                
                # Skip if no support context
                if not support:
                    continue
                
                questions.append(BenchmarkQuestion(
                    id=str(i),
                    question=question,
                    answer=answer,
                    supporting_facts=[],
                    context=[support],
                    reasoning_type="scientific",
                    metadata={
                        "distractor1": item.get("distractor1", ""),
                        "distractor2": item.get("distractor2", ""),
                        "distractor3": item.get("distractor3", ""),
                    }
                ))
            
            return BenchmarkDataset(
                name="SciQ",
                description="Scientific reasoning QA with supporting context",
                questions=questions,
                metrics=["answer_f1", "answer_em"],
                source_url="https://allenai.org/data/sciq",
            )
            
        except Exception as e:
            logger.warning("sciq_load_failed", error=str(e))
            return BenchmarkDataset(
                name="SciQ",
                description=f"SciQ (load failed: {str(e)[:50]})",
                questions=[],
                metrics=["answer_f1"],
                source_url="https://allenai.org/data/sciq",
            )
    
    @staticmethod
    def load_quality(
        max_samples: int | None = None,
    ) -> BenchmarkDataset:
        """
        Load QuALITY dataset for long document QA.
        
        QuALITY features (ideal for RNSR):
        - Long articles (2,000-8,000 words)
        - Multiple-choice questions
        - Requires reading entire document
        - Tests long-range comprehension
        
        Paper: Pang et al., NAACL 2022
        URL: https://github.com/nyu-mll/quality
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
            
            # Use emozilla/quality which is available on HuggingFace
            dataset = load_dataset("emozilla/quality", split="validation")
            
            questions = []
            for item in dataset:
                if max_samples and len(questions) >= max_samples:
                    break
                
                article = item.get("article", "")
                question = item.get("question", "")
                options = item.get("options", [])
                gold_label = item.get("answer", 0)
                is_hard = item.get("hard", False)
                
                # Format answer as the correct option
                answer = options[gold_label] if gold_label < len(options) else ""
                
                questions.append(BenchmarkQuestion(
                    id=str(len(questions)),
                    question=question,
                    answer=answer,
                    supporting_facts=[],
                    context=[article],  # Full article as context
                    reasoning_type="long-document",
                    metadata={
                        "options": options,
                        "gold_label": gold_label,
                        "is_hard": is_hard,
                        "article_length": len(article.split()),
                    }
                ))
            
            return BenchmarkDataset(
                name="QuALITY",
                description="Long document multiple-choice QA",
                questions=questions,
                metrics=["accuracy", "answer_em"],
                source_url="https://github.com/nyu-mll/quality",
            )
            
        except Exception as e:
            logger.warning("quality_load_failed", error=str(e))
            return BenchmarkDataset(
                name="QuALITY",
                description=f"QuALITY (load failed: {str(e)[:50]})",
                questions=[],
                metrics=["accuracy", "answer_em"],
                source_url="https://github.com/nyu-mll/quality",
            )

    @staticmethod
    def load_financebench(
        split: str = "train",
        max_samples: int | None = None,
    ) -> BenchmarkDataset:
        """
        Load FinanceBench dataset.
        
        FinanceBench features:
        - Financial QA over complex PDFs
        - Requires table/chart understanding
        - Document-level retrieval
        """
        try:
            from rnsr.benchmarks.finance_bench import FinanceBenchLoader
            return FinanceBenchLoader.load(split=split, max_samples=max_samples)
        except Exception as e:
            logger.error("Failed to load FinanceBench", error=str(e))
            return BenchmarkDataset(
                name="FinanceBench",
                description="Financial QA (Failed to load)",
                questions=[],
                metrics=[],
                source_url=""
            )

    
    @staticmethod
    def load_narrative_qa(
        max_samples: int | None = None,
    ) -> BenchmarkDataset:
        """
        Load NarrativeQA dataset for very long document QA.
        
        NarrativeQA features (stress test for RNSR):
        - Full books and movie scripts
        - Very long context (10k-100k+ words)
        - Tests extreme long-range comprehension
        
        Paper: Kočiský et al., TACL 2018
        URL: https://github.com/deepmind/narrativeqa
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
            
            dataset = load_dataset("narrativeqa", split="validation")
            
            questions = []
            for item in dataset:
                if max_samples and len(questions) >= max_samples:
                    break
                
                # NarrativeQA has summaries as proxy for full documents
                document = item.get("document", {})
                summary = document.get("summary", {}).get("text", "")
                
                question = item.get("question", {}).get("text", "")
                answers = item.get("answers", [])
                answer = answers[0].get("text", "") if answers else ""
                
                questions.append(BenchmarkQuestion(
                    id=item.get("document", {}).get("id", str(len(questions))),
                    question=question,
                    answer=answer,
                    supporting_facts=[],
                    context=[summary],  # Using summary as proxy
                    reasoning_type="narrative",
                    metadata={
                        "kind": document.get("kind", ""),
                        "all_answers": [a.get("text", "") for a in answers],
                    }
                ))
            
            return BenchmarkDataset(
                name="NarrativeQA",
                description="Very long document QA (books/scripts)",
                questions=questions,
                metrics=["answer_f1", "rouge_l"],
                source_url="https://github.com/deepmind/narrativeqa",
            )
            
        except ImportError:
            logger.warning("datasets library not installed")
            return BenchmarkDataset(
                name="NarrativeQA",
                description="NarrativeQA (not loaded - install 'datasets')",
                questions=[],
                metrics=["answer_f1", "rouge_l"],
                source_url="https://github.com/deepmind/narrativeqa",
            )


# =============================================================================
# RAGAS Metrics Integration
# =============================================================================

@dataclass
class RAGASMetrics:
    """Standard RAGAS evaluation metrics."""
    
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevancy: float = 0.0
    answer_correctness: float = 0.0
    
    def overall_score(self) -> float:
        """Compute weighted overall score."""
        weights = {
            "faithfulness": 0.2,
            "answer_relevancy": 0.2,
            "context_precision": 0.15,
            "context_recall": 0.15,
            "context_relevancy": 0.15,
            "answer_correctness": 0.15,
        }
        
        total = 0.0
        for metric, weight in weights.items():
            total += getattr(self, metric) * weight
        
        return total
    
    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "context_relevancy": self.context_relevancy,
            "answer_correctness": self.answer_correctness,
            "overall": self.overall_score(),
        }


class RAGASEvaluator:
    """
    Evaluate RAG systems using RAGAS metrics.
    
    RAGAS (Retrieval Augmented Generation Assessment) provides
    standard metrics for evaluating RAG pipelines:
    
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved contexts relevant?
    - Context Recall: Are all relevant contexts retrieved?
    
    See: https://github.com/explodinggradients/ragas
    """
    
    def __init__(
        self,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.5-flash",
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
    
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetrics:
        """
        Evaluate a single RAG response using RAGAS metrics.
        """
        try:
            from ragas import evaluate  # type: ignore[import-not-found]
            from ragas.metrics import (  # type: ignore[import-not-found]
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset  # type: ignore[import-not-found]
            
            # Prepare data
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)
            
            result = evaluate(dataset, metrics=metrics)
            
            return RAGASMetrics(
                faithfulness=result.get("faithfulness", 0.0),
                answer_relevancy=result.get("answer_relevancy", 0.0),
                context_precision=result.get("context_precision", 0.0),
                context_recall=result.get("context_recall", 0.0) if ground_truth else 0.0,
            )
            
        except ImportError:
            logger.warning("ragas library not installed, returning zero metrics")
            return RAGASMetrics()
    
    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> RAGASMetrics:
        """Evaluate a batch of responses and return aggregated metrics."""
        all_metrics = []
        
        for i in range(len(questions)):
            gt = ground_truths[i] if ground_truths else None
            metrics = self.evaluate(
                questions[i],
                answers[i],
                contexts[i],
                gt,
            )
            all_metrics.append(metrics)
        
        # Aggregate
        if not all_metrics:
            return RAGASMetrics()
        
        return RAGASMetrics(
            faithfulness=sum(m.faithfulness for m in all_metrics) / len(all_metrics),
            answer_relevancy=sum(m.answer_relevancy for m in all_metrics) / len(all_metrics),
            context_precision=sum(m.context_precision for m in all_metrics) / len(all_metrics),
            context_recall=sum(m.context_recall for m in all_metrics) / len(all_metrics),
            context_relevancy=sum(m.context_relevancy for m in all_metrics) / len(all_metrics),
            answer_correctness=sum(m.answer_correctness for m in all_metrics) / len(all_metrics),
        )


# =============================================================================
# LLM-as-Judge Evaluator (semantic correctness vs ground truth)
# =============================================================================

JUDGE_PROMPT = """You are evaluating whether a predicted answer is correct given a question and ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}

Does the predicted answer convey the same information as the ground truth? The predicted answer may be verbose, include source citations, or use different wording - focus on semantic equivalence. Ignore formatting and minor phrasing differences.

**Numeric and derived answers:** Treat numeric answers as correct when the **value** matches the ground truth even if units or format differ (e.g. 8325 thousand = 8.325 million = 8325000; "8325 thousand" vs "$8.325 million"). When the question asks for a derived value (average, total, sum, ratio), treat the prediction as correct if it states or clearly implies the same number, even if the wording differs.

Respond with ONLY valid JSON (no markdown, no extra text):
{{"verdict": "correct"|"partial"|"incorrect", "score": 1.0|0.5|0.0, "explanation": "brief reason"}}

Use: verdict "correct" and score 1.0 when the predicted answer clearly contains the same factual answer (including numerically equivalent values). Use "partial" and 0.5 when it is partly right. Use "incorrect" and 0.0 when it is wrong or does not address the question."""


def _normalize_numeric_for_judge(text: str) -> float | None:
    """
    Extract a single numeric value from a string for judge pre-check.
    Handles formats like "8325 thousand", "8.325 million", "16650", "$1.9 million".
    Returns None if no single clear number is found.
    """
    if not text or not text.strip():
        return None
    text = text.strip().lower()
    # Remove $ and normalize spaces; keep commas for digit grouping
    text_clean = text.replace("$", " ").replace(",", "")
    # Match number (optional decimals) with optional scale word immediately after
    pattern = r"(\d+(?:\.\d+)?)\s*(thousand|million|billion|k|m|b)?"
    matches = list(re.finditer(pattern, text_clean))
    if not matches:
        return None
    # Use the last match (often the final answer in a sentence)
    m = matches[-1]
    num_str = m.group(1)
    scale = m.group(2)
    try:
        val = float(num_str)
    except ValueError:
        return None
    if scale:
        if scale in ("thousand", "k"):
            val *= 1e3
        elif scale in ("million", "m"):
            val *= 1e6
        elif scale in ("billion", "b"):
            val *= 1e9
    return val


@dataclass
class LLMJudgeResult:
    """Result from LLM-as-judge evaluation."""

    verdict: Literal["correct", "partial", "incorrect"]
    score: float
    explanation: str
    raw_response: str = ""

    @property
    def is_correct(self) -> bool:
        return self.verdict == "correct"

    @property
    def is_partial(self) -> bool:
        return self.verdict == "partial"


class LLMJudgeEvaluator:
    """
    Evaluate predicted answers against ground truth using an LLM judge.

    Semantically compares RNSR's verbose, sourced answers to benchmark
    ground truth so that correct content is not penalized for formatting.
    """

    def __init__(
        self,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._llm: Any = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            from rnsr.llm import LLMProvider, get_llm

            kwargs: dict[str, Any] = {}
            if self.llm_provider is not None:
                provider = (
                    LLMProvider(self.llm_provider)
                    if isinstance(self.llm_provider, str)
                    else self.llm_provider
                )
                kwargs["provider"] = provider
            if self.llm_model is not None:
                kwargs["model"] = self.llm_model
            self._llm = get_llm(**kwargs)
        return self._llm

    def evaluate(
        self,
        question: str,
        predicted_answer: str,
        ground_truth: str,
        all_acceptable_answers: list[str] | None = None,
    ) -> LLMJudgeResult:
        """
        Judge whether the predicted answer is correct vs ground truth.

        Args:
            question: The question that was asked.
            predicted_answer: RNSR's (or model's) answer.
            ground_truth: Reference correct answer.
            all_acceptable_answers: Optional list of alternative correct answers.

        Returns:
            LLMJudgeResult with verdict, score (1.0/0.5/0.0), and explanation.
        """
        # Numeric pre-check: if both GT and prediction have the same numeric value, treat as correct
        gt_num = _normalize_numeric_for_judge(ground_truth)
        pred_num = _normalize_numeric_for_judge(predicted_answer)
        if gt_num is not None and pred_num is not None:
            if gt_num == 0 and pred_num == 0:
                return LLMJudgeResult(
                    verdict="correct",
                    score=1.0,
                    explanation="Both ground truth and prediction are zero (numeric pre-check).",
                    raw_response="",
                )
            if gt_num != 0:
                rel = abs(gt_num - pred_num) / abs(gt_num)
                if rel <= 0.02:
                    return LLMJudgeResult(
                        verdict="correct",
                        score=1.0,
                        explanation="Numeric value matches ground truth (numeric pre-check).",
                        raw_response="",
                    )

        gt_display = ground_truth
        if all_acceptable_answers:
            gt_display = " | ".join(all_acceptable_answers[:5])
            if len(all_acceptable_answers) > 5:
                gt_display += " (or others)"

        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=gt_display,
            prediction=predicted_answer[:4000],
        )

        try:
            llm = self._get_llm()
            response = llm.complete(prompt)
            raw = str(response).strip()
            # Strip markdown code block if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)
            verdict = data.get("verdict", "incorrect").lower()
            if verdict not in ("correct", "partial", "incorrect"):
                verdict = "incorrect"
            score = float(data.get("score", 0.0))
            if verdict == "correct":
                score = max(score, 1.0)
            elif verdict == "partial":
                score = max(0.0, min(1.0, score))
            else:
                score = 0.0
            explanation = str(data.get("explanation", ""))
            return LLMJudgeResult(
                verdict=verdict,
                score=score,
                explanation=explanation,
                raw_response=raw,
            )
        except Exception as e:
            logger.warning("llm_judge_failed", error=str(e))
            return LLMJudgeResult(
                verdict="incorrect",
                score=0.0,
                explanation=f"Judge failed: {e}",
                raw_response="",
            )


# =============================================================================
# Multi-Hop Reasoning Metrics (for HotpotQA/MuSiQue)
# =============================================================================

@dataclass  
class MultiHopMetrics:
    """Metrics for multi-hop reasoning evaluation."""
    
    answer_em: float = 0.0  # Exact match
    answer_f1: float = 0.0  # Token-level F1
    support_em: float = 0.0  # Supporting fact EM
    support_f1: float = 0.0  # Supporting fact F1
    joint_em: float = 0.0   # Joint answer + support EM
    joint_f1: float = 0.0   # Joint answer + support F1
    
    def to_dict(self) -> dict[str, float]:
        return {
            "answer_em": self.answer_em,
            "answer_f1": self.answer_f1,
            "support_em": self.support_em,
            "support_f1": self.support_f1,
            "joint_em": self.joint_em,
            "joint_f1": self.joint_f1,
        }


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    common = set(pred_tokens) & set(gold_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def evaluate_multihop(
    predictions: list[dict[str, Any]],
    ground_truths: list[BenchmarkQuestion],
) -> MultiHopMetrics:
    """
    Evaluate multi-hop QA predictions against ground truth.
    
    Args:
        predictions: List of {"answer": str, "supporting_facts": list[str]}
        ground_truths: List of BenchmarkQuestion with answers and supporting facts
    """
    answer_ems = []
    answer_f1s = []
    support_ems = []
    support_f1s = []
    
    for pred, gold in zip(predictions, ground_truths):
        # Answer metrics
        answer_ems.append(compute_em(pred.get("answer", ""), gold.answer))
        answer_f1s.append(compute_f1(pred.get("answer", ""), gold.answer))
        
        # Supporting facts metrics
        pred_facts = set(pred.get("supporting_facts", []))
        gold_facts = set(gold.supporting_facts)
        
        if gold_facts:
            support_em = float(pred_facts == gold_facts)
            
            common = pred_facts & gold_facts
            prec = len(common) / len(pred_facts) if pred_facts else 0
            rec = len(common) / len(gold_facts) if gold_facts else 0
            support_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            support_ems.append(support_em)
            support_f1s.append(support_f1)
    
    n = len(predictions)
    
    return MultiHopMetrics(
        answer_em=sum(answer_ems) / n if n else 0,
        answer_f1=sum(answer_f1s) / n if n else 0,
        support_em=sum(support_ems) / len(support_ems) if support_ems else 0,
        support_f1=sum(support_f1s) / len(support_f1s) if support_f1s else 0,
        joint_em=(sum(answer_ems) / n) * (sum(support_ems) / len(support_ems)) if n and support_ems else 0,
        joint_f1=(sum(answer_f1s) / n) * (sum(support_f1s) / len(support_f1s)) if n and support_f1s else 0,
    )


# =============================================================================
# RNSR vs Baseline Comparison
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing RNSR against a baseline."""
    
    dataset_name: str
    rnsr_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    baseline_name: str
    improvement: dict[str, float]  # RNSR - baseline for each metric
    relative_improvement: dict[str, float]  # (RNSR - baseline) / baseline
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"Comparison: RNSR vs {self.baseline_name}",
            f"Dataset: {self.dataset_name}",
            f"{'='*60}",
            "",
            f"{'Metric':<25} {'RNSR':>10} {'Baseline':>10} {'Δ':>10} {'%':>10}",
            "-" * 65,
        ]
        
        for metric in self.rnsr_metrics:
            rnsr_val = self.rnsr_metrics.get(metric, 0)
            base_val = self.baseline_metrics.get(metric, 0)
            delta = self.improvement.get(metric, 0)
            rel = self.relative_improvement.get(metric, 0) * 100
            
            lines.append(f"{metric:<25} {rnsr_val:>10.3f} {base_val:>10.3f} {delta:>+10.3f} {rel:>+9.1f}%")
        
        lines.append("=" * 65)
        return "\n".join(lines)


def compare_rnsr_vs_baseline(
    rnsr_results: dict[str, float],
    baseline_results: dict[str, float],
    dataset_name: str,
    baseline_name: str,
) -> ComparisonResult:
    """Compare RNSR results against a baseline."""
    improvement = {}
    relative_improvement = {}
    
    for metric in rnsr_results:
        rnsr_val = rnsr_results.get(metric, 0)
        base_val = baseline_results.get(metric, 0)
        
        improvement[metric] = rnsr_val - base_val
        if base_val > 0:
            relative_improvement[metric] = (rnsr_val - base_val) / base_val
        else:
            relative_improvement[metric] = 0.0
    
    return ComparisonResult(
        dataset_name=dataset_name,
        rnsr_metrics=rnsr_results,
        baseline_metrics=baseline_results,
        baseline_name=baseline_name,
        improvement=improvement,
        relative_improvement=relative_improvement,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Baselines
    "BaselineRAG",
    "BaselineResult",
    "NaiveChunkRAG",
    "SemanticChunkRAG",
    
    # Benchmarks
    "BenchmarkQuestion",
    "BenchmarkDataset",
    "BenchmarkLoader",
    
    # RAGAS
    "RAGASMetrics",
    "RAGASEvaluator",
    
    # Multi-hop
    "MultiHopMetrics",
    "evaluate_multihop",
    "compute_em",
    "compute_f1",
    
    # Comparison
    "ComparisonResult",
    "compare_rnsr_vs_baseline",
]
