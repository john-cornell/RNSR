"""
MultiHiertt Benchmark Loader for RNSR Evaluation

MultiHiertt is a benchmark for multi-step numerical reasoning over
hierarchical tables in financial documents. It requires:
- Parsing multiple related tables
- Performing arithmetic operations across table cells
- Understanding hierarchical table structure (merged cells, sub-headers)

Repository: https://github.com/psunlpgroup/MultiHiertt
Paper: "MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular
       and Textual Data" (ACL 2022)

Key metrics:
- Exact Match (EM): Whether the predicted answer matches exactly
- F1: Token-level overlap between prediction and ground truth
- Execution Accuracy: Whether the predicted program produces the correct result
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Optional

import structlog

from rnsr.benchmarks.standard_benchmarks import BenchmarkDataset, BenchmarkQuestion

logger = structlog.get_logger(__name__)

CACHE_DIR = Path("rnsr/benchmarks/data/multihiertt")


class MultiHierttLoader:
    """Loader for the MultiHiertt dataset."""

    @staticmethod
    def load(
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """
        Load the MultiHiertt dataset from HuggingFace.

        Args:
            split: Dataset split to load ('train', 'dev', 'test')
            max_samples: Max number of questions to load

        Returns:
            BenchmarkDataset containing MultiHiertt questions
        """
        try:
            from datasets import load_dataset  # type: ignore
            dataset = load_dataset("TheBigAiNerd/MultiHiertt", split=split)
        except Exception as e:
            logger.error("Failed to load MultiHiertt dataset", error=str(e))
            # Fallback: try loading from local JSON if available
            return MultiHierttLoader._load_from_local(split, max_samples)

        questions: list[BenchmarkQuestion] = []
        count = 0

        for item in dataset:
            if not isinstance(item, dict):
                continue
            if max_samples and count >= max_samples:
                break

            # MultiHiertt provides tables as HTML or structured data + text paragraphs
            table_data = item.get("table", "") or item.get("tables", "")
            paragraphs = item.get("paragraphs", "") or item.get("text", "")
            question_text = item.get("question", item.get("qa", {}).get("question", ""))
            answer = item.get("answer", item.get("qa", {}).get("answer", ""))
            program = item.get("program", item.get("qa", {}).get("program", ""))

            if not question_text:
                continue

            # Build context from tables + paragraphs
            contexts: list[str] = []
            if isinstance(table_data, str) and table_data:
                contexts.append(f"[TABLE]\n{table_data}")
            elif isinstance(table_data, list):
                for i, t in enumerate(table_data):
                    contexts.append(f"[TABLE {i + 1}]\n{t}")

            if isinstance(paragraphs, str) and paragraphs:
                contexts.append(f"[TEXT]\n{paragraphs}")
            elif isinstance(paragraphs, list):
                for p in paragraphs:
                    contexts.append(f"[TEXT]\n{p}")

            q = BenchmarkQuestion(
                id=f"mh_{count}",
                question=str(question_text),
                answer=str(answer),
                supporting_facts=[],
                context=contexts,
                reasoning_type="multi-step-numerical",
                metadata={
                    "dataset": "multihiertt",
                    "program": str(program),
                    "requires_arithmetic": True,
                    "split": split,
                },
            )

            questions.append(q)
            count += 1

        return BenchmarkDataset(
            name="MultiHiertt",
            description="Multi-step numerical reasoning over hierarchical tables (ACL 2022)",
            questions=questions,
            metrics=["exact_match", "f1", "execution_accuracy"],
            source_url="https://github.com/psunlpgroup/MultiHiertt",
        )

    @staticmethod
    def _load_from_local(
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """Fallback: load from local JSON files if available."""
        local_path = CACHE_DIR / f"{split}.json"
        if not local_path.exists():
            logger.warning("No local MultiHiertt data found", path=str(local_path))
            return BenchmarkDataset(
                name="MultiHiertt",
                description="Multi-step numerical reasoning (failed to load)",
                questions=[],
                metrics=["exact_match", "f1"],
                source_url="https://github.com/psunlpgroup/MultiHiertt",
            )

        with open(local_path) as f:
            data = json.load(f)

        questions: list[BenchmarkQuestion] = []
        items = data if isinstance(data, list) else data.get("data", [])

        for i, item in enumerate(items):
            if max_samples and i >= max_samples:
                break

            q = BenchmarkQuestion(
                id=f"mh_{i}",
                question=item.get("question", ""),
                answer=str(item.get("answer", "")),
                context=item.get("context", []),
                reasoning_type="multi-step-numerical",
                metadata={"dataset": "multihiertt", "split": split},
            )
            questions.append(q)

        return BenchmarkDataset(
            name="MultiHiertt",
            description="Multi-step numerical reasoning over hierarchical tables (ACL 2022)",
            questions=questions,
            metrics=["exact_match", "f1", "execution_accuracy"],
            source_url="https://github.com/psunlpgroup/MultiHiertt",
        )
