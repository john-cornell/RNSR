"""
TAT-QA Benchmark Loader for RNSR Evaluation

TAT-QA (Tabular And Textual dataset for Question Answering) requires
jointly reasoning over tables and text in financial reports.

Questions are categorized by answer type:
- Span extraction (from text or table)
- Multi-span extraction
- Counting
- Arithmetic (addition, subtraction, multiplication, division, etc.)

Repository: https://nextplusplus.github.io/TAT-QA/
Paper: "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular
       and Textual Content in Finance" (ACL 2021)

Key metrics:
- Exact Match (EM): Exact string match
- F1: Token-level F1 score
- Per-type accuracy: Breakdown by answer type (span, count, arithmetic)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog

from rnsr.benchmarks.standard_benchmarks import BenchmarkDataset, BenchmarkQuestion

logger = structlog.get_logger(__name__)

CACHE_DIR = Path("rnsr/benchmarks/data/tatqa")


class TATQALoader:
    """Loader for the TAT-QA dataset."""

    @staticmethod
    def load(
        split: str = "dev",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """
        Load the TAT-QA dataset from HuggingFace.

        Args:
            split: Dataset split to load ('train', 'dev')
            max_samples: Max number of questions to load

        Returns:
            BenchmarkDataset containing TAT-QA questions
        """
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            json_path = hf_hub_download(
                repo_id="next-tat/TAT-QA",
                filename=f"tatqa_dataset_{split}.json",
                repo_type="dataset",
            )
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load TAT-QA dataset", error=str(e))
            return TATQALoader._load_from_local(split, max_samples)

        questions: list[BenchmarkQuestion] = []
        count = 0

        for doc in data:
            if not isinstance(doc, dict):
                continue
            if max_samples and count >= max_samples:
                break

            # TAT-QA raw JSON: each doc has table, paragraphs, questions
            table = doc.get("table", {})
            paragraphs = doc.get("paragraphs", [])

            for qa in doc.get("questions", []):
                if max_samples and count >= max_samples:
                    break

                question_text = qa.get("question", "")
                answer = qa.get("answer", "")
                answer_type = qa.get("answer_type", "span")
                scale = qa.get("scale", "")
                derivation = qa.get("derivation", "")

                if not question_text:
                    continue

                # Build context from table + paragraphs
                contexts: list[str] = []

                # Format table
                if isinstance(table, dict) and table:
                    table_str = TATQALoader._format_table(table)
                    if table_str:
                        contexts.append(f"[TABLE]\n{table_str}")
                elif isinstance(table, str) and table:
                    contexts.append(f"[TABLE]\n{table}")
                elif isinstance(table, list):
                    table_str = TATQALoader._format_table_list(table)
                    if table_str:
                        contexts.append(f"[TABLE]\n{table_str}")

                # Format paragraphs
                if isinstance(paragraphs, list):
                    for p in paragraphs:
                        text = p.get("text", p) if isinstance(p, dict) else str(p)
                        contexts.append(f"[TEXT]\n{text}")
                elif isinstance(paragraphs, str) and paragraphs:
                    contexts.append(f"[TEXT]\n{paragraphs}")

                # Normalize answer
                if isinstance(answer, list):
                    answer_str = ", ".join(str(a) for a in answer)
                else:
                    answer_str = str(answer)
                if scale:
                    answer_str = f"{answer_str} {scale}"

                q = BenchmarkQuestion(
                    id=f"tatqa_{count}",
                    question=str(question_text),
                    answer=answer_str,
                    supporting_facts=[],
                    context=contexts,
                    reasoning_type=f"hybrid-{answer_type}",
                    metadata={
                        "dataset": "tatqa",
                        "answer_type": str(answer_type),
                        "scale": str(scale),
                        "derivation": str(derivation),
                        "split": split,
                    },
                )

                questions.append(q)
                count += 1

        return BenchmarkDataset(
            name="TAT-QA",
            description="Tabular and textual QA on financial reports (ACL 2021)",
            questions=questions,
            metrics=["exact_match", "f1", "per_type_accuracy"],
            source_url="https://nextplusplus.github.io/TAT-QA/",
        )

    @staticmethod
    def _format_table(table: dict[str, Any]) -> str:
        """Format a table dict into readable text."""
        rows = table.get("rows", table.get("data", []))
        header = table.get("header", table.get("columns", []))

        if not rows:
            return ""

        lines: list[str] = []
        if header:
            lines.append(" | ".join(str(h) for h in header))
            lines.append("-" * len(lines[0]))

        for row in rows:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            elif isinstance(row, dict):
                lines.append(" | ".join(str(v) for v in row.values()))

        return "\n".join(lines)

    @staticmethod
    def _format_table_list(table: list[Any]) -> str:
        """Format a table given as list of rows."""
        if not table:
            return ""
        lines: list[str] = []
        for row in table:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            elif isinstance(row, dict):
                lines.append(" | ".join(str(v) for v in row.values()))
        return "\n".join(lines)

    @staticmethod
    def _load_from_local(
        split: str = "dev",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """Fallback: load from local JSON files."""
        local_path = CACHE_DIR / f"tatqa_dataset_{split}.json"
        if not local_path.exists():
            logger.warning("No local TAT-QA data found", path=str(local_path))
            return BenchmarkDataset(
                name="TAT-QA",
                description="Tabular and textual financial QA (failed to load)",
                questions=[],
                metrics=["exact_match", "f1"],
                source_url="https://nextplusplus.github.io/TAT-QA/",
            )

        with open(local_path) as f:
            data = json.load(f)

        questions: list[BenchmarkQuestion] = []
        count = 0

        for doc in data:
            table = doc.get("table", {})
            paragraphs = doc.get("paragraphs", [])

            for qa in doc.get("questions", []):
                if max_samples and count >= max_samples:
                    break

                contexts: list[str] = []
                if table:
                    contexts.append(f"[TABLE]\n{json.dumps(table, indent=2)}")
                for p in paragraphs:
                    text = p.get("text", str(p)) if isinstance(p, dict) else str(p)
                    contexts.append(f"[TEXT]\n{text}")

                answer = qa.get("answer", "")
                if isinstance(answer, list):
                    answer = ", ".join(str(a) for a in answer)

                q = BenchmarkQuestion(
                    id=f"tatqa_{count}",
                    question=qa.get("question", ""),
                    answer=str(answer),
                    context=contexts,
                    reasoning_type=f"hybrid-{qa.get('answer_type', 'span')}",
                    metadata={
                        "dataset": "tatqa",
                        "answer_type": qa.get("answer_type", "span"),
                        "split": split,
                    },
                )
                questions.append(q)
                count += 1

        return BenchmarkDataset(
            name="TAT-QA",
            description="Tabular and textual QA on financial reports (ACL 2021)",
            questions=questions,
            metrics=["exact_match", "f1", "per_type_accuracy"],
            source_url="https://nextplusplus.github.io/TAT-QA/",
        )
