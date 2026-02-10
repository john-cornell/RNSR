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
import re
from pathlib import Path
from typing import Any, Optional

import structlog

from rnsr.benchmarks.standard_benchmarks import BenchmarkDataset, BenchmarkQuestion

logger = structlog.get_logger(__name__)


def _html_table_to_pipe(html: str) -> str:
    """Convert an HTML ``<table>`` string to pipe-separated rows.

    Handles ``<tr>`` / ``<td>`` / ``<th>`` elements and strips inner tags
    like ``<i>``, ``<b>``, ``<br>``.  Returns a multi-line string where
    each line is one row with cells separated by `` | ``.
    """
    # Strip inner formatting tags but keep their text
    clean = re.sub(r"</?(?:i|b|em|strong|span|br|br/)>", "", html, flags=re.I)

    rows: list[str] = []
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", clean, flags=re.I | re.S):
        cells: list[str] = []
        for cell_match in re.finditer(
            r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", tr_match.group(1), flags=re.I | re.S
        ):
            cell_text = cell_match.group(1).strip()
            # Collapse whitespace
            cell_text = re.sub(r"\s+", " ", cell_text)
            cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)

CACHE_DIR = Path("rnsr/benchmarks/data/multihiertt")


class MultiHierttLoader:
    """Loader for the MultiHiertt dataset."""

    @staticmethod
    def load(
        split: str = "dev",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """
        Load the MultiHiertt dataset from HuggingFace.

        Uses hf_hub_download to get raw JSON from yilunzhao/MultiHiertt.
        The 'dev' split is used by default because 'test' has no answers.

        Args:
            split: Dataset split to load ('train', 'dev')
            max_samples: Max number of questions to load

        Returns:
            BenchmarkDataset containing MultiHiertt questions
        """
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            json_path = hf_hub_download(
                repo_id="yilunzhao/MultiHiertt",
                filename=f"multihiertt_data/{split}.json",
                repo_type="dataset",
            )
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load MultiHiertt dataset", error=str(e))
            return MultiHierttLoader._load_from_local(split, max_samples)

        questions: list[BenchmarkQuestion] = []
        count = 0

        for item in data:
            if not isinstance(item, dict):
                continue
            if max_samples and count >= max_samples:
                break

            # MultiHiertt JSON: each item has tables, paragraphs, qa
            # NOTE: `tables` contains actual HTML table strings.
            # `table_description` is a dict of cell-coordinate references
            # (e.g. "0-2-1") and should NOT be used for content.
            tables_raw = item.get("tables", [])
            paragraphs = item.get("paragraphs", [])
            qa = item.get("qa", {})
            question_text = qa.get("question", item.get("question", ""))
            answer = qa.get("exe_ans", qa.get("answer", item.get("answer", "")))
            program = qa.get("program", item.get("program", ""))

            if not question_text:
                continue

            # Build context from tables + paragraphs
            contexts: list[str] = []

            # --- Tables ---
            if isinstance(tables_raw, list):
                for i, t in enumerate(tables_raw):
                    if isinstance(t, str) and "<table" in t.lower():
                        # HTML table → convert to pipe-separated
                        table_str = _html_table_to_pipe(t)
                        if table_str.strip():
                            contexts.append(f"[TABLE {i + 1}]\n{table_str}")
                        else:
                            contexts.append(f"[TABLE {i + 1}]\n{t}")
                    elif isinstance(t, list):
                        # List-of-lists (rows × cells)
                        table_str = "\n".join(
                            " | ".join(str(cell) for cell in row) for row in t
                        )
                        contexts.append(f"[TABLE {i + 1}]\n{table_str}")
                    elif isinstance(t, str) and t.strip():
                        contexts.append(f"[TABLE {i + 1}]\n{t}")

            # --- Paragraphs (skip bare table markers like "## Table 0 ##") ---
            _TABLE_MARKER_RE = re.compile(r"^##\s*Table\s+\d+\s*##$", re.I)
            if isinstance(paragraphs, str) and paragraphs:
                contexts.append(f"[TEXT]\n{paragraphs}")
            elif isinstance(paragraphs, list):
                for p in paragraphs:
                    if isinstance(p, str) and _TABLE_MARKER_RE.match(p.strip()):
                        continue  # skip useless marker
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
