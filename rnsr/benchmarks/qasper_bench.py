"""
QASPER Benchmark Loader for RNSR Evaluation

QASPER is a benchmark for question answering over NLP research papers.
Questions require reading full-length papers and reasoning across multiple
sections (introduction, methods, results, etc.).

Answer types:
- Extractive: Span from the paper
- Abstractive: Free-form text answer
- Yes/No: Boolean answer
- Unanswerable: Question cannot be answered from the paper

Repository: https://allenai.org/data/qasper
Paper: "A Dataset of Information-Seeking Questions and Answers Anchored
       in Research Papers" (NAACL 2021)

Key metrics:
- F1: Token-level overlap
- Answer evidence retrieval: Whether the system finds the right paragraphs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog

from rnsr.benchmarks.standard_benchmarks import BenchmarkDataset, BenchmarkQuestion

logger = structlog.get_logger(__name__)

CACHE_DIR = Path("rnsr/benchmarks/data/qasper")


class QASPERLoader:
    """Loader for the QASPER dataset."""

    @staticmethod
    def load(
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """
        Load the QASPER dataset from HuggingFace.

        Args:
            split: Dataset split to load ('train', 'validation', 'test')
            max_samples: Max number of questions to load

        Returns:
            BenchmarkDataset containing QASPER questions
        """
        try:
            from datasets import load_dataset  # type: ignore
            dataset = load_dataset("allenai/qasper", split=split)
        except Exception as e:
            logger.error("Failed to load QASPER dataset", error=str(e))
            return QASPERLoader._load_from_local(split, max_samples)

        questions: list[BenchmarkQuestion] = []
        count = 0

        for paper in dataset:
            if not isinstance(paper, dict):
                continue
            if max_samples and count >= max_samples:
                break

            # Build full paper context from sections
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            full_text = paper.get("full_text", {})

            contexts: list[str] = []
            if title:
                contexts.append(f"[TITLE]\n{title}")
            if abstract:
                contexts.append(f"[ABSTRACT]\n{abstract}")

            # full_text is typically {section_name: [paragraphs]}
            section_names = full_text.get("section_name", [])
            paragraphs_list = full_text.get("paragraphs", [])

            for sec_name, sec_paragraphs in zip(section_names, paragraphs_list):
                if isinstance(sec_paragraphs, list):
                    sec_text = "\n".join(str(p) for p in sec_paragraphs)
                else:
                    sec_text = str(sec_paragraphs)
                contexts.append(f"[{sec_name}]\n{sec_text}")

            # Extract questions and answers for this paper
            qas = paper.get("qas", {})
            qa_questions = qas.get("question", [])
            qa_answers_list = qas.get("answers", [])

            for q_text, answers_data in zip(qa_questions, qa_answers_list):
                if max_samples and count >= max_samples:
                    break

                if not q_text:
                    continue

                # Each question can have multiple annotator answers
                # Use the first answer as ground truth
                answer_str, answer_type, evidence = QASPERLoader._extract_answer(
                    answers_data
                )

                if not answer_str:
                    continue

                q = BenchmarkQuestion(
                    id=f"qasper_{count}",
                    question=str(q_text),
                    answer=answer_str,
                    supporting_facts=evidence,
                    context=contexts,
                    reasoning_type=f"scientific-{answer_type}",
                    metadata={
                        "dataset": "qasper",
                        "paper_title": title,
                        "answer_type": answer_type,
                        "split": split,
                    },
                )

                questions.append(q)
                count += 1

        return BenchmarkDataset(
            name="QASPER",
            description="Question answering over NLP research papers (NAACL 2021)",
            questions=questions,
            metrics=["f1", "evidence_f1"],
            source_url="https://allenai.org/data/qasper",
        )

    @staticmethod
    def _extract_answer(
        answers_data: Any,
    ) -> tuple[str, str, list[str]]:
        """
        Extract the best answer from QASPER answer annotations.

        Returns:
            (answer_text, answer_type, evidence_list)
        """
        if not answers_data:
            return "", "unknown", []

        # answers_data can be a dict with lists of annotator answers
        answer_list = answers_data.get("answer", [])
        if not answer_list:
            return "", "unknown", []

        # Use first annotator's answer
        first = answer_list[0] if isinstance(answer_list, list) else answer_list

        if isinstance(first, dict):
            # QASPER format: {free_form_answer, extractive_spans, yes_no, unanswerable}
            if first.get("unanswerable", False):
                return "Unanswerable", "unanswerable", []
            elif first.get("yes_no") is not None:
                return str(first["yes_no"]).capitalize(), "yes_no", []
            elif first.get("extractive_spans"):
                spans = first["extractive_spans"]
                if isinstance(spans, list):
                    return " ".join(str(s) for s in spans), "extractive", list(spans)
                return str(spans), "extractive", [str(spans)]
            elif first.get("free_form_answer"):
                return str(first["free_form_answer"]), "abstractive", []

        # Fallback: treat as string
        return str(first), "unknown", []

    @staticmethod
    def _load_from_local(
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """Fallback: load from local JSON files."""
        local_path = CACHE_DIR / f"qasper-{split}-v0.3.json"
        if not local_path.exists():
            logger.warning("No local QASPER data found", path=str(local_path))
            return BenchmarkDataset(
                name="QASPER",
                description="Scientific paper QA (failed to load)",
                questions=[],
                metrics=["f1", "evidence_f1"],
                source_url="https://allenai.org/data/qasper",
            )

        with open(local_path) as f:
            data = json.load(f)

        questions: list[BenchmarkQuestion] = []
        count = 0

        for paper_id, paper in data.items():
            if max_samples and count >= max_samples:
                break

            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            sections = paper.get("full_text", [])

            contexts: list[str] = []
            if title:
                contexts.append(f"[TITLE]\n{title}")
            if abstract:
                contexts.append(f"[ABSTRACT]\n{abstract}")
            for sec in sections:
                sec_name = sec.get("section_name", "Section")
                sec_text = "\n".join(sec.get("paragraphs", []))
                contexts.append(f"[{sec_name}]\n{sec_text}")

            for qa in paper.get("qas", []):
                if max_samples and count >= max_samples:
                    break

                question_text = qa.get("question", "")
                if not question_text:
                    continue

                # Get first annotator answer
                answers = qa.get("answers", [])
                if answers:
                    ans = answers[0].get("answer", {})
                    if ans.get("unanswerable"):
                        answer_str = "Unanswerable"
                    elif ans.get("extractive_spans"):
                        answer_str = " ".join(ans["extractive_spans"])
                    elif ans.get("free_form_answer"):
                        answer_str = ans["free_form_answer"]
                    elif ans.get("yes_no") is not None:
                        answer_str = str(ans["yes_no"]).capitalize()
                    else:
                        answer_str = ""
                else:
                    answer_str = ""

                q = BenchmarkQuestion(
                    id=f"qasper_{count}",
                    question=question_text,
                    answer=answer_str,
                    context=contexts,
                    reasoning_type="scientific-qa",
                    metadata={
                        "dataset": "qasper",
                        "paper_title": title,
                        "split": split,
                    },
                )
                questions.append(q)
                count += 1

        return BenchmarkDataset(
            name="QASPER",
            description="Question answering over NLP research papers (NAACL 2021)",
            questions=questions,
            metrics=["f1", "evidence_f1"],
            source_url="https://allenai.org/data/qasper",
        )
