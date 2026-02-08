"""
DocVQA Benchmark Loader for RNSR Evaluation

DocVQA is a benchmark for visual question answering on document images.
Questions are posed about the content of scanned/photographed documents
including forms, receipts, letters, reports, and invoices.

This tests RNSR's vision pipeline (LayoutLM, OCR fallback, chart/table parsing).

Repository: https://www.docvqa.org/
Paper: "DocVQA: A Dataset for VQA on Document Images" (WACV 2021)

Key metrics:
- ANLS (Average Normalized Levenshtein Similarity):
  Measures string similarity between prediction and ground truth,
  allowing for minor OCR/spelling variations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog

from rnsr.benchmarks.standard_benchmarks import BenchmarkDataset, BenchmarkQuestion

logger = structlog.get_logger(__name__)

CACHE_DIR = Path("rnsr/benchmarks/data/docvqa")


def _compute_anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Compute ANLS (Average Normalized Levenshtein Similarity).

    ANLS = 1 - NL(pred, gt) if NL < threshold, else 0
    where NL is the normalized Levenshtein distance.
    """
    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0

    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()

    if pred == gt:
        return 1.0

    # Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    distance = dp[m][n]
    max_len = max(m, n)
    nl = distance / max_len if max_len > 0 else 0.0

    if nl < threshold:
        return 1.0 - nl
    return 0.0


class DocVQALoader:
    """Loader for the DocVQA dataset."""

    @staticmethod
    def load(
        split: str = "validation",
        max_samples: Optional[int] = None,
        download_images: bool = False,
    ) -> BenchmarkDataset:
        """
        Load the DocVQA dataset from HuggingFace.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Max number of questions to load
            download_images: Whether to download document images locally

        Returns:
            BenchmarkDataset containing DocVQA questions
        """
        try:
            from datasets import load_dataset  # type: ignore
            dataset = load_dataset("lmms-lab/DocVQA", split=split)
        except Exception:
            try:
                from datasets import load_dataset  # type: ignore
                dataset = load_dataset("nielsr/docvqa", split=split)
            except Exception as e:
                logger.error("Failed to load DocVQA dataset", error=str(e))
                return DocVQALoader._load_from_local(split, max_samples)

        questions: list[BenchmarkQuestion] = []
        count = 0

        for item in dataset:
            if not isinstance(item, dict):
                continue
            if max_samples and count >= max_samples:
                break

            question_text = item.get("question", "")
            if not question_text:
                continue

            # DocVQA answers can be a list of acceptable answers
            answers = item.get("answers", item.get("answer", []))
            if isinstance(answers, list):
                answer_str = answers[0] if answers else ""
                all_answers = answers
            else:
                answer_str = str(answers)
                all_answers = [answer_str]

            # Image handling
            image = item.get("image", None)
            image_path = None
            if download_images and image is not None:
                image_path = DocVQALoader._save_image(image, f"docvqa_{count}")

            # Build context: for DocVQA we may extract text via OCR or
            # provide the image path for vision-based processing
            contexts: list[str] = []

            # Some DocVQA variants include OCR text
            ocr_text = item.get("ocr_text", "") or item.get("words", "")
            if isinstance(ocr_text, list):
                ocr_text = " ".join(str(w) for w in ocr_text)
            if ocr_text:
                contexts.append(f"[OCR TEXT]\n{ocr_text}")

            q = BenchmarkQuestion(
                id=f"docvqa_{count}",
                question=str(question_text),
                answer=str(answer_str),
                supporting_facts=[],
                context=contexts,
                reasoning_type="visual-document-qa",
                metadata={
                    "dataset": "docvqa",
                    "all_answers": all_answers,
                    "image_path": str(image_path) if image_path else None,
                    "has_image": image is not None,
                    "split": split,
                },
            )

            questions.append(q)
            count += 1

        return BenchmarkDataset(
            name="DocVQA",
            description="Visual question answering on document images (WACV 2021)",
            questions=questions,
            metrics=["anls"],
            source_url="https://www.docvqa.org/",
        )

    @staticmethod
    def _save_image(image: Any, name: str) -> Optional[Path]:
        """Save a PIL image to disk."""
        try:
            from PIL import Image  # type: ignore

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = CACHE_DIR / f"{name}.png"
            if path.exists():
                return path

            if isinstance(image, Image.Image):
                image.save(path)
                return path
        except Exception as e:
            logger.warning("Failed to save DocVQA image", error=str(e))
        return None

    @staticmethod
    def compute_anls_score(
        predictions: list[str],
        ground_truths: list[list[str]],
    ) -> float:
        """
        Compute average ANLS across all predictions.

        Args:
            predictions: List of predicted answers
            ground_truths: List of lists of acceptable answers per question

        Returns:
            Average ANLS score
        """
        if not predictions:
            return 0.0

        scores: list[float] = []
        for pred, gts in zip(predictions, ground_truths):
            # Take max ANLS across all acceptable ground truth answers
            max_anls = max(_compute_anls(pred, gt) for gt in gts) if gts else 0.0
            scores.append(max_anls)

        return sum(scores) / len(scores)

    @staticmethod
    def _load_from_local(
        split: str = "validation",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """Fallback: load from local JSON files."""
        local_path = CACHE_DIR / f"{split}.json"
        if not local_path.exists():
            logger.warning("No local DocVQA data found", path=str(local_path))
            return BenchmarkDataset(
                name="DocVQA",
                description="Visual document QA (failed to load)",
                questions=[],
                metrics=["anls"],
                source_url="https://www.docvqa.org/",
            )

        with open(local_path) as f:
            data = json.load(f)

        questions: list[BenchmarkQuestion] = []
        items = data.get("data", data) if isinstance(data, dict) else data

        for i, item in enumerate(items):
            if max_samples and i >= max_samples:
                break

            answers = item.get("answers", [])
            answer_str = answers[0] if answers else ""

            q = BenchmarkQuestion(
                id=f"docvqa_{i}",
                question=item.get("question", ""),
                answer=str(answer_str),
                context=item.get("context", []),
                reasoning_type="visual-document-qa",
                metadata={
                    "dataset": "docvqa",
                    "all_answers": answers,
                    "split": split,
                },
            )
            questions.append(q)

        return BenchmarkDataset(
            name="DocVQA",
            description="Visual question answering on document images (WACV 2021)",
            questions=questions,
            metrics=["anls"],
            source_url="https://www.docvqa.org/",
        )
