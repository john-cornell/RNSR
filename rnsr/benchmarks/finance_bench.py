"""
FinanceBench Dataset Loader for RNSR Evaluation

FinanceBench is a benchmark for financial question answering (QA) using large language models (LLMs).
It assesses the ability of LLMs to answer questions about financial documents, requiring retrieval
from complex PDFs (tables, charts, etc.).

Repository: https://huggingface.co/datasets/PatronusAI/financebench
"""

import os
import requests
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import structlog
from datasets import load_dataset  # type: ignore

from rnsr.benchmarks.standard_benchmarks import BenchmarkDataset, BenchmarkQuestion

logger = structlog.get_logger(__name__)

CACHE_DIR = Path("rnsr/benchmarks/data/financebench")

class FinanceBenchLoader:
    """Loader for the FinanceBench dataset."""
    
    @staticmethod
    def _download_pdf(url: str, doc_name: str) -> Optional[Path]:
        """
        Download PDF from URL and cache it locally.
        Returns the path to the cached PDF.
        """
        if not url:
            return None
            
        # Create a safe filename (hash + original name sanitized)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        safe_name = "".join(c for c in doc_name if c.isalnum() or c in (' ', '.', '_', '-')).strip()
        safe_name = safe_name.replace(" ", "_")
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
            
        file_path = CACHE_DIR / f"{url_hash}_{safe_name}"
        
        if file_path.exists():
            return file_path
        
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading PDF", url=url, path=str(file_path))
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                f.write(response.content)
                
            return file_path
        except Exception as e:
            logger.error("Failed to download PDF", url=url, error=str(e))
            return None

    @staticmethod
    def load(
        split: str = "train",
        max_samples: Optional[int] = None,
        download_pdfs: bool = True
    ) -> BenchmarkDataset:
        """
        Load the FinanceBench dataset.
        
        Args:
            split: Dataset split to load (usually 'train' as test is hidden or same)
            max_samples: Max number of questions to load
            download_pdfs: Whether to download the referenced PDFs
            
        Returns:
            BenchmarkDataset containing FinanceBench questions
        """
        try:
            dataset = load_dataset("PatronusAI/financebench", split=split)
        except Exception as e:
            logger.error("Failed to load FinanceBench dataset", error=str(e))
            return BenchmarkDataset(
                name="FinanceBench",
                description="Financial QA (Failed to load)",
                questions=[],
                metrics=[],
                source_url=""
            )
        
        questions: List[BenchmarkQuestion] = []
        
        # FinanceBench structure:
        # question, answer, evidence_text, doc_name, doc_link, id, etc.
        
        count = 0
        for item in dataset:
            if not isinstance(item, dict):
                continue

            if max_samples and count >= max_samples:
                break
                
            doc_link = item.get("doc_link")
            doc_name = item.get("doc_name", "unknown_doc")
            
            pdf_path = None
            if download_pdfs and doc_link:
                # Some links might be missing or broken, handle gracefully
                pdf_path = FinanceBenchLoader._download_pdf(doc_link, doc_name)
            
            # If we couldn't get the PDF, we might skip or mark it
            # But we'll include it with metadata indicating missing file for now
            
            # Extract answer - typically a string in FinanceBench
            answer = item.get("answer", "")
            
            # Create question object
            q = BenchmarkQuestion(
                id=f"fb_{count}",  # FinanceBench doesn't have stable IDs in some versions
                question=item["question"],
                answer=str(answer),
                supporting_facts=[item.get("evidence_text", "")],
                context=[],  # Context is the PDF file, not text chunks
                reasoning_type="financial-retrieval",
                metadata={
                    "doc_name": doc_name,
                    "doc_link": doc_link,
                    "pdf_path": str(pdf_path) if pdf_path else None,
                    "page_index": item.get("page_index"), # Sometimes available
                    "dataset": "financebench"
                }
            )
            
            questions.append(q)
            count += 1
            
        return BenchmarkDataset(
            name="FinanceBench",
            description="Financial Question Answering on Complex PDFs",
            questions=questions,
            metrics=["answer_evaluation_llm"], # Will need LLM-based eval
            source_url="https://huggingface.co/datasets/PatronusAI/financebench"
        )
