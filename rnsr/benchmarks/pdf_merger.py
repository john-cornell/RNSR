"""
PDF Merger Utility for "Chaos Mode" Benchmarking.

This tool merges multiple random PDFs into a single "Frankenstein" document
to test retrieving information from a specific sub-document within a larger,
noisy context. This simulates searching through a "Binder" or a merged scan package.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import structlog

# Try to import fitz (PyMuPDF)
try:
    import fitz  # type: ignore
except ImportError:
    fitz = None

logger = structlog.get_logger(__name__)

class PDFMerger:
    """Helper to merge PDFs for chaos testing."""
    
    @staticmethod
    def merge_pdfs(
        target_pdf_path: Path, 
        distractor_pdf_paths: List[Path], 
        output_path: Path,
        insert_position: str = "random"
    ) -> Dict[str, Any]:
        """
        Merge target_pdf into a list of distractor_pdfs.
        
        Args:
            target_pdf_path: The PDF containing the answer.
            distractor_pdf_paths: List of irrelevant PDFs.
            output_path: Where to save the merged file.
            insert_position: 'start', 'end', or 'random'.
            
        Returns:
            Metadata about the merge (page ranges).
        """
        if not fitz:
            raise ImportError("PyMuPDF (fitz) is required for PDF merging. Install 'pymupdf'.")
            
        merged_doc = fitz.open()
        
        # Prepare list of docs to merge
        # (doc_object, is_target, label)
        docs_to_merge = []
        
        # Load distractors
        for p in distractor_pdf_paths:
            try:
                doc = fitz.open(p)
                docs_to_merge.append((doc, False, p.name))
            except Exception as e:
                logger.warning(f"Could not open distractor {p}: {e}")

        # Load target
        try:
            target_doc = fitz.open(target_pdf_path)
            target_entry = (target_doc, True, target_pdf_path.name)
        except Exception as e:
            logger.error(f"Could not open target PDF {target_pdf_path}: {e}")
            return {}

        # Insert target
        if insert_position == "start":
            docs_to_merge.insert(0, target_entry)
        elif insert_position == "end":
            docs_to_merge.append(target_entry)
        else: # random
            idx = random.randint(0, len(docs_to_merge))
            docs_to_merge.insert(idx, target_entry)
            
        # Perform merge
        current_page = 0
        target_page_range = (0, 0)
        
        for doc, is_target, label in docs_to_merge:
            page_count = doc.page_count
            merged_doc.insert_pdf(doc)
            
            start = current_page
            end = current_page + page_count - 1
            
            if is_target:
                target_page_range = (start, end)
                
            current_page += page_count
            doc.close()
            
        # Save
        merged_doc.save(output_path)
        merged_doc.close()
        
        return {
            "merged_file": str(output_path),
            "target_filename": target_pdf_path.name,
            "target_page_range": target_page_range, # 0-indexed, inclusive
            "total_pages": current_page
        }

    @staticmethod
    def create_chaos_dataset(
        base_dataset_questions, 
        pool_of_pdfs: List[Path], 
        output_dir: Path,
        num_distractors: int = 3
    ):
        """
        Takes a list of BenchmarkQuestions (e.g. from FinanceBench) and creates
        a new version where each source PDF is merged with random distractors.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        chaos_questions = []
        
        for q in base_dataset_questions:
            try:
                metadata = q.metadata or {}
                original_pdf = metadata.get("pdf_path")
                
                if not original_pdf:
                    continue
                    
                target_path = Path(original_pdf)
                if not target_path.exists():
                    logger.warning(f"Original PDF not found: {target_path}")
                    continue
                
                # Select random distractors (excluding self)
                candidates = [p for p in pool_of_pdfs if p.name != target_path.name]
                if len(candidates) < num_distractors:
                    distractors = candidates
                else:
                    distractors = random.sample(candidates, num_distractors)
                
                # Merge
                merged_filename = f"chaos_{q.id}.pdf"
                merged_path = output_dir / merged_filename
                
                merge_info = PDFMerger.merge_pdfs(
                    target_path, distractors, merged_path
                )
                
                # Clone question and update metadata
                new_meta = metadata.copy()
                new_meta.update({
                    "original_pdf_path": str(target_path),
                    "pdf_path": str(merged_path), # Point to the chaotic file
                    "chaos_mode": True,
                    "target_page_range": merge_info["target_page_range"],
                    "distractors": [d.name for d in distractors]
                })
                
                # Update question text slightly to hint? No, let's keep it hard.
                # q.question = q.question # Unchanged
                
                # Update question object (requires creating new instance as it might be immutable-ish)
                from rnsr.benchmarks.standard_benchmarks import BenchmarkQuestion
                new_q = BenchmarkQuestion(
                    id=f"{q.id}_chaos",
                    question=q.question,
                    answer=q.answer,
                    supporting_facts=q.supporting_facts,
                    context=q.context, # Usually empty for PDF benchmarks
                    reasoning_type=q.reasoning_type,
                    metadata=new_meta
                )
                chaos_questions.append(new_q)
                
            except Exception as e:
                logger.error(f"Failed to process chaos for question {q.id}: {e}")
                
        return chaos_questions

