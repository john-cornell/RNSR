"""
Text Builder - Build Document Tree from Raw Text

This module enables RLM processing for text that doesn't come from PDFs.
Used by benchmarks (which provide raw text contexts) and APIs.

The key insight from the research paper:
- Traditional RAG stuffs all context into the LLM prompt → Context Rot
- RLM stores document as variable (DOC_VAR) and navigates via summaries
- This requires a hierarchical tree structure

For raw text, we:
1. Apply semantic chunking to create logical segments
2. Use hierarchical clustering to create a tree structure
3. Generate summaries for navigation (skeleton index)
4. Store full text in KV store (the DOC_VAR abstraction)

Usage:
    from rnsr.ingestion.text_builder import build_tree_from_text
    
    # From a single text string
    tree = build_tree_from_text("Long document text here...")
    
    # From multiple context chunks (benchmark datasets)
    tree = build_tree_from_contexts(["context1", "context2", "context3"])
    
    # Then use with skeleton index as normal
    skeleton, kv_store = build_skeleton_index(tree)
    answer = run_navigator(question, skeleton, kv_store)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Union
from uuid import uuid4

import structlog

from rnsr.models import DocumentNode, DocumentTree

logger = structlog.get_logger(__name__)


@dataclass
class TextSegment:
    """A segment of text with metadata."""
    
    text: str
    start_char: int
    end_char: int
    level: int = 0
    header: str = ""
    segment_id: str = ""
    
    def __post_init__(self):
        if not self.segment_id:
            # Generate stable ID from content hash
            content_hash = hashlib.md5(self.text[:100].encode()).hexdigest()[:8]
            self.segment_id = f"seg_{content_hash}"


# =============================================================================
# Header Detection Patterns
# =============================================================================

# Common header patterns in text documents
HEADER_PATTERNS: list[tuple[str, Union[int, Callable[[re.Match[str]], int]]]] = [
    # Numbered headers: "1.", "1.1", "1.1.1", etc.
    (r'^(\d+\.)+\s+(.+)$', 1),  # level based on dot count
    # Markdown-style: "# Title", "## Subtitle"
    (r'^(#{1,4})\s+(.+)$', lambda m: len(m.group(1))),
    # ALL CAPS (likely headers)
    (r'^([A-Z][A-Z\s]{4,50})$', 1),
    # "Chapter X:", "Section X:", "Part X:"
    (r'^(Chapter|Section|Part)\s+\d+[:\.]?\s*(.*)$', 1),
    # Roman numerals: "I.", "II.", "III."
    (r'^(I{1,3}|IV|V|VI{1,3}|IX|X)[\.\)]\s+(.+)$', 1),
]


def _detect_headers_in_text(text: str) -> list[tuple[int, str, int]]:
    """
    Detect headers in text and their positions.
    
    Returns list of (char_position, header_text, level).
    """
    headers: list[tuple[int, str, int]] = []
    lines = text.split('\n')
    char_pos = 0
    
    for line in lines:
        stripped = line.strip()
        
        for pattern, level_info in HEADER_PATTERNS:
            match = re.match(pattern, stripped)
            if match:
                # Determine level
                if callable(level_info):
                    level = int(level_info(match))
                elif pattern.startswith(r'^(\d+\.)+'):
                    # Count dots for numbered headers
                    level = stripped.count('.', 0, match.end(1))
                else:
                    level = int(level_info)
                
                headers.append((char_pos, stripped, min(level, 3)))
                break
        
        char_pos += len(line) + 1  # +1 for newline
    
    return headers


# =============================================================================
# Semantic Chunking
# =============================================================================

def _semantic_chunk_text(
    text: str,
    chunk_size: int = 2000,
    min_chunk_size: int = 100,
) -> list[TextSegment]:
    """
    Chunk text semantically by detecting natural boundaries.
    
    Strategy:
    1. First try to split on detected headers
    2. Then split on paragraph breaks
    3. Finally split on sentences if still too large
    """
    if len(text) < chunk_size:
        return [TextSegment(text=text, start_char=0, end_char=len(text))]
    
    segments: list[TextSegment] = []
    headers = _detect_headers_in_text(text)
    
    if headers:
        # Split on headers
        for i, (pos, header_text, level) in enumerate(headers):
            start = pos
            end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            
            segment_text = text[start:end].strip()
            if len(segment_text) >= min_chunk_size:
                segments.append(TextSegment(
                    text=segment_text,
                    start_char=start,
                    end_char=end,
                    level=level,
                    header=header_text,
                ))
        
        # If we got good segments, use them
        if len(segments) >= 2:
            return segments
    
    # Fallback: split on paragraph breaks
    paragraphs = re.split(r'\n\n+', text)
    current_chunk = ""
    current_start = 0
    char_pos = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            segments.append(TextSegment(
                text=current_chunk,
                start_char=current_start,
                end_char=char_pos,
            ))
            current_chunk = para
            current_start = char_pos
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        char_pos += len(para) + 2  # Account for \n\n
    
    # Add final chunk
    if current_chunk and len(current_chunk) >= min_chunk_size:
        segments.append(TextSegment(
            text=current_chunk,
            start_char=current_start,
            end_char=len(text),
        ))
    
    # If we only got one segment, try harder to split it
    if len(segments) <= 1 and len(text) > chunk_size:
        # Split on sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        segments = []
        current_chunk = ""
        current_start = 0
        char_pos = 0
        
        for sent in sentences:
            if len(current_chunk) + len(sent) > chunk_size and current_chunk:
                segments.append(TextSegment(
                    text=current_chunk,
                    start_char=current_start,
                    end_char=char_pos,
                ))
                current_chunk = sent
                current_start = char_pos
            else:
                current_chunk = (current_chunk + " " + sent).strip() if current_chunk else sent
            
            char_pos += len(sent) + 1
        
        if current_chunk:
            segments.append(TextSegment(
                text=current_chunk,
                start_char=current_start,
                end_char=len(text),
            ))
    
    return segments if segments else [TextSegment(text=text, start_char=0, end_char=len(text))]


# =============================================================================
# Tree Building
# =============================================================================

def _build_hierarchy_from_segments(
    segments: list[TextSegment],
    max_children: int = 10,
) -> DocumentNode:
    """
    Build a hierarchical tree from flat segments.
    
    Uses segment levels (from headers) to create proper nesting,
    or creates a balanced tree if no levels detected.
    """
    if not segments:
        return DocumentNode(
            id="root",
            content="",
            header="Document",
            level=0,
            children=[],
        )
    
    # Check if we have level information
    has_levels = any(s.level > 0 for s in segments)
    
    if has_levels:
        # Build tree using detected levels
        root = DocumentNode(
            id="root",
            content="",
            header="Document",
            level=0,
            children=[],
        )
        
        # Stack-based tree building
        stack: list[DocumentNode] = [root]
        
        for seg in segments:
            node = DocumentNode(
                id=seg.segment_id,
                content=seg.text,
                header=seg.header or f"Section {len(root.children) + 1}",
                level=seg.level + 1,  # Root is level 0
                children=[],
            )
            
            # Find appropriate parent
            while len(stack) > 1 and stack[-1].level >= node.level:
                stack.pop()
            
            stack[-1].children.append(node)
            stack.append(node)
        
        return root
    
    else:
        # No levels detected - create balanced tree
        root = DocumentNode(
            id="root",
            content="",
            header="Document",
            level=0,
            children=[],
        )
        
        if len(segments) <= max_children:
            # All segments as direct children
            for i, seg in enumerate(segments):
                root.children.append(DocumentNode(
                    id=seg.segment_id,
                    content=seg.text,
                    header=seg.header or f"Section {i + 1}",
                    level=1,
                    children=[],
                ))
        else:
            # Group into intermediate nodes
            group_size = (len(segments) + max_children - 1) // max_children
            
            for group_idx in range(0, len(segments), group_size):
                group_segments = segments[group_idx:group_idx + group_size]
                
                group_node = DocumentNode(
                    id=f"group_{group_idx}",
                    content="",
                    header=f"Part {group_idx // group_size + 1}",
                    level=1,
                    children=[],
                )
                
                for seg in group_segments:
                    group_node.children.append(DocumentNode(
                        id=seg.segment_id,
                        content=seg.text,
                        header=seg.header or "Segment",
                        level=2,
                        children=[],
                    ))
                
                root.children.append(group_node)
        
        return root


def _count_nodes(node: DocumentNode) -> int:
    """Count total nodes in tree."""
    return 1 + sum(_count_nodes(child) for child in node.children)


def _get_tree_depth(node: DocumentNode) -> int:
    """Get maximum depth of tree."""
    if not node.children:
        return 1
    return 1 + max(_get_tree_depth(child) for child in node.children)


# =============================================================================
# Public API
# =============================================================================

def build_tree_from_text(
    text: str | list[str],
    chunk_size: int = 2000,
    generate_summaries: bool = False,
) -> DocumentTree:
    """
    Build a document tree from raw text for RLM processing.
    
    This is the key function that enables RNSR/RLM on non-PDF inputs.
    The resulting tree can be used with build_skeleton_index() and
    run_navigator() for full RLM query processing.
    
    Args:
        text: Either a single text string or a list of context chunks.
              Lists are common from benchmark datasets.
        chunk_size: Target size for semantic chunks (characters).
        generate_summaries: Whether to generate LLM summaries for
                           navigation. Generally not needed since
                           skeleton_index handles this.
    
    Returns:
        DocumentTree suitable for skeleton indexing.
    
    Example:
        # From benchmark contexts
        tree = build_tree_from_text(["ctx1", "ctx2", "ctx3"])
        skeleton, kv = build_skeleton_index(tree)
        result = run_navigator("question", skeleton, kv)
    """
    # Handle list of contexts (common from benchmarks)
    if isinstance(text, list):
        if len(text) == 0:
            root = DocumentNode(
                id="root",
                content="",
                header="Empty Document",
                level=0,
                children=[],
            )
            return DocumentTree(
                id=f"doc_{uuid4().hex[:8]}",
                title="Empty Document",
                root=root,
                total_nodes=1,
                ingestion_tier=2,
            )
        
        # If contexts are already chunked, use them directly
        if all(len(ctx) < chunk_size * 2 for ctx in text):
            segments = [
                TextSegment(
                    text=ctx,
                    start_char=0,
                    end_char=len(ctx),
                    header=f"Context {i + 1}",
                )
                for i, ctx in enumerate(text)
                if ctx.strip()
            ]
        else:
            # Combine and re-chunk
            combined = "\n\n---\n\n".join(text)
            segments = _semantic_chunk_text(combined, chunk_size)
    else:
        segments = _semantic_chunk_text(text, chunk_size)
    
    # Build hierarchical tree
    root = _build_hierarchy_from_segments(segments)
    
    # Create DocumentTree wrapper
    tree = DocumentTree(
        id=f"doc_{uuid4().hex[:8]}",
        title="Text Document",
        root=root,
        total_nodes=_count_nodes(root),
        ingestion_tier=2,  # Mark as semantic tier (not PDF-based)
    )
    
    logger.info(
        "tree_built_from_text",
        num_segments=len(segments),
        tree_depth=_get_tree_depth(root),
        total_nodes=tree.total_nodes,
    )
    
    return tree


def build_tree_from_contexts(
    contexts: list[str],
    question: str | None = None,
) -> DocumentTree:
    """
    Build a tree optimized for benchmark evaluation.
    
    This is a convenience wrapper for benchmark datasets that
    provide pre-chunked context passages.
    
    Args:
        contexts: List of context passages from benchmark.
        question: Optional question for context (unused currently).
    
    Returns:
        DocumentTree for skeleton indexing.
    """
    return build_tree_from_text(
        contexts,
        chunk_size=2000,
        generate_summaries=False,  # Skip summaries for speed in benchmarks
    )
