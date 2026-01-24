"""
Semantic Fallback - TIER 2: For Flat Text Documents

When the Font Histogram Analyzer detects no font variance (flat text),
this module uses LlamaIndex's SemanticSplitterNodeParser to generate
"synthetic" sections based on embedding shifts.

Use this fallback when:
- Document has uniform font size throughout
- No headers can be detected via font analysis
- Document is machine-generated with no formatting
"""

from __future__ import annotations

from pathlib import Path

import fitz
import structlog

from rnsr.models import DocumentNode, DocumentTree

logger = structlog.get_logger(__name__)


def extract_raw_text(pdf_path: Path | str) -> str:
    """
    Extract all text from a PDF as a single string.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Full text content of the document.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    # get_text() returns str when called with no args or "text"
    full_text = "\n\n".join(str(page.get_text()) for page in doc)
    doc.close()
    
    return full_text


def try_semantic_splitter_ingestion(
    pdf_path: Path | str,
    embed_provider: str | None = None,
) -> DocumentTree:
    """
    TIER 2 Fallback: Use semantic splitting for flat text documents.
    
    When Font Histogram detects no font variance, this method:
    1. Extracts raw text from the PDF
    2. Uses embedding-based splitting to find natural breaks
    3. Generates synthetic section headers
    
    Args:
        pdf_path: Path to the PDF file.
        embed_provider: Embedding provider ("openai", "gemini", or None for auto).
        
    Returns:
        DocumentTree with synthetic sections.
    """
    pdf_path = Path(pdf_path)
    
    logger.info("using_semantic_splitter", path=str(pdf_path))
    
    # Extract raw text
    full_text = extract_raw_text(pdf_path)
    
    if not full_text.strip():
        logger.warning("no_text_extracted", path=str(pdf_path))
        # Return minimal tree
        root = DocumentNode(id="root", level=0, header="Document")
        return DocumentTree(
            title="Empty Document",
            root=root,
            total_nodes=1,
            ingestion_tier=2,
            ingestion_method="semantic_splitter",
        )
    
    # Try to import LlamaIndex components
    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        
        # Get embedding model (supports OpenAI, Gemini, auto-detect)
        embed_model = _get_embedding_model(embed_provider)
        
        # Create semantic splitter
        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=95,
            buffer_size=1,
        )
        
        # Split document
        llama_doc = Document(text=full_text)
        nodes = splitter.get_nodes_from_documents([llama_doc])
        
        logger.info(
            "semantic_split_complete",
            chunks=len(nodes),
        )
        
        # Build tree from semantic chunks
        return _build_tree_from_semantic_nodes(nodes, pdf_path.stem)
        
    except ImportError as e:
        logger.warning(
            "llama_index_not_available",
            error=str(e),
            fallback="simple_chunking",
        )
        # Fall back to simple chunking
        return _simple_chunk_fallback(full_text, pdf_path.stem)


def _get_embedding_model(provider: str | None = None):
    """
    Get embedding model with multi-provider support.
    
    Supports: OpenAI, Gemini, auto-detect.
    
    Args:
        provider: "openai", "gemini", or None for auto-detect.
        
    Returns:
        LlamaIndex-compatible embedding model.
    """
    import os
    
    # Auto-detect provider if not specified
    if provider is None:
        if os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError(
                "No embedding API key found. "
                "Set GOOGLE_API_KEY or OPENAI_API_KEY."
            )
    
    provider = provider.lower()
    
    if provider == "gemini":
        try:
            from llama_index.embeddings.gemini import GeminiEmbedding
            
            logger.info("using_gemini_embeddings")
            return GeminiEmbedding(model_name="models/text-embedding-004")
        except ImportError:
            raise ImportError(
                "Gemini embeddings not installed. "
                "Install with: pip install llama-index-embeddings-gemini"
            )
    
    elif provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            logger.info("using_openai_embeddings")
            return OpenAIEmbedding(model="text-embedding-3-small")
        except ImportError:
            raise ImportError(
                "OpenAI embeddings not installed. "
                "Install with: pip install llama-index-embeddings-openai"
            )
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def _build_tree_from_semantic_nodes(nodes: list, title: str) -> DocumentTree:
    """
    Build a flat tree structure from semantic splitter nodes.
    
    Since semantic splitting doesn't provide hierarchy, we create
    a flat structure with synthetic section headers.
    """
    root = DocumentNode(
        id="root",
        level=0,
        header=title,
    )
    
    for i, node in enumerate(nodes, 1):
        # Generate synthetic header from first few words
        text = node.text.strip()
        synthetic_header = _generate_synthetic_header(text, i)
        
        section = DocumentNode(
            id=f"sec_{i:03d}",
            level=1,  # All at same level (flat)
            header=synthetic_header,
            content=text,
        )
        root.children.append(section)
    
    return DocumentTree(
        title=title,
        root=root,
        total_nodes=len(nodes) + 1,
        ingestion_tier=2,
        ingestion_method="semantic_splitter",
    )


def _simple_chunk_fallback(text: str, title: str, chunk_size: int = 1000) -> DocumentTree:
    """
    Simple chunking fallback when LlamaIndex is not available.
    
    Splits text into fixed-size chunks.
    """
    logger.info("using_simple_chunking", chunk_size=chunk_size)
    
    root = DocumentNode(
        id="root",
        level=0,
        header=title,
    )
    
    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    
    # Group paragraphs into chunks
    current_chunk = ""
    chunk_num = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunk_num += 1
                section = DocumentNode(
                    id=f"sec_{chunk_num:03d}",
                    level=1,
                    header=_generate_synthetic_header(current_chunk, chunk_num),
                    content=current_chunk,
                )
                root.children.append(section)
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add final chunk
    if current_chunk:
        chunk_num += 1
        section = DocumentNode(
            id=f"sec_{chunk_num:03d}",
            level=1,
            header=_generate_synthetic_header(current_chunk, chunk_num),
            content=current_chunk,
        )
        root.children.append(section)
    
    return DocumentTree(
        title=title,
        root=root,
        total_nodes=chunk_num + 1,
        ingestion_tier=2,
        ingestion_method="semantic_splitter",
    )


def _generate_synthetic_header(text: str, section_num: int) -> str:
    """
    Generate a synthetic header from text content.
    
    Takes the first few meaningful words from the text.
    """
    # Get first sentence or first N words
    words = text.split()[:10]
    
    if not words:
        return f"Section {section_num}"
    
    header = " ".join(words)
    
    # Truncate at sentence end if present
    for punct in ".!?":
        if punct in header:
            header = header.split(punct)[0] + punct
            break
    
    # Ensure reasonable length
    if len(header) > 60:
        header = header[:57] + "..."
    
    return header
