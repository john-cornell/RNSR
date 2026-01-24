"""
Tree Builder - Document Tree Assembly

This module builds a hierarchical document tree from classified spans.
Uses a stack-based parser to:
1. Process blocks in reading order (top-to-bottom, left-to-right)
2. Assign body text to the nearest preceding header
3. Output nested DocumentNode structure

The tree structure enables:
- Hierarchical navigation by the agent
- Section-based summarization
- Context-aware retrieval
"""

from __future__ import annotations

from uuid import uuid4

import structlog

from rnsr.models import ClassifiedSpan, DocumentNode, DocumentTree

logger = structlog.get_logger(__name__)


class TreeBuilder:
    """
    Builds a hierarchical document tree from classified spans.
    
    Uses a stack-based approach where:
    - Headers push new nodes onto the stack (at appropriate level)
    - Body text appends to the current node's content
    - The stack maintains the current path in the hierarchy
    """

    def __init__(self):
        """Initialize the Tree Builder."""
        self._current_page = -1

    def build_tree(
        self,
        spans: list[ClassifiedSpan],
        document_title: str = "",
    ) -> DocumentTree:
        """
        Build a document tree from classified spans.
        
        Args:
            spans: List of ClassifiedSpan (headers and body text).
            document_title: Optional title for the document.
            
        Returns:
            DocumentTree with hierarchical structure.
        """
        if not spans:
            logger.warning("empty_spans_list")
            root = DocumentNode(id="root", level=0, header="Document")
            return DocumentTree(
                title=document_title or "Untitled",
                root=root,
                total_nodes=1,
            )
        
        # Sort spans by reading order (page, then top-to-bottom, left-to-right)
        sorted_spans = self._sort_by_reading_order(spans)
        
        # Initialize root node
        root = DocumentNode(
            id="root",
            level=0,
            header=document_title or self._extract_title(sorted_spans),
        )
        
        # Stack tracks current path: [(level, node), ...]
        # Start with root at level 0
        stack: list[tuple[int, DocumentNode]] = [(0, root)]
        
        # Process each span
        for span in sorted_spans:
            if span.role == "header":
                self._process_header(span, stack)
            else:
                self._process_body(span, stack)
        
        # Count total nodes
        total_nodes = self._count_nodes(root)
        
        logger.info(
            "tree_built",
            total_nodes=total_nodes,
            max_depth=self._get_max_depth(root),
        )
        
        return DocumentTree(
            title=root.header,
            root=root,
            total_nodes=total_nodes,
        )

    def _sort_by_reading_order(
        self, 
        spans: list[ClassifiedSpan],
    ) -> list[ClassifiedSpan]:
        """
        Sort spans by reading order: page number, then y position, then x position.
        """
        return sorted(
            spans,
            key=lambda s: (
                s.page_num,
                s.bbox.y0,  # Top to bottom
                s.bbox.x0,  # Left to right
            ),
        )

    def _extract_title(self, spans: list[ClassifiedSpan]) -> str:
        """
        Extract document title from the first H1 header.
        """
        for span in spans:
            if span.role == "header" and span.header_level == 1:
                return span.text.strip()
        
        # Fallback: use first header of any level
        for span in spans:
            if span.role == "header":
                return span.text.strip()
        
        return "Untitled Document"

    def _process_header(
        self,
        span: ClassifiedSpan,
        stack: list[tuple[int, DocumentNode]],
    ) -> None:
        """
        Process a header span by adding a new node to the tree.
        
        The stack is adjusted so that:
        - Headers pop nodes until finding a parent with lower level
        - Then push the new header node
        """
        level = span.header_level
        
        # Pop stack until we find a parent with level < this header
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        
        # Create new node for this header
        new_node = DocumentNode(
            id=f"sec_{str(uuid4())[:6]}",
            level=level,
            header=span.text.strip(),
            page_num=span.page_num,
            bbox=span.bbox,
        )
        
        # Add as child of current top of stack
        parent_node = stack[-1][1]
        parent_node.children.append(new_node)
        
        # Push onto stack
        stack.append((level, new_node))
        
        logger.debug(
            "header_added",
            level=level,
            header=span.text[:50],
            parent=parent_node.header[:30] if parent_node.header else "root",
        )

    def _process_body(
        self,
        span: ClassifiedSpan,
        stack: list[tuple[int, DocumentNode]],
    ) -> None:
        """
        Process body text by appending to the current node's content.
        """
        if not stack:
            return
        
        current_node = stack[-1][1]
        
        # Append text with appropriate spacing
        if current_node.content:
            # Check if we need a new paragraph (different page or significant y gap)
            current_node.content += " " + span.text.strip()
        else:
            current_node.content = span.text.strip()
        
        # Update page number if not set
        if current_node.page_num is None:
            current_node.page_num = span.page_num

    def _count_nodes(self, node: DocumentNode) -> int:
        """Recursively count all nodes in the tree."""
        count = 1  # Count this node
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node: DocumentNode, current_depth: int = 0) -> int:
        """Get the maximum depth of the tree."""
        if not node.children:
            return current_depth
        
        return max(
            self._get_max_depth(child, current_depth + 1)
            for child in node.children
        )


def build_document_tree(
    spans: list[ClassifiedSpan],
    title: str = "",
) -> DocumentTree:
    """
    Convenience function to build a document tree from classified spans.
    
    Args:
        spans: List of ClassifiedSpan from header classification.
        title: Optional document title.
        
    Returns:
        DocumentTree with hierarchical structure.
        
    Example:
        analysis, raw_spans = analyze_font_histogram("doc.pdf")
        classified = classify_headers(raw_spans, analysis)
        tree = build_document_tree(classified)
    """
    builder = TreeBuilder()
    return builder.build_tree(spans, title)


def tree_to_dict(tree: DocumentTree) -> dict:
    """
    Convert a DocumentTree to a nested dictionary for serialization.
    
    This produces the JSON structure specified in the plan.
    """
    def node_to_dict(node: DocumentNode) -> dict:
        return {
            "id": node.id,
            "type": "section" if node.level > 0 else "document",
            "level": node.level,
            "header": node.header,
            "content": node.content,
            "page_num": node.page_num,
            "children": [node_to_dict(child) for child in node.children],
        }
    
    return {
        "id": tree.id,
        "title": tree.title,
        "total_nodes": tree.total_nodes,
        "ingestion_tier": tree.ingestion_tier,
        "ingestion_method": tree.ingestion_method,
        "root": node_to_dict(tree.root),
    }
