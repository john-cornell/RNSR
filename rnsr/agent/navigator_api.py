"""
Navigator API - REPL Environment for Tree-Based Retrieval

Implements Section 5.1 Phase III of the research paper:
"The agent does not interact with the index via vector search. 
It interacts via a Python REPL pre-loaded with a Navigator API."

The Navigator API provides:
- list_children(node_id): Returns summaries of immediate children
- read_node(node_id): Returns full text content of a node  
- search_index(query): Keyword/hybrid search for entry points

This creates a "Reasoning via Planning" (RAP) loop where the agent:
1. Planning: Calls list_children('root') to see high-level structure
2. Decomposition: Breaks query into sub-tasks
3. Recursive Execution: Navigates tree, stores findings as variables
4. Variable Accumulation: Stores text in REPL variables (clause_2023, etc.)
5. Synthesis: Compares variables and generates final response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


@dataclass
class NavigatorAPI:
    """
    Navigator API for tree-based document retrieval.
    
    This is the "REPL Environment" from Section 5.1 Phase III.
    The agent interacts with the document tree via these methods
    rather than direct vector search.
    
    Example usage (from research paper):
        ```
        # Planning
        nav.list_children('root')  
        # -> ["2023 Agreement", "2024 Agreement"]
        
        # Navigation
        nav.list_children('2023_agreement')
        # -> ["Legal Terms", "Payment Terms", ...]
        
        # Reading
        clause_2023 = nav.read_node('legal_terms_indemnification')
        clause_2024 = nav.read_node('2024_legal_terms_indemnification')
        
        # Synthesis
        # Agent compares clause_2023 vs clause_2024
        ```
    """
    
    skeleton: dict[str, SkeletonNode]
    kv_store: KVStore
    variable_store: VariableStore
    
    # Navigation state
    current_node_id: str = "root"
    path_history: list[str] = field(default_factory=list)
    accumulated_context: dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with root in path history."""
        if "root" in self.skeleton:
            self.path_history.append("root")
    
    def list_children(self, node_id: str | None = None) -> list[dict[str, Any]]:
        """
        Returns the summaries of immediate children.
        
        Per Section 5.1: "list_children(node_id): Returns the 
        summaries of immediate children."
        
        Args:
            node_id: Node to list children of. Defaults to current node.
            
        Returns:
            List of dicts with id, header, summary, and has_children.
            
        Example:
            >>> nav.list_children('root')
            [
                {'id': 'sec_001', 'header': '2023 Agreement', 
                 'summary': 'Master agreement...', 'has_children': True},
                {'id': 'sec_002', 'header': '2024 Agreement',
                 'summary': 'Amended agreement...', 'has_children': True},
            ]
        """
        node_id = node_id or self.current_node_id
        node = self.skeleton.get(node_id)
        
        if node is None:
            logger.warning("node_not_found", node_id=node_id)
            return []
        
        children = []
        for child_id in node.child_ids:
            child = self.skeleton.get(child_id)
            if child:
                children.append({
                    "id": child.node_id,
                    "header": child.header,
                    "summary": child.summary,
                    "level": child.level,
                    "has_children": len(child.child_ids) > 0,
                })
        
        logger.debug(
            "list_children",
            node_id=node_id,
            num_children=len(children),
        )
        
        return children
    
    def read_node(self, node_id: str) -> str:
        """
        Returns the full text content of a node.
        
        Per Section 5.1: "read_node(node_id): Returns the full 
        text content of a node."
        
        This fetches from the KV store (not the skeleton summaries).
        
        Args:
            node_id: Node to read content from.
            
        Returns:
            Full text content of the node.
            
        Example:
            >>> content = nav.read_node('indemnification_clause')
            >>> print(content)
            "The indemnifying party shall hold harmless..."
        """
        content = self.kv_store.get(node_id)
        
        if content is None:
            # Try getting from skeleton summary as fallback
            node = self.skeleton.get(node_id)
            if node:
                content = node.summary
            else:
                logger.warning("content_not_found", node_id=node_id)
                return ""
        
        logger.debug(
            "read_node",
            node_id=node_id,
            content_length=len(content),
        )
        
        return content
    
    def search_index(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Keyword/hybrid search to find initial entry points in the tree.
        
        Per Section 5.1: "search_index(query): A keyword/hybrid 
        search to find initial entry points in the tree."
        
        This is useful for jumping directly to relevant sections
        without traversing from root.
        
        Args:
            query: Search query (keywords).
            top_k: Maximum number of results.
            
        Returns:
            List of matching nodes with scores.
            
        Example:
            >>> nav.search_index("indemnification")
            [
                {'id': 'sec_003_01', 'header': 'Indemnification',
                 'score': 0.95, 'path': 'root > Legal Terms > Indemnification'},
            ]
        """
        query_lower = query.lower()
        results = []
        
        for node_id, node in self.skeleton.items():
            # Simple keyword matching (in production, use BM25 or hybrid)
            score = 0.0
            
            # Check header match
            if query_lower in node.header.lower():
                score += 0.6
            
            # Check summary match
            if node.summary and query_lower in node.summary.lower():
                score += 0.4
            
            if score > 0:
                # Build path
                path = self._get_path_to_node(node_id)
                
                results.append({
                    "id": node_id,
                    "header": node.header,
                    "summary": node.summary[:200] if node.summary else "",
                    "level": node.level,
                    "score": score,
                    "path": " > ".join(path),
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        logger.debug(
            "search_index",
            query=query,
            num_results=len(results),
        )
        
        return results
    
    def navigate_to(self, node_id: str) -> dict[str, Any]:
        """
        Navigate to a specific node and make it the current node.
        
        Args:
            node_id: Node to navigate to.
            
        Returns:
            Node info dict if successful.
        """
        node = self.skeleton.get(node_id)
        if node is None:
            logger.warning("navigate_failed", node_id=node_id)
            return {"error": f"Node {node_id} not found"}
        
        self.current_node_id = node_id
        self.path_history.append(node_id)
        
        return {
            "id": node.node_id,
            "header": node.header,
            "summary": node.summary,
            "level": node.level,
            "children": self.list_children(node_id),
        }
    
    def go_back(self) -> dict[str, Any]:
        """
        Navigate back to previous node in history.
        
        Returns:
            Previous node info, or current if at start.
        """
        if len(self.path_history) > 1:
            self.path_history.pop()
            self.current_node_id = self.path_history[-1]
        
        return self.navigate_to(self.current_node_id)
    
    def store_variable(
        self,
        name: str,
        content: str,
        source_node_id: str = "",
    ) -> str:
        """
        Store content as a named variable for later synthesis.
        
        Per Section 5.1: "Variable Accumulation: The text found 
        is stored in REPL variables (clause_2023, clause_2024)."
        
        Args:
            name: Variable name (e.g., "clause_2023").
            content: Content to store.
            source_node_id: Source node for traceability.
            
        Returns:
            Pointer name (e.g., "$CLAUSE_2023").
        """
        if not name.startswith("$"):
            name = "$" + name.upper().replace(" ", "_")
        
        self.variable_store.assign(name, content, source_node_id)
        self.accumulated_context[name] = content
        
        logger.info("variable_stored", name=name, length=len(content))
        
        return name
    
    def get_variable(self, name: str) -> str | None:
        """
        Retrieve a stored variable by name.
        
        Args:
            name: Variable name (with or without $).
            
        Returns:
            Stored content, or None if not found.
        """
        if not name.startswith("$"):
            name = "$" + name.upper()
        
        return self.variable_store.resolve(name)
    
    def compare_variables(self, *names: str) -> dict[str, str]:
        """
        Get multiple variables for comparison.
        
        Per Section 5.1: "Synthesis: The agent uses the LLM to 
        compare the content of these two variables."
        
        Args:
            *names: Variable names to compare.
            
        Returns:
            Dict mapping names to their content.
        """
        result = {}
        for name in names:
            if not name.startswith("$"):
                name = "$" + name.upper()
            content = self.variable_store.resolve(name)
            result[name] = content if content else "[Not found]"
        
        return result
    
    def list_variables(self) -> list[str]:
        """List all stored variable names."""
        return [v.pointer for v in self.variable_store.list_variables()]
    
    def get_current_location(self) -> dict[str, Any]:
        """Get info about current node and navigation state."""
        node = self.skeleton.get(self.current_node_id)
        if node is None:
            return {"error": "Current node not found"}
        
        return {
            "current_node": {
                "id": node.node_id,
                "header": node.header,
                "summary": node.summary,
                "level": node.level,
            },
            "path": self.path_history,
            "variables_stored": len(self.accumulated_context),
        }
    
    def _get_path_to_node(self, target_id: str) -> list[str]:
        """Build the path from root to a node."""
        path = []
        
        # Simple BFS to find path
        def find_path(current_id: str, target: str, current_path: list[str]) -> list[str] | None:
            if current_id == target:
                return current_path + [self.skeleton[current_id].header]
            
            node = self.skeleton.get(current_id)
            if node is None:
                return None
            
            for child_id in node.child_ids:
                result = find_path(child_id, target, current_path + [node.header])
                if result:
                    return result
            
            return None
        
        if "root" in self.skeleton:
            path = find_path("root", target_id, []) or [target_id]
        else:
            path = [target_id]
        
        return path


def create_navigator(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore | None = None,
) -> NavigatorAPI:
    """
    Create a Navigator API instance.
    
    This is the main factory function for creating the REPL environment.
    
    Args:
        skeleton: Skeleton index from build_skeleton_index().
        kv_store: KV store with full content.
        variable_store: Optional variable store (created if None).
        
    Returns:
        Configured NavigatorAPI instance.
        
    Example:
        from rnsr import ingest_document, build_skeleton_index
        from rnsr.agent.navigator_api import create_navigator
        
        result = ingest_document("contract.pdf")
        skeleton, kv_store = build_skeleton_index(result.tree)
        nav = create_navigator(skeleton, kv_store)
        
        # Now use nav.list_children(), nav.read_node(), etc.
    """
    if variable_store is None:
        variable_store = VariableStore()
    
    return NavigatorAPI(
        skeleton=skeleton,
        kv_store=kv_store,
        variable_store=variable_store,
    )


# Convenience function for RAP-style queries
def execute_rap_query(
    nav: NavigatorAPI,
    query: str,
    llm_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    Execute a Reasoning-via-Planning (RAP) query.
    
    Implements the RAP loop from Section 5.1:
    1. Planning: list_children('root')
    2. Decomposition: (done by LLM)
    3. Recursive Execution: navigate and read
    4. Variable Accumulation: store findings
    5. Synthesis: compare and answer
    
    Args:
        nav: Navigator API instance.
        query: User query.
        llm_fn: Optional LLM function for synthesis.
        
    Returns:
        Dict with answer, variables, and trace.
    """
    trace = []
    
    # Step 1: Planning - see top-level structure
    root_children = nav.list_children("root")
    trace.append({
        "step": "planning",
        "action": "list_children('root')",
        "result": [c["header"] for c in root_children],
    })
    
    # Step 2: Search for relevant sections
    search_results = nav.search_index(query)
    trace.append({
        "step": "search",
        "action": f"search_index('{query}')",
        "result": [r["header"] for r in search_results[:3]],
    })
    
    # Step 3: Read top matching nodes
    variables = []
    for result in search_results[:3]:
        node_id = result["id"]
        content = nav.read_node(node_id)
        if content:
            var_name = nav.store_variable(
                result["header"].replace(" ", "_"),
                content,
                node_id,
            )
            variables.append(var_name)
            trace.append({
                "step": "read",
                "action": f"read_node('{node_id}')",
                "stored_as": var_name,
            })
    
    # Step 4: Synthesize answer
    if llm_fn and variables:
        context = "\n\n".join(
            f"=== {v} ===\n{nav.get_variable(v)}"
            for v in variables
        )
        prompt = f"Based on the following context, answer: {query}\n\n{context}"
        answer = llm_fn(prompt)
    else:
        answer = f"Found {len(variables)} relevant sections: {variables}"
    
    return {
        "answer": answer,
        "variables": variables,
        "trace": trace,
    }
