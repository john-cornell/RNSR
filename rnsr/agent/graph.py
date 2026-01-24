"""
Agent Graph - LangGraph State Machine for Document Navigation

Implements the Navigator Agent that:
1. Decomposes queries into sub-questions
2. Navigates the document tree (expand/traverse decisions)
3. Stores findings as pointers (Variable Stitching)
4. Synthesizes final answer from stored pointers

Agent State follows Appendix C specification:
- question: Current question being answered
- sub_questions: Decomposed sub-questions
- current_node_id: Where we are in the document tree
- visited_nodes: Navigation history
- variables: Stored findings as $POINTER -> content
- pending_questions: Sub-questions not yet answered
- context: Accumulated context (pointers only!)
- answer: Final synthesized answer
- trace: Full retrieval trace for transparency
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict, cast

import structlog

from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode, TraceEntry

logger = structlog.get_logger(__name__)


# =============================================================================
# Agent State Definition (Appendix C)
# =============================================================================


class AgentState(TypedDict):
    """
    State for the RNSR Navigator Agent.
    
    All fields use pointer-based Variable Stitching:
    - Full content stored in VariableStore
    - Only $POINTER names in state fields
    """
    
    # Query processing
    question: str
    sub_questions: list[str]
    pending_questions: list[str]
    current_sub_question: str | None
    
    # Navigation state
    current_node_id: str | None
    visited_nodes: list[str]
    navigation_path: list[str]
    
    # Variable stitching (pointers only!)
    variables: list[str]  # List of $POINTER names
    context: str  # Contains pointers, not full content
    
    # Output
    answer: str | None
    confidence: float
    
    # Traceability
    trace: list[dict[str, Any]]
    iteration: int
    max_iterations: int


# =============================================================================
# Navigator Tools
# =============================================================================


def create_navigator_tools(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
) -> dict[str, Any]:
    """
    Create tools for the Navigator Agent.
    
    Args:
        skeleton: Skeleton index (node_id -> SkeletonNode).
        kv_store: KV store with full content.
        variable_store: Variable store for findings.
        
    Returns:
        Dictionary of tool functions.
    """
    
    def get_node_summary(node_id: str) -> str:
        """Get the summary of a skeleton node."""
        node = skeleton.get(node_id)
        if node is None:
            return f"Node {node_id} not found"
        
        children_info = ""
        if node.child_ids:
            children = [skeleton.get(cid) for cid in node.child_ids]
            children_info = "\nChildren:\n" + "\n".join(
                f"  - {c.node_id}: {c.header}" for c in children if c
            )
        
        return f"""
Node: {node.node_id}
Header: {node.header}
Level: {node.level}
Summary: {node.summary}
{children_info}
"""
    
    def navigate_to_child(node_id: str, child_id: str) -> str:
        """Navigate to a child node."""
        node = skeleton.get(node_id)
        if node is None:
            return f"Node {node_id} not found"
        
        if child_id not in node.child_ids:
            return f"{child_id} is not a child of {node_id}"
        
        child = skeleton.get(child_id)
        if child is None:
            return f"Child {child_id} not found"
        
        return get_node_summary(child_id)
    
    def expand_node(node_id: str) -> str:
        """
        EXPAND: Fetch full content and store as variable.
        
        Use when the summary answers the question.
        """
        node = skeleton.get(node_id)
        if node is None:
            return f"Node {node_id} not found"
        
        # Fetch full content from KV store
        content = kv_store.get(node_id)
        if content is None:
            return f"No content found for {node_id}"
        
        # Generate pointer name
        pointer = generate_pointer_name(node.header)
        
        # Store as variable
        variable_store.assign(pointer, content, node_id)
        
        logger.info(
            "node_expanded",
            node_id=node_id,
            pointer=pointer,
            chars=len(content),
        )
        
        return f"Stored content as {pointer} ({len(content)} chars)"
    
    def store_finding(
        pointer_name: str,
        content: str,
        source_node_id: str = "",
    ) -> str:
        """
        Store a finding as a pointer variable.
        
        Args:
            pointer_name: Name like $LIABILITY_CLAUSE (must start with $)
            content: Full text content to store
            source_node_id: Source node for traceability
        """
        if not pointer_name.startswith("$"):
            pointer_name = "$" + pointer_name.upper()
        
        try:
            meta = variable_store.assign(pointer_name, content, source_node_id)
            return f"Stored as {pointer_name} ({meta.char_count} chars)"
        except Exception as e:
            return f"Error storing: {e}"
    
    def compare_variables(*pointers: str) -> str:
        """
        Compare multiple stored variables.
        
        Resolves pointers and returns content for comparison.
        Use during synthesis phase only.
        """
        results = []
        for pointer in pointers:
            content = variable_store.resolve(pointer)
            if content:
                results.append(f"=== {pointer} ===\n{content}")
            else:
                results.append(f"=== {pointer} ===\n[Not found]")
        
        return "\n\n".join(results)
    
    def list_stored_variables() -> str:
        """List all stored variables with metadata."""
        variables = variable_store.list_variables()
        if not variables:
            return "No variables stored yet."
        
        lines = ["Stored Variables:"]
        for v in variables:
            lines.append(f"  {v.pointer}: {v.char_count} chars from {v.source_node_id}")
        
        return "\n".join(lines)
    
    def synthesize_from_variables() -> str:
        """
        Get all stored content for final synthesis.
        
        Call this at the end to get full content for answer generation.
        """
        variables = variable_store.list_variables()
        if not variables:
            return "No variables to synthesize from."
        
        parts = []
        for v in variables:
            content = variable_store.resolve(v.pointer)
            if content:
                parts.append(f"=== {v.pointer} ===\n{content}")
        
        return "\n\n".join(parts)
    
    return {
        "get_node_summary": get_node_summary,
        "navigate_to_child": navigate_to_child,
        "expand_node": expand_node,
        "store_finding": store_finding,
        "compare_variables": compare_variables,
        "list_stored_variables": list_stored_variables,
        "synthesize_from_variables": synthesize_from_variables,
    }


# =============================================================================
# State Management Functions
# =============================================================================


def create_initial_state(
    question: str,
    root_node_id: str,
    max_iterations: int = 20,
) -> AgentState:
    """Create the initial agent state."""
    return AgentState(
        question=question,
        sub_questions=[],
        pending_questions=[],
        current_sub_question=None,
        current_node_id=root_node_id,
        visited_nodes=[],
        navigation_path=[root_node_id],
        variables=[],
        context="",
        answer=None,
        confidence=0.0,
        trace=[],
        iteration=0,
        max_iterations=max_iterations,
    )


def add_trace_entry(
    state: AgentState,
    node_type: Literal["decomposition", "navigation", "variable_stitching", "synthesis"],
    action: str,
    details: dict | None = None,
) -> None:
    """Add a trace entry to the state."""
    entry = TraceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        node_type=node_type,
        action=action,
        details=details or {},
    )
    state["trace"].append(entry.model_dump())


# =============================================================================
# LangGraph Node Functions
# =============================================================================


def decompose_query(state: AgentState) -> AgentState:
    """
    Decompose the main question into sub-questions.
    
    This is typically done by an LLM, but here's a simple version.
    """
    new_state = cast(AgentState, dict(state))  # Copy
    
    # Simple decomposition (in production, use LLM)
    question = new_state["question"]
    
    # For now, just use the main question
    new_state["sub_questions"] = [question]
    new_state["pending_questions"] = [question]
    new_state["current_sub_question"] = question
    
    add_trace_entry(
        new_state,
        "decomposition",
        f"Decomposed into {len(new_state['sub_questions'])} sub-questions",
        {"sub_questions": new_state["sub_questions"]},
    )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def navigate_tree(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
) -> AgentState:
    """
    Navigate the document tree based on current sub-question.
    
    Makes expand/traverse decisions based on node summaries.
    """
    new_state = cast(AgentState, dict(state))
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        add_trace_entry(new_state, "navigation", "No current node")
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None:
        add_trace_entry(new_state, "navigation", f"Node {node_id} not found")
        return new_state
    
    # Add to visited
    if node_id not in new_state["visited_nodes"]:
        new_state["visited_nodes"].append(node_id)
    
    add_trace_entry(
        new_state,
        "navigation",
        f"At node {node_id}: {node.header}",
        {"summary": node.summary, "children": node.child_ids},
    )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def should_expand(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
) -> Literal["expand", "traverse", "done"]:
    """
    Decide whether to expand (fetch content) or traverse (go to children).
    
    In production, this uses an LLM to evaluate if the summary answers
    the question. Here's a simple version based on node properties.
    """
    node_id = state.get("current_node_id")
    if node_id is None:
        return "done"
    
    node = skeleton.get(node_id)
    if node is None:
        return "done"
    
    # Check iteration limit
    if state.get("iteration", 0) >= state.get("max_iterations", 20):
        return "done"
    
    # If no children, must expand
    if not node.child_ids:
        return "expand"
    
    # Simple heuristic: expand if at leaf-ish level (H3)
    if node.level >= 3:
        return "expand"
    
    # Otherwise traverse to children
    return "traverse"


def expand_current_node(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
) -> AgentState:
    """
    EXPAND: Fetch full content and store as variable.
    """
    new_state = cast(AgentState, dict(state))
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None:
        return new_state
    
    # Fetch full content
    content = kv_store.get(node_id)
    if content is None:
        add_trace_entry(new_state, "variable_stitching", f"No content for {node_id}")
        return new_state
    
    # Generate and store as variable
    pointer = generate_pointer_name(node.header)
    variable_store.assign(pointer, content, node_id)
    
    new_state["variables"].append(pointer)
    new_state["context"] += f"\nFound: {pointer} (from {node.header})"
    
    add_trace_entry(
        new_state,
        "variable_stitching",
        f"Stored {pointer}",
        {"node_id": node_id, "header": node.header, "chars": len(content)},
    )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def traverse_to_children(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
) -> AgentState:
    """
    TRAVERSE: Navigate to child nodes.
    """
    new_state = cast(AgentState, dict(state))
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None or not node.child_ids:
        return new_state
    
    # For simplicity, go to first unvisited child
    for child_id in node.child_ids:
        if child_id not in new_state["visited_nodes"]:
            new_state["current_node_id"] = child_id
            new_state["navigation_path"].append(child_id)
            
            add_trace_entry(
                new_state,
                "navigation",
                f"Traversed to {child_id}",
                {"from": node_id, "to": child_id},
            )
            break
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def synthesize_answer(
    state: AgentState,
    variable_store: VariableStore,
) -> AgentState:
    """
    Synthesize final answer from stored variables using an LLM.
    
    Uses the configured LLM to generate a concise answer
    from the resolved variable content.
    """
    new_state = cast(AgentState, dict(state))
    
    # Resolve all variables
    pointers = new_state["variables"]
    
    if not pointers:
        new_state["answer"] = "No relevant content found."
        new_state["confidence"] = 0.0
    else:
        # Collect all content
        contents = []
        for pointer in pointers:
            content = variable_store.resolve(pointer)
            if content:
                contents.append(content)
        
        context_text = "\n\n---\n\n".join(contents)
        question = new_state["question"]
        
        # Use LLM to synthesize answer
        try:
            from rnsr.llm import get_llm
            
            llm = get_llm()
            
            prompt = f"""Based on the following context, answer the question concisely.
If the answer cannot be determined from the context, say "Cannot determine from available context."

Question: {question}

Context:
{context_text}

Answer (be concise and direct):"""
            
            response = llm.complete(prompt)
            new_state["answer"] = str(response).strip()
            new_state["confidence"] = min(1.0, len(pointers) * 0.3)
            
        except Exception as e:
            logger.warning("llm_synthesis_failed", error=str(e))
            # Fallback to context concatenation
            new_state["answer"] = f"Context found:\n{context_text}"
            new_state["confidence"] = min(1.0, len(pointers) * 0.2)
    
    add_trace_entry(
        new_state,
        "synthesis",
        "Generated answer from variables",
        {"variables_used": pointers, "confidence": new_state["confidence"]},
    )
    
    return new_state


# =============================================================================
# Graph Builder (LangGraph)
# =============================================================================


def build_navigator_graph(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
) -> Any:
    """
    Build the LangGraph state machine for document navigation.
    
    Returns a compiled graph that can be invoked with a question.
    
    Usage:
        graph = build_navigator_graph(skeleton, kv_store)
        result = graph.invoke({"question": "What are the payment terms?"})
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        raise ImportError(
            "LangGraph not installed. Install with: pip install langgraph"
        )
    
    variable_store = VariableStore()
    
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("decompose", decompose_query)
    
    graph.add_node(
        "navigate",
        lambda state: navigate_tree(cast(AgentState, state), skeleton),
    )
    
    graph.add_node(
        "expand",
        lambda state: expand_current_node(cast(AgentState, state), skeleton, kv_store, variable_store),
    )
    
    graph.add_node(
        "traverse",
        lambda state: traverse_to_children(cast(AgentState, state), skeleton),
    )
    
    graph.add_node(
        "synthesize",
        lambda state: synthesize_answer(cast(AgentState, state), variable_store),
    )
    
    # Add edges
    graph.add_edge("decompose", "navigate")
    
    # Conditional edge based on expand/traverse decision
    graph.add_conditional_edges(
        "navigate",
        lambda s: should_expand(s, skeleton),
        {
            "expand": "expand",
            "traverse": "traverse",
            "done": "synthesize",
        },
    )
    
    graph.add_edge("expand", "navigate")
    graph.add_edge("traverse", "navigate")
    graph.add_edge("synthesize", END)
    
    # Set entry point
    graph.set_entry_point("decompose")
    
    logger.info("navigator_graph_built")
    
    return graph.compile()


# =============================================================================
# High-Level API
# =============================================================================


def run_navigator(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """
    Run the navigator agent on a question.
    
    Args:
        question: User's question.
        skeleton: Skeleton index.
        kv_store: KV store with full content.
        max_iterations: Maximum navigation iterations.
        
    Returns:
        Dictionary with answer, confidence, trace.
        
    Example:
        result = run_navigator(
            "What are the liability terms?",
            skeleton,
            kv_store,
        )
        print(result["answer"])
    """
    # Get root node
    root_id = None
    for node in skeleton.values():
        if node.level == 0:
            root_id = node.node_id
            break
    
    if root_id is None:
        return {
            "answer": "Error: No root node found in skeleton index.",
            "confidence": 0.0,
            "trace": [],
        }
    
    # Build and run graph
    graph = build_navigator_graph(skeleton, kv_store)
    
    initial_state = create_initial_state(
        question=question,
        root_node_id=root_id,
        max_iterations=max_iterations,
    )
    
    final_state = graph.invoke(initial_state)
    
    return {
        "answer": final_state.get("answer", ""),
        "confidence": final_state.get("confidence", 0.0),
        "trace": final_state.get("trace", []),
        "variables_used": final_state.get("variables", []),
        "nodes_visited": final_state.get("visited_nodes", []),
    }
