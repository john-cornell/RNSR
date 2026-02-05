"""
Agent Graph - LangGraph State Machine for Document Navigation

Implements the Navigator Agent with full RLM (Recursive Language Model) support:

1. Decomposes queries into sub-questions (Section 2.2 - Recursive Loop)
2. Navigates the document tree (expand/traverse decisions)
3. Stores findings as pointers (Variable Stitching - Section 2.2)
4. Synthesizes final answer from stored pointers
5. Tree of Thoughts prompting (Section 7.2)
6. Recursive sub-LLM invocation for complex queries

Agent State follows Appendix C specification:
- question: Current question being answered
- sub_questions: Decomposed sub-questions (via LLM)
- current_node_id: Where we are in the document tree
- visited_nodes: Navigation history
- variables: Stored findings as $POINTER -> content
- pending_questions: Sub-questions not yet answered
- context: Accumulated context (pointers only!)
- answer: Final synthesized answer
- trace: Full retrieval trace for transparency
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict, cast

import structlog

from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.indexing.kv_store import KVStore
from rnsr.indexing.semantic_search import SemanticSearcher
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
    
    # Tree of Thoughts (ToT) state - Section 7.2
    nodes_to_visit: list[str] # Queue for parallel exploration
    scored_candidates: list[dict[str, Any]]  # [{node_id, score, reasoning}]
    backtrack_stack: list[str]  # Stack of parent node IDs for backtracking
    dead_ends: list[str]  # Nodes marked as dead ends
    top_k: int  # Number of top candidates to explore
    tot_selection_threshold: float  # Minimum probability for selection
    tot_dead_end_threshold: float   # Probability threshold for dead end
    
    # Variable stitching (pointers only!)
    variables: list[str]  # List of $POINTER names
    context: str  # Contains pointers, not full content
    
    # Output
    answer: str | None
    confidence: float
    
    # Question metadata (e.g., multiple-choice options)
    metadata: dict[str, Any]
    
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
    top_k: int = 3,
    metadata: dict[str, Any] | None = None,
    tot_selection_threshold: float = 0.4,
    tot_dead_end_threshold: float = 0.1,
) -> AgentState:
    """Create the initial agent state with ToT support."""
    return AgentState(
        question=question,
        sub_questions=[],
        pending_questions=[],
        current_sub_question=None,
        current_node_id=root_node_id,
        visited_nodes=[],
        navigation_path=[root_node_id],
        # Tree of Thoughts state
        nodes_to_visit=[],
        scored_candidates=[],
        backtrack_stack=[],
        dead_ends=[],
        top_k=top_k,
        tot_selection_threshold=tot_selection_threshold,
        tot_dead_end_threshold=tot_dead_end_threshold,
        # Variable stitching
        variables=[],
        context="",
        answer=None,
        confidence=0.0,
        metadata=metadata or {},
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
# Tree of Thoughts (ToT) Prompting - Section 7.2
# =============================================================================

# ToT System Prompt as specified in the research paper
TOT_SYSTEM_PROMPT = """You are a Deep Research Agent navigating a document tree.

You are currently at Node: {current_node_summary}
Children Nodes: {children_summaries}
Your Goal: {query}

EVALUATION TASK:
For each child node, estimate the probability (0.0 to 1.0) that it contains relevant information for the goal.

INSTRUCTIONS:
1. Evaluate: For each child node, analyze its summary and estimate relevance probability.
2. Be OPEN-MINDED: Select nodes with probability > {selection_threshold} (moderate evidence of relevance).
3. Look for matches: Prefer nodes with facts/entities mentioned in the query, but also consider broad thematic matches.
4. Balance PRECISION and RECALL: Do not prune branches too early. If unsure, include the node.
5. Plan: Select the top-{top_k} most promising nodes.
6. Reasoning: Explain briefly what SPECIFIC content in the summary makes it relevant.
7. Backtrack Signal: If NO child seems relevant (all probabilities < {dead_end_threshold}), report "DEAD_END".

OUTPUT FORMAT (JSON):
{{
    "evaluations": [
        {{"node_id": "...", "probability": 0.85, "reasoning": "Summary mentions X which directly relates to query about Y"}},
        {{"node_id": "...", "probability": 0.60, "reasoning": "Contains information about Z"}},
        ...
    ],
    "selected_nodes": ["node_id_1", "node_id_2", ...],
    "is_dead_end": false,
    "backtrack_reason": null
}}

If this is a dead end:
{{
    "evaluations": [...],
    "selected_nodes": [],
    "is_dead_end": true,
    "backtrack_reason": "None of the children appear to contain information about X."
}}

Respond ONLY with the JSON, no other text."""


def _format_node_summary(node: SkeletonNode) -> str:
    """Format a node's summary for the ToT prompt."""
    return f"[{node.node_id}] {node.header}: {node.summary or '(no summary)'}"


def _format_children_summaries(
    skeleton: dict[str, SkeletonNode],
    child_ids: list[str],
) -> str:
    """Format all children summaries for the ToT prompt."""
    if not child_ids:
        return "(no children - this is a leaf node)"
    
    lines = []
    for child_id in child_ids:
        child = skeleton.get(child_id)
        if child:
            lines.append(f"  - {_format_node_summary(child)}")
    
    return "\n".join(lines) if lines else "(no children found)"


def _verify_node_relevance(
    node_id: str,
    query: str,
    kv_store: Any,
    min_keyword_overlap: float = 0.2,
) -> tuple[bool, float]:
    """
    Verify that a node's full content is actually relevant to the query.
    
    This prevents ToT from selecting nodes based on misleading summaries.
    
    Args:
        node_id: The node ID to verify.
        query: The query to check relevance against.
        kv_store: Key-value store to fetch full content.
        min_keyword_overlap: Minimum fraction of query keywords that must appear.
        
    Returns:
        Tuple of (is_relevant, relevance_score).
    """
    if not kv_store or not query:
        return True, 1.0  # Can't verify without store or query
    
    content = kv_store.get(node_id) or ""
    if not content:
        return False, 0.0
    
    # Extract keywords from query (basic tokenization)
    query_lower = query.lower()
    stop_words = {
        "what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "or",
        "how", "when", "where", "who", "which", "that", "this", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "with", "from", "by", "on", "at", "as", "it", "its", "their", "they",
    }
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower)) - stop_words
    
    if not query_words:
        return True, 1.0  # No meaningful keywords
    
    # Check content for query keywords
    content_lower = content.lower()
    matched_keywords = sum(1 for kw in query_words if kw in content_lower)
    overlap_score = matched_keywords / len(query_words)
    
    is_relevant = overlap_score >= min_keyword_overlap
    
    return is_relevant, overlap_score


def evaluate_children_with_tot(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    top_k_override: int | None = None,
    kv_store: Any = None,
) -> dict[str, Any]:
    """
    Use Tree of Thoughts prompting to evaluate child nodes.
    
    This implements Section 7.2 of the research paper:
    "We use Tree of Thoughts (ToT) which explicitly encourages the model 
    to explore multiple branches of the index before committing to a path."
    
    Args:
        state: Current agent state.
        skeleton: Skeleton index.
        top_k_override: Optional override for adaptive exploration.
        kv_store: Optional key-value store for content verification.
        
    Returns:
        Dictionary with evaluations, selected_nodes, is_dead_end, and backtrack_reason.
    """
    node_id = state.get("current_node_id")
    if node_id is None:
        return {"evaluations": [], "selected_nodes": [], "is_dead_end": True, "backtrack_reason": "No current node"}
    
    node = skeleton.get(node_id)
    if node is None:
        return {"evaluations": [], "selected_nodes": [], "is_dead_end": True, "backtrack_reason": "Node not found"}
    
    # If no children, this is a leaf - expand instead of traverse
    if not node.child_ids:
        return {"evaluations": [], "selected_nodes": [], "is_dead_end": False, "backtrack_reason": None, "is_leaf": True}
    
    # Format the ToT prompt
    current_summary = _format_node_summary(node)
    children_summaries = _format_children_summaries(skeleton, node.child_ids)
    query = state.get("current_sub_question") or state.get("question", "")
    top_k = top_k_override if top_k_override is not None else state.get("top_k", 3)
    selection_threshold = state.get("tot_selection_threshold", 0.4)
    dead_end_threshold = state.get("tot_dead_end_threshold", 0.1)
    
    prompt = TOT_SYSTEM_PROMPT.format(
        current_node_summary=current_summary,
        children_summaries=children_summaries,
        query=query,
        top_k=top_k,
        selection_threshold=selection_threshold,
        dead_end_threshold=dead_end_threshold,
    )
    
    # Call the LLM
    try:
        from rnsr.llm import get_llm
        import json
        
        llm = get_llm()
        response = llm.complete(prompt)
        response_text = str(response).strip()
        
        # Parse JSON response with robust error handling
        try:
            # Handle potential markdown code blocks
            if "```" in response_text:
                match = re.search(r'\{[\s\S]*\}', response_text)
                if match:
                    response_text = match.group(0)

            result = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("tot_json_repair_attempt", original_response=response_text)
            # Fallback to asking the LLM to fix the JSON
            repair_prompt = f"""The following text is a malformed JSON object. Please fix it and return ONLY the corrected JSON. Do not add any commentary.

Malformed JSON:
{response_text}"""
            from rnsr.llm import get_llm

            llm = get_llm()
            repaired_response_text = str(llm.complete(repair_prompt)).strip()
            
            # Final attempt to parse the repaired JSON
            if "```" in repaired_response_text:
                 match = re.search(r'\{[\s\S]*\}', repaired_response_text)
                 if match:
                    repaired_response_text = match.group(0)

            result = json.loads(repaired_response_text)
        
        # CONTENT VERIFICATION: Verify selected nodes actually contain relevant content
        # This prevents ToT from being misled by misleading summaries
        if kv_store and result.get("selected_nodes"):
            query = state.get("current_sub_question") or state.get("question", "")
            original_selected = result["selected_nodes"]
            verified_selected = []
            unverified = []
            
            for node_id in original_selected:
                is_relevant, score = _verify_node_relevance(node_id, query, kv_store)
                if is_relevant:
                    verified_selected.append(node_id)
                else:
                    unverified.append({"node_id": node_id, "score": score})
            
            if unverified:
                logger.warning(
                    "tot_nodes_failed_content_verification",
                    original_count=len(original_selected),
                    verified_count=len(verified_selected),
                    unverified=unverified,
                )
            
            # Update result with verified nodes
            result["selected_nodes"] = verified_selected
            result["unverified_nodes"] = [u["node_id"] for u in unverified]
            
            # If all nodes failed verification, mark as potential dead end
            if not verified_selected and original_selected:
                logger.warning(
                    "tot_all_nodes_failed_verification",
                    original_nodes=original_selected,
                    query=query[:100],
                )
                # Don't mark as dead end immediately - fallback to exploring unverified
                # nodes since summaries might still be useful
                result["selected_nodes"] = original_selected[:2]  # Keep top 2 as fallback
                result["verification_warning"] = "All selected nodes failed content verification"
        
        logger.debug(
            "tot_evaluation_complete",
            node_id=node_id,
            evaluations=len(result.get("evaluations", [])),
            selected=result.get("selected_nodes", []),
            is_dead_end=result.get("is_dead_end", False),
        )
        
        return result
        
    except Exception as e:
        logger.warning("tot_evaluation_failed", error=str(e), node_id=node_id)
        
        # Fallback: use simple heuristic (select first unvisited children)
        visited = state.get("visited_nodes", [])
        dead_ends = state.get("dead_ends", [])
        
        unvisited = [
            cid for cid in node.child_ids 
            if cid not in visited and cid not in dead_ends
        ]
        
        return {
            "evaluations": [{"node_id": cid, "probability": 0.5, "reasoning": "Fallback selection"} for cid in unvisited[:top_k]],
            "selected_nodes": unvisited[:top_k],
            "is_dead_end": len(unvisited) == 0,
            "backtrack_reason": "No unvisited children" if len(unvisited) == 0 else None,
        }


def backtrack_to_parent(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
) -> AgentState:
    """
    Backtrack to parent node when current path is a dead end.
    
    Implements the backtracking logic from Section 7.2:
    "Backtrack: If a node yields no useful info, report 'Dead End' and return to parent."
    """
    new_state = cast(AgentState, dict(state))
    
    current_id = new_state["current_node_id"]
    
    # Mark current node as dead end
    if current_id and current_id not in new_state["dead_ends"]:
        new_state["dead_ends"].append(current_id)
    
    # Try to backtrack using the backtrack stack
    if new_state["backtrack_stack"]:
        parent_id = new_state["backtrack_stack"].pop()
        new_state["current_node_id"] = parent_id
        new_state["navigation_path"].append(parent_id)
        
        add_trace_entry(
            new_state,
            "navigation",
            f"Backtracked to {parent_id} (dead end at {current_id})",
            {"from": current_id, "to": parent_id, "reason": "dead_end"},
        )
        
        logger.debug("backtrack_success", from_node=current_id, to_node=parent_id)
    else:
        # No parent to backtrack to - we're done exploring
        add_trace_entry(
            new_state,
            "navigation",
            "Cannot backtrack - at root or stack empty",
            {"current": current_id},
        )
        logger.debug("backtrack_failed", reason="empty_stack")
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


# =============================================================================
# LangGraph Node Functions
# =============================================================================


# Decomposition prompt for RLM recursive sub-task generation
DECOMPOSITION_PROMPT = """You are analyzing a complex query to decompose it into sub-tasks.

Query: {query}

Document Structure (top-level sections):
{structure}

TASK: Decompose this query into specific sub-tasks that can be executed independently.

RULES:
1. Each sub-task should target a specific section or piece of information
2. Sub-tasks should be answerable by reading specific document sections
3. For comparison queries, create one sub-task per item being compared
4. For multi-hop queries, create sequential sub-tasks (find A, then use A to find B)
5. Maximum 5 sub-tasks to maintain efficiency

OUTPUT FORMAT (JSON):
{{
    "sub_tasks": [
        {{"id": 1, "task": "Find X in section Y", "target_section": "section_hint"}},
        {{"id": 2, "task": "Extract Z from W", "target_section": "section_hint"}}
    ],
    "synthesis_plan": "How to combine sub-task results into final answer"
}}

Return ONLY valid JSON."""


def decompose_query(state: AgentState) -> AgentState:
    """
    Decompose the main question into sub-questions using LLM.
    
    Implements Section 2.2 "The Recursive Loop":
    "The model's capability to divide a complex reasoning task into 
    smaller, manageable sub-tasks and invoke instances of itself to solve them."
    """
    new_state = cast(AgentState, dict(state))  # Copy
    question = new_state["question"]
    
    # Try LLM-based decomposition
    try:
        from rnsr.llm import get_llm
        import json
        
        llm = get_llm()
        
        # Get document structure for context
        # Note: skeleton is not available here, so we'll do basic decomposition
        structure_hint = "Document sections available for navigation"
        
        prompt = DECOMPOSITION_PROMPT.format(
            query=question,
            structure=structure_hint,
        )
        
        response = llm.complete(prompt)
        response_text = str(response).strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            decomposition = json.loads(json_match.group())
            sub_tasks = decomposition.get("sub_tasks", [])
            
            if sub_tasks:
                sub_questions = [t["task"] for t in sub_tasks]
                new_state["sub_questions"] = sub_questions
                new_state["pending_questions"] = sub_questions.copy()
                new_state["current_sub_question"] = sub_questions[0]
                
                add_trace_entry(
                    new_state,
                    "decomposition",
                    f"LLM decomposed into {len(sub_questions)} sub-tasks",
                    {
                        "sub_questions": sub_questions,
                        "synthesis_plan": decomposition.get("synthesis_plan", ""),
                    },
                )
                
                new_state["iteration"] = new_state["iteration"] + 1
                return new_state
    
    except Exception as e:
        logger.warning("llm_decomposition_failed", error=str(e))
    
    # Fallback: simple decomposition patterns
    sub_questions = _simple_decompose(question)
    new_state["sub_questions"] = sub_questions
    new_state["pending_questions"] = sub_questions.copy()
    new_state["current_sub_question"] = sub_questions[0] if sub_questions else question
    
    add_trace_entry(
        new_state,
        "decomposition",
        f"Decomposed into {len(sub_questions)} sub-questions (fallback)",
        {"sub_questions": sub_questions},
    )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def _simple_decompose(question: str) -> list[str]:
    """
    Simple pattern-based decomposition fallback.
    
    Handles common query patterns without LLM.
    """
    question_lower = question.lower()
    
    # Pattern: "Compare X and Y" or "X vs Y"
    compare_patterns = [
        r"compare\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+)",
        r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
    ]
    
    for pattern in compare_patterns:
        match = re.search(pattern, question_lower)
        if match:
            item1, item2 = match.groups()
            return [
                f"Find information about {item1.strip()}",
                f"Find information about {item2.strip()}",
            ]
    
    # Pattern: "What are the X in sections A, B, and C"
    multi_section = re.search(
        r"in\s+(?:sections?\s+)?(.+?,\s*.+?(?:,\s*.+)*)",
        question_lower,
    )
    if multi_section:
        sections = [s.strip() for s in multi_section.group(1).split(",")]
        return [f"Find relevant information in {s}" for s in sections[:5]]
    
    # Pattern: "List all X" or "Find all X" - may need iteration
    if re.search(r"(list|find|show)\s+all", question_lower):
        return [
            f"Search for all relevant sections",
            question,  # Original as synthesis query
        ]
    
    # Default: single question
    return [question]


def navigate_tree(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
) -> AgentState:
    """
    Navigate the document tree based on current sub-question.
    
    This function now handles the node visitation queue for multi-path exploration.
    """
    new_state = cast(AgentState, dict(state))

    # Add detailed logging to debug navigation loops
    logger.debug(
        "navigate_step_start",
        iteration=new_state["iteration"],
        current_node=new_state["current_node_id"],
        queue=new_state["nodes_to_visit"],
        visited=new_state["visited_nodes"],
    )

    # If current node is None, try to pop from the visit queue
    if new_state["current_node_id"] is None and new_state["nodes_to_visit"]:
        next_node_id = new_state["nodes_to_visit"].pop(0)
        new_state["current_node_id"] = next_node_id
        
        # Also add to navigation path
        if next_node_id not in new_state["navigation_path"]:
            new_state["navigation_path"].append(next_node_id)
            
        add_trace_entry(
            new_state,
            "navigation",
            f"Visiting next node from queue: {next_node_id}",
            {"queue_size": len(new_state["nodes_to_visit"])},
        )
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        add_trace_entry(new_state, "navigation", "No current node to navigate to and queue is empty")
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None:
        add_trace_entry(new_state, "navigation", f"Node {node_id} not found")
        return new_state
    
    # Don't add to visited here - let expand/traverse handle that
    # to avoid preventing expansion of newly queued nodes
    
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
) -> Literal["expand", "traverse", "backtrack", "done"]:
    """
    Decide whether to expand (fetch content), traverse (go to children), 
    backtrack (dead end), or finish.
    
    Uses Tree of Thoughts evaluation to make intelligent decisions.
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
    
    # Removed variable count limit - iteration limit is sufficient
    
    # If no children, must expand (leaf node)
    # But only if not already visited
    visited = state.get("visited_nodes", [])
    if not node.child_ids:
        if node_id in visited:
            # Already expanded this leaf, done with it
            return "done"
        return "expand"
    
    # Check if all children are visited or dead ends
    visited = state.get("visited_nodes", [])
    dead_ends = state.get("dead_ends", [])
    unvisited_children = [
        cid for cid in node.child_ids 
        if cid not in visited and cid not in dead_ends
    ]
    
    if not unvisited_children:
        # All children explored - backtrack or done
        if state.get("backtrack_stack"):
            return "backtrack"
        return "done"
    
    # Adaptive exploration: increase top_k if we haven't found much information yet
    base_top_k = state.get("top_k", 3)
    variables_found = len(state.get("variables", []))
    iteration = state.get("iteration", 0)
    
    if iteration > 5 and variables_found == 0:
        adaptive_top_k = min(len(node.child_ids), base_top_k * 3)
    elif iteration > 3 and variables_found < 2:
        adaptive_top_k = min(len(node.child_ids), base_top_k * 2)
    else:
        adaptive_top_k = base_top_k

    # Use ToT evaluation to decide
    tot_result = evaluate_children_with_tot(state, skeleton, top_k_override=adaptive_top_k)
    
    # Check for dead end signal from ToT
    if tot_result.get("is_dead_end", False):
        if state.get("backtrack_stack"):
            return "backtrack"
        return "done"
    
    # Check if ToT says this is a leaf (should expand)
    if tot_result.get("is_leaf", False):
        return "expand"
    
    # If ToT selected nodes, traverse
    if tot_result.get("selected_nodes"):
        return "traverse"
    
    # Fallback: expand if at deep level (H3+)
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
    
    # Check if already visited to prevent infinite loops
    if node_id in new_state.get("visited_nodes", []):
        add_trace_entry(
            new_state,
            "navigation", 
            f"Skipping already-visited node {node_id}",
            {"node_id": node_id},
        )
        new_state["current_node_id"] = None
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None:
        new_state["current_node_id"] = None
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    # Fetch full content
    content = kv_store.get(node_id)
    if content is None:
        add_trace_entry(new_state, "variable_stitching", f"No content for {node_id}")
        new_state["current_node_id"] = None
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    # Generate and store as variable
    pointer = generate_pointer_name(node.header)
    variable_store.assign(pointer, content, node_id)
    
    new_state["variables"].append(pointer)
    new_state["context"] += f"\nFound: {pointer} (from {node.header})"
    
    # Mark node as visited and clear current node
    if node_id not in new_state["visited_nodes"]:
        new_state["visited_nodes"].append(node_id)
    new_state["current_node_id"] = None  # Process queue next
    
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
    semantic_searcher: SemanticSearcher | None = None,
) -> AgentState:
    """
    TRAVERSE: Navigate to child nodes using Tree of Thoughts reasoning.
    
    Primary Strategy: Tree of Thoughts (ToT) with adaptive exploration.
    - Uses LLM reasoning to evaluate child nodes based on summaries
    - Dynamically increases exploration when insufficient variables found
    - Preserves document structure and context (per research paper Section 7.2)
    
    Optional: Semantic search as shortcut for finding entry points (Section 9.1)
    """
    new_state = cast(AgentState, dict(state))
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None or not node.child_ids:
        return new_state
    
    question = new_state["question"]
    base_top_k = new_state.get("top_k", 3)
    
    # Adaptive exploration: increase top_k if we haven't found much information yet
    # This addresses the original issue: "agent should be able to explore all nodes if it needs to"
    variables_found = len(new_state.get("variables", []))
    iteration = new_state.get("iteration", 0)
    
    # Dynamic top_k based on progress
    if iteration > 5 and variables_found == 0:
        # Not finding anything - expand search significantly
        adaptive_top_k = min(len(node.child_ids), base_top_k * 3)
        add_trace_entry(
            new_state,
            "navigation",
            f"Expanding search: {variables_found} variables after {iteration} iterations",
            {"base_top_k": base_top_k, "adaptive_top_k": adaptive_top_k},
        )
    elif iteration > 3 and variables_found < 2:
        # Finding very little - expand moderately
        adaptive_top_k = min(len(node.child_ids), base_top_k * 2)
    else:
        # Normal exploration
        adaptive_top_k = base_top_k
    
    # Use Tree of Thoughts as primary navigation method (per research paper)
    tot_result = evaluate_children_with_tot(new_state, skeleton, top_k_override=adaptive_top_k)
    
    # Check for dead end
    if tot_result.get("is_dead_end", False):
        add_trace_entry(
            new_state,
            "navigation",
            f"Dead end detected at {node_id}: {tot_result.get('backtrack_reason', 'Unknown')}",
            {"node_id": node_id, "backtrack_reason": tot_result.get("backtrack_reason")},
        )
        # Mark as dead end and don't change current node
        if node_id not in new_state["dead_ends"]:
            new_state["dead_ends"].append(node_id)
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    selected_nodes = tot_result.get("selected_nodes", [])
    evaluations = tot_result.get("evaluations", [])
    new_state["scored_candidates"] = evaluations

    # Queue selected children for visitation
    if selected_nodes:
        new_state["nodes_to_visit"].extend(selected_nodes)
        # Mark parent as visited since we've traversed its children
        if node_id not in new_state["visited_nodes"]:
            new_state["visited_nodes"].append(node_id)
        # Unset current node to force the next navigate step to pull from the queue
        new_state["current_node_id"] = None
        add_trace_entry(
            new_state,
            "navigation",
            f"Queued {len(selected_nodes)} children from {node_id}",
            {"nodes": selected_nodes, "parent": node_id},
        )
    else:
        # No candidates available - this should trigger backtracking
        add_trace_entry(
            new_state,
            "navigation",
            f"No unvisited candidates at {node_id}",
            {"node_id": node_id},
        )
    
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

            metadata = new_state.get("metadata", {})
            options = metadata.get("options")

            if options and isinstance(options, list) and len(options) > 0:
                # QuALITY multiple-choice question - STRICT GROUNDING MODE
                # Ensure each option is a string and format with letters
                options_text = "\n".join([f"{chr(65+i)}. {str(opt)}" for i, opt in enumerate(options)])
                prompt = f"""Answer this multiple-choice question using ONLY the provided context. Do NOT infer or guess.

STRICT GROUNDING RULES:
1. Select the option that is EXPLICITLY supported by the context
2. You MUST cite the exact text from the context that supports your choice
3. If NO option is explicitly supported, respond with: "Cannot determine from context"
4. Do NOT choose an option based on inference or general knowledge

Question: {question}

Options:
{options_text}

Context:
{context_text}

Instructions:
1. Read the context carefully
2. Find the EXACT text that supports one of the options
3. Respond with the letter, full option text, AND your supporting quote
4. Format: "X. [option text] (Source: 'exact quote from context')"

Your answer:"""
            else:
                # Standard open-ended question - STRICT GROUNDING MODE
                prompt = f"""Answer the question using ONLY the provided context. Do NOT infer, guess, or add information not explicitly stated.

STRICT GROUNDING RULES:
1. Answer ONLY using information explicitly stated in the context below
2. For EVERY claim you make, cite the source by quoting the exact text in quotation marks
3. Do NOT infer, deduce, or extrapolate beyond what is explicitly written
4. If the answer is NOT explicitly stated in the context, respond with: "I cannot answer this from the provided text."
5. If only a partial answer is available, state what IS explicitly mentioned and acknowledge what is missing

FORMAT YOUR ANSWER AS:
- Your answer with inline citations like: "The value is X" (Source: "exact quote from context")
- If multiple sources support a claim, cite each one

Question: {question}

Context:
{context_text}

Answer (cite sources for every claim, or state "I cannot answer this from the provided text"):"""

            response = llm.complete(prompt)
            answer = str(response).strip()
            
            # Normalize multiple-choice answers
            if options:
                answer_lower = answer.lower().strip()
                matched = False
                
                for i, opt in enumerate(options):
                    letter = chr(65 + i)  # A, B, C, D
                    opt_lower = opt.lower()
                    
                    # Match patterns: "A.", "A)", "(A)", "A. answer text"
                    if (answer_lower.startswith(f"{letter.lower()}.") or 
                        answer_lower.startswith(f"{letter.lower()})") or 
                        answer_lower.startswith(f"({letter.lower()})")):
                        answer = opt
                        matched = True
                        break
                    
                    # Exact match (case insensitive)
                    if answer_lower == opt_lower:
                        answer = opt
                        matched = True
                        break
                    
                    # Check if option text is contained in answer
                    if opt_lower in answer_lower or answer_lower in opt_lower:
                        # Use similarity to pick best match if multiple partial matches
                        answer = opt
                        matched = True
                        break
                
                new_state["confidence"] = 0.8 if matched else 0.1
            else:
                new_state["confidence"] = min(1.0, len(pointers) * 0.3)
            
            new_state["answer"] = answer

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
# RLM Recursive Execution Functions (Section 2.2)
# =============================================================================


def execute_sub_task_with_llm(
    sub_task: str,
    context: str,
    llm_fn: Any = None,
) -> str:
    """
    Execute a single sub-task using an LLM.
    
    Implements the recursive LLM pattern from Section 2.2:
    "For each contract, call the LLM API (invoking itself) with a 
    specific sub-prompt: 'Extract liability clause from this text.'"
    
    Args:
        sub_task: The sub-task description/prompt.
        context: The context to process.
        llm_fn: Optional LLM function. If None, uses default.
        
    Returns:
        LLM response as string.
    """
    if llm_fn is None:
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
            llm_fn = lambda p: str(llm.complete(p))
        except Exception as e:
            logger.warning("llm_not_available", error=str(e))
            return f"[Error: LLM not available - {str(e)}]"
    
    prompt = f"""{sub_task}

Context:
{context}

Response:"""
    
    try:
        return llm_fn(prompt)
    except Exception as e:
        logger.error("sub_task_execution_failed", error=str(e))
        return f"[Error: {str(e)}]"


def batch_execute_sub_tasks(
    sub_tasks: list[str],
    contexts: list[str],
    batch_size: int = 5,
    max_parallel: int = 4,
) -> list[str]:
    """
    Execute multiple sub-tasks in parallel batches.
    
    Implements Section 2.3 "Optimization via Batching":
    "Instead of making 1,000 individual calls to summarize 1,000 paragraphs,
    the RLM writes code to group paragraphs into chunks of 5 and processes 
    them in parallel threads."
    
    Args:
        sub_tasks: List of sub-task prompts.
        contexts: List of contexts for each sub-task.
        batch_size: Items per batch (default 5).
        max_parallel: Max parallel threads (default 4).
        
    Returns:
        List of results for each sub-task.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if len(sub_tasks) != len(contexts):
        raise ValueError("sub_tasks and contexts must have same length")
    
    if not sub_tasks:
        return []
    
    logger.info(
        "batch_execution_start",
        num_tasks=len(sub_tasks),
        batch_size=batch_size,
    )
    
    # Get LLM function
    try:
        from rnsr.llm import get_llm
        llm = get_llm()
        llm_fn = lambda p: str(llm.complete(p))
    except Exception as e:
        logger.error("batch_llm_failed", error=str(e))
        return [f"[Error: {str(e)}]"] * len(sub_tasks)
    
    results: list[str] = [""] * len(sub_tasks)
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_idx: dict[Any, int] = {}
        
        for idx, (task, ctx) in enumerate(zip(sub_tasks, contexts)):
            future = executor.submit(execute_sub_task_with_llm, task, ctx, llm_fn)
            future_to_idx[future] = idx
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=120)
            except Exception as e:
                results[idx] = f"[Error: {str(e)}]"
    
    logger.info("batch_execution_complete", num_results=len(results))
    return results


def process_pending_questions(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
) -> AgentState:
    """
    Process all pending sub-questions using recursive LLM calls.
    
    This implements the full RLM recursive loop:
    1. For each pending question, find relevant context
    2. Invoke sub-LLM to extract answer
    3. Store result as variable
    4. Move to next question
    """
    new_state = cast(AgentState, dict(state))
    pending = new_state.get("pending_questions", [])
    
    if not pending:
        return new_state
    
    current_question = pending[0]
    
    # Find relevant content for this question
    # Use current node and its children
    node_id = new_state.get("current_node_id") or "root"
    node = skeleton.get(node_id)
    
    if node is None:
        # Pop and continue
        new_state["pending_questions"] = pending[1:]
        return new_state
    
    # Get context from current node and children
    context_parts = []
    
    # Add current node content
    current_content = kv_store.get(node_id) if node_id else None
    if current_content:
        context_parts.append(f"[{node.header}]\n{current_content}")
    
    # Add children summaries
    for child_id in node.child_ids[:5]:  # Limit to 5 children
        child = skeleton.get(child_id)
        if child:
            child_content = kv_store.get(child_id)
            if child_content:
                context_parts.append(f"[{child.header}]\n{child_content[:2000]}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Execute sub-task with LLM
    result = execute_sub_task_with_llm(
        sub_task=f"Answer this question: {current_question}",
        context=context,
    )
    
    # Store as variable
    pointer = generate_pointer_name(current_question[:30])
    variable_store.assign(pointer, result, node_id or "root")
    new_state["variables"].append(pointer)
    
    add_trace_entry(
        new_state,
        "decomposition",
        f"Processed sub-question via recursive LLM",
        {
            "question": current_question,
            "pointer": pointer,
            "context_length": len(context),
        },
    )
    
    # Pop processed question
    new_state["pending_questions"] = pending[1:]
    if pending[1:]:
        new_state["current_sub_question"] = pending[1]
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


# =============================================================================
# Graph Builder (LangGraph)
# =============================================================================


def build_navigator_graph(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    semantic_searcher: SemanticSearcher | None = None,
) -> Any:
    """
    Build the LangGraph state machine for document navigation.
    
    Implements Tree of Thoughts (ToT) navigation with backtracking support.
    
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
        lambda state: traverse_to_children(
            cast(AgentState, state),
            skeleton,
            semantic_searcher,
        ),
    )
    
    # Add backtrack node for ToT dead-end handling
    graph.add_node(
        "backtrack",
        lambda state: backtrack_to_parent(cast(AgentState, state), skeleton),
    )
    
    graph.add_node(
        "synthesize",
        lambda state: synthesize_answer(cast(AgentState, state), variable_store),
    )
    
    # Add edges
    graph.add_edge("decompose", "navigate")
    
    # Conditional edge based on expand/traverse/backtrack decision
    graph.add_conditional_edges(
        "navigate",
        lambda s: should_expand(s, skeleton),
        {
            "expand": "expand",
            "traverse": "traverse",
            "backtrack": "backtrack",
            "done": "synthesize",
        },
    )
    
    # After expand, traverse, or backtrack, always return to the main navigation
    # handler, which will decide what to do next (e.g., pull from queue)
    graph.add_edge("expand", "navigate")
    graph.add_edge("traverse", "navigate")
    graph.add_edge("backtrack", "navigate")
    
    graph.add_edge("synthesize", END)
    
    # Set entry point
    graph.set_entry_point("decompose")
    
    logger.info("navigator_graph_built", features=["ToT", "backtracking"])
    
    return graph.compile()


# =============================================================================
# High-Level API
# =============================================================================


def run_navigator(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    max_iterations: int = 20,
    top_k: int | None = None,
    use_semantic_search: bool = True,
    semantic_searcher: SemanticSearcher | None = None,
    metadata: dict[str, Any] | None = None,
    tot_selection_threshold: float = 0.4,
    tot_dead_end_threshold: float = 0.1,
) -> dict[str, Any]:
    """
    Run the navigator agent on a question.
    
    Args:
        question: User's question.
        skeleton: Skeleton index.
        kv_store: KV store with full content.
        max_iterations: Maximum navigation iterations.
        top_k: Number of top children to explore (default: auto-detect based on tree depth).
        use_semantic_search: Use semantic search (O(log N)) instead of ToT evaluation (O(N)).
            Allows exploring ALL leaf nodes ranked by relevance, preventing missed data.
        semantic_searcher: Optional pre-built semantic searcher. If None and use_semantic_search=True, creates one.
        tot_selection_threshold: Minimum probability for ToT node selection (0.0-1.0).
        tot_dead_end_threshold: Probability threshold for declaring a dead end (0.0-1.0).
        
    Returns:
        Dictionary with answer, confidence, trace.
        
    Example:
        result = run_navigator(
            "What are the liability terms?",
            skeleton,
            kv_store,
            use_semantic_search=True,  # Enable semantic search
        )
        print(result["answer"])
    """
    # Get root node
    root_id = None
    root_node = None
    for node in skeleton.values():
        if node.level == 0:
            root_id = node.node_id
            root_node = node
            break
    
    if root_id is None:
        return {
            "answer": "Error: No root node found in skeleton index.",
            "confidence": 0.0,
            "trace": [],
        }
    
    # Auto-detect top_k based on tree structure
    if top_k is None:
        num_root_children = len(root_node.child_ids) if root_node else 0
        if num_root_children > 10:
            # Flat hierarchy (e.g., QuALITY): explore more children
            top_k = min(10, num_root_children)
        else:
            # Deep hierarchy (e.g., PDFs): explore fewer
            top_k = 3
    
    # Semantic search disabled by default per research paper Section 9.1:
    # "Hybridize Search: Give the agent a vector_search tool as a SHORTCUT.
    # The agent can use vector search to find a starting node and then
    # SWITCH TO TREE TRAVERSAL for local exploration."
    # 
    # Primary navigation uses ToT reasoning-based retrieval.
    if use_semantic_search and semantic_searcher is None:
        try:
            from rnsr.indexing.semantic_search import create_semantic_searcher
            semantic_searcher = create_semantic_searcher(skeleton, kv_store)
            logger.info(
                "semantic_search_optional_tool",
                nodes=len(skeleton),
                note="Available as shortcut for entry points only",
            )
        except Exception as e:
            logger.warning(
                "semantic_search_unavailable",
                error=str(e),
            )
            semantic_searcher = None
    
    logger.info(
        "using_tot_reasoning_navigation",
        method="Tree of Thoughts",
        adaptive_exploration=True,
        note="LLM reasons about document structure to navigate",
    )
    
    # Build and run graph
    graph = build_navigator_graph(skeleton, kv_store, semantic_searcher)
    
    initial_state = create_initial_state(
        question=question,
        root_node_id=root_id,
        max_iterations=max_iterations,
        top_k=top_k,
        metadata=metadata,
        tot_selection_threshold=tot_selection_threshold,
        tot_dead_end_threshold=tot_dead_end_threshold,
    )
    
    final_state = graph.invoke(initial_state)
    
    return {
        "answer": final_state.get("answer", ""),
        "confidence": final_state.get("confidence", 0.0),
        "trace": final_state.get("trace", []),
        "variables_used": final_state.get("variables", []),
        "nodes_visited": final_state.get("visited_nodes", []),
    }
