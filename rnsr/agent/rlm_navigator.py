"""
RLM Navigator - Recursive Language Model Navigator with Full REPL Integration

This module implements the full RLM (Recursive Language Model) pattern from the
arxiv paper "Recursive Language Models" combined with RNSR's tree-based retrieval.

Key Features:
1. Full REPL environment with code execution for document filtering
2. Pre-LLM filtering using regex/keyword search before ToT evaluation
3. Deep recursive sub-LLM calls (configurable depth)
4. Answer verification loops
5. Async parallel sub-LLM processing

This is the state-of-the-art combination of:
- PageIndex: Vectorless, reasoning-based tree search
- RLMs: REPL environment with recursive sub-LLM calls
- RNSR: Latent hierarchy reconstruction + variable stitching
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

import structlog

from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode, TraceEntry

logger = structlog.get_logger(__name__)


# =============================================================================
# RLM Configuration
# =============================================================================


@dataclass
class RLMConfig:
    """Configuration for the RLM Navigator."""
    
    # Recursion control
    max_recursion_depth: int = 3  # Max depth for recursive sub-LLM calls
    max_iterations: int = 30  # Max navigation iterations
    
    # Tree of Thoughts parameters
    top_k: int = 3  # Base children to explore
    selection_threshold: float = 0.4  # Min probability for selection
    dead_end_threshold: float = 0.1  # Threshold for dead end
    
    # Pre-filtering
    enable_pre_filtering: bool = True  # Use regex/keyword filtering before ToT
    pre_filter_min_matches: int = 1  # Min keyword matches to include node
    
    # REPL execution
    enable_code_execution: bool = True  # Allow LLM to write/execute code
    max_code_execution_time: int = 30  # Seconds
    
    # Answer verification
    enable_verification: bool = True  # Verify answers with sub-LLM
    verification_retries: int = 2  # Max verification attempts
    
    # Async processing
    enable_async: bool = True  # Use async for parallel sub-LLM calls
    max_concurrent_calls: int = 5  # Max parallel LLM calls
    
    # Vision mode
    enable_vision: bool = False  # Use vision LLM for page images
    vision_model: str = "gemini-2.5-flash"  # Vision model to use


# =============================================================================
# Pre-Filtering Engine (Before ToT Evaluation)
# =============================================================================


class PreFilterEngine:
    """
    Pre-filters nodes before expensive ToT LLM evaluation.
    
    Implements the key RLM insight: use code (regex, keywords) to filter
    before sending to LLM. This dramatically reduces LLM calls.
    
    Example:
        # Query: "What is the liability clause?"
        # Instead of evaluating all 50 children with LLM:
        # 1. Extract keywords: ["liability", "clause", "indemnification"]
        # 2. Regex search children summaries
        # 3. Only send matching children to ToT evaluation
    """
    
    def __init__(self, config: RLMConfig):
        self.config = config
        self._keyword_cache: dict[str, list[str]] = {}
    
    def extract_keywords(self, query: str) -> list[str]:
        """Extract searchable keywords from a query."""
        if query in self._keyword_cache:
            return self._keyword_cache[query]
        
        # Remove stop words and extract content words
        stop_words = {
            "what", "is", "the", "a", "an", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "and",
            "but", "if", "or", "because", "as", "until", "while", "this", "that",
            "these", "those", "find", "show", "list", "describe", "explain", "tell",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Add quoted phrases as single keywords
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted)
        
        # Add capitalized words (likely proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        keywords.extend([pn.lower() for pn in proper_nouns])
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        self._keyword_cache[query] = unique_keywords
        logger.debug("keywords_extracted", query=query[:50], keywords=unique_keywords)
        return unique_keywords
    
    def filter_nodes_by_keywords(
        self,
        nodes: list[SkeletonNode],
        keywords: list[str],
        min_matches: int | None = None,
    ) -> tuple[list[SkeletonNode], list[SkeletonNode]]:
        """
        Filter nodes by keyword matching.
        
        Returns:
            Tuple of (matching_nodes, remaining_nodes)
        """
        if not self.config.enable_pre_filtering:
            return nodes, []
        
        if not keywords:
            return nodes, []
        
        min_matches = min_matches or self.config.pre_filter_min_matches
        
        matching = []
        remaining = []
        
        for node in nodes:
            # Search in header and summary
            search_text = f"{node.header} {node.summary}".lower()
            
            matches = sum(1 for kw in keywords if kw in search_text)
            
            if matches >= min_matches:
                matching.append(node)
            else:
                remaining.append(node)
        
        logger.debug(
            "pre_filter_complete",
            total=len(nodes),
            matching=len(matching),
            remaining=len(remaining),
            keywords=keywords[:5],
        )
        
        return matching, remaining
    
    def regex_search_nodes(
        self,
        nodes: list[SkeletonNode],
        pattern: str,
    ) -> list[tuple[SkeletonNode, list[str]]]:
        """
        Search nodes using regex pattern.
        
        Returns:
            List of (node, matches) tuples.
        """
        results = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning("invalid_regex_pattern", pattern=pattern, error=str(e))
            return results
        
        for node in nodes:
            search_text = f"{node.header}\n{node.summary}"
            matches = regex.findall(search_text)
            if matches:
                results.append((node, matches))
        
        return results


# =============================================================================
# Deep Recursive Sub-LLM Engine
# =============================================================================


class RecursiveSubLLMEngine:
    """
    Enables true multi-level recursive sub-LLM calls.
    
    Unlike single-level decomposition, this allows sub-LLMs to spawn
    their own sub-LLMs up to a configurable depth.
    
    Example:
        Query: "Compare the liability clauses in 2023 vs 2024 contracts"
        
        Depth 0 (Root): Decompose into sub-tasks
        ├── Depth 1: "Find 2023 liability clause"
        │   └── Depth 2: "Extract specific terms"
        └── Depth 1: "Find 2024 liability clause"
            └── Depth 2: "Extract specific terms"
    """
    
    def __init__(
        self,
        config: RLMConfig,
        llm_fn: Callable[[str], str] | None = None,
    ):
        self.config = config
        self._llm_fn = llm_fn
        self._call_count = 0
        self._depth_stats: dict[int, int] = {}
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function for sub-calls."""
        self._llm_fn = llm_fn
    
    def recursive_call(
        self,
        prompt: str,
        context: str,
        depth: int = 0,
        allow_sub_calls: bool = True,
    ) -> str:
        """
        Execute a recursive LLM call.
        
        Args:
            prompt: The task/question for the LLM.
            context: Context to process.
            depth: Current recursion depth.
            allow_sub_calls: Whether this call can spawn sub-calls.
            
        Returns:
            LLM response.
        """
        if self._llm_fn is None:
            return "[ERROR: LLM function not configured]"
        
        if depth >= self.config.max_recursion_depth:
            allow_sub_calls = False
            logger.debug("max_recursion_depth_reached", depth=depth)
        
        # Track stats
        self._call_count += 1
        self._depth_stats[depth] = self._depth_stats.get(depth, 0) + 1
        
        # Build the prompt with recursion capability
        if allow_sub_calls:
            system_instruction = f"""You are a sub-LLM at recursion depth {depth}.
You can decompose complex tasks into sub-tasks.
If you need to process multiple items independently, list them as:
SUB_TASK[1]: <task description>
SUB_TASK[2]: <task description>
...
These will be processed by sub-LLMs and results aggregated.
"""
        else:
            system_instruction = f"""You are a sub-LLM at max recursion depth {depth}.
Provide a direct answer without further decomposition."""
        
        full_prompt = f"""{system_instruction}

Task: {prompt}

Context:
{context}

Response:"""
        
        try:
            response = self._llm_fn(full_prompt)
            
            # Check for sub-task declarations and process them
            if allow_sub_calls and "SUB_TASK[" in response:
                response = self._process_sub_tasks(response, depth + 1)
            
            return response
            
        except Exception as e:
            logger.error("recursive_call_failed", depth=depth, error=str(e))
            return f"[ERROR: {str(e)}]"
    
    def _process_sub_tasks(self, response: str, depth: int) -> str:
        """Process SUB_TASK declarations in the response."""
        # Extract sub-tasks
        sub_tasks = re.findall(r'SUB_TASK\[(\d+)\]:\s*(.+?)(?=SUB_TASK\[|$)', response, re.DOTALL)
        
        if not sub_tasks:
            return response
        
        logger.debug("processing_sub_tasks", count=len(sub_tasks), depth=depth)
        
        # Process each sub-task recursively
        results = []
        for idx, (task_num, task_desc) in enumerate(sub_tasks):
            result = self.recursive_call(
                prompt=task_desc.strip(),
                context="(inherited from parent)",
                depth=depth,
                allow_sub_calls=(depth < self.config.max_recursion_depth),
            )
            results.append(f"Result[{task_num}]: {result}")
        
        # Synthesize results
        synthesis_prompt = f"""Synthesize the following sub-task results into a coherent answer:

{chr(10).join(results)}

Original task: {response.split('SUB_TASK[')[0].strip()}

Synthesized answer:"""
        
        return self._llm_fn(synthesis_prompt) if self._llm_fn else "\n".join(results)
    
    async def async_recursive_call(
        self,
        prompt: str,
        context: str,
        depth: int = 0,
    ) -> str:
        """Async version of recursive_call for parallel processing."""
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recursive_call(prompt, context, depth),
        )
    
    def batch_recursive_calls(
        self,
        prompts: list[str],
        contexts: list[str],
        depth: int = 0,
    ) -> list[str]:
        """
        Execute multiple recursive calls in parallel.
        
        Uses ThreadPoolExecutor for parallel processing.
        """
        if len(prompts) != len(contexts):
            raise ValueError("prompts and contexts must have same length")
        
        if not prompts:
            return []
        
        results: list[str] = [""] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_calls) as executor:
            futures = {}
            for idx, (prompt, context) in enumerate(zip(prompts, contexts)):
                future = executor.submit(
                    self.recursive_call,
                    prompt,
                    context,
                    depth,
                )
                futures[future] = idx
            
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=60)
                except Exception as e:
                    results[idx] = f"[ERROR: {str(e)}]"
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get call statistics."""
        return {
            "total_calls": self._call_count,
            "calls_by_depth": dict(self._depth_stats),
        }


# =============================================================================
# Answer Verification Engine
# =============================================================================


class AnswerVerificationEngine:
    """
    Verifies answers using sub-LLM calls.
    
    Implements the RLM pattern of using sub-LLMs to verify answers
    before returning, ensuring higher accuracy.
    """
    
    def __init__(
        self,
        config: RLMConfig,
        llm_fn: Callable[[str], str] | None = None,
    ):
        self.config = config
        self._llm_fn = llm_fn
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self._llm_fn = llm_fn
    
    def verify_answer(
        self,
        question: str,
        proposed_answer: str,
        evidence: list[str],
        attempt: int = 0,
    ) -> dict[str, Any]:
        """
        Verify an answer using sub-LLM evaluation.
        
        Returns:
            Dict with 'is_valid', 'confidence', 'issues', 'improved_answer'.
        """
        if not self.config.enable_verification:
            return {
                "is_valid": True,
                "confidence": 0.7,
                "issues": [],
                "improved_answer": proposed_answer,
            }
        
        if self._llm_fn is None:
            return {
                "is_valid": True,
                "confidence": 0.5,
                "issues": ["LLM not configured for verification"],
                "improved_answer": proposed_answer,
            }
        
        evidence_text = "\n---\n".join(evidence) if evidence else "(no evidence provided)"
        
        verification_prompt = f"""You are a fact-checker verifying an answer.

Question: {question}

Proposed Answer: {proposed_answer}

Evidence:
{evidence_text}

VERIFICATION TASK:
1. Check if the answer is supported by the evidence
2. Check if the answer directly addresses the question
3. Check for any factual errors or unsupported claims

OUTPUT FORMAT (JSON):
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of issues if any"],
    "improved_answer": "corrected answer if needed, or null if valid"
}}

Respond ONLY with JSON:"""
        
        try:
            import json
            
            response = self._llm_fn(verification_prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                
                # If not valid and we have retries left, try to improve
                if not result.get("is_valid", True) and attempt < self.config.verification_retries:
                    logger.debug(
                        "answer_verification_failed",
                        attempt=attempt,
                        issues=result.get("issues", []),
                    )
                    
                    # Try with improved answer
                    if result.get("improved_answer"):
                        return self.verify_answer(
                            question,
                            result["improved_answer"],
                            evidence,
                            attempt + 1,
                        )
                
                return result
            else:
                logger.warning("verification_json_parse_failed", response=response[:200])
                return {
                    "is_valid": True,
                    "confidence": 0.5,
                    "issues": ["Could not parse verification response"],
                    "improved_answer": proposed_answer,
                }
                
        except Exception as e:
            logger.error("verification_failed", error=str(e))
            return {
                "is_valid": True,
                "confidence": 0.3,
                "issues": [str(e)],
                "improved_answer": proposed_answer,
            }


# =============================================================================
# Enhanced RLM Navigator Agent State
# =============================================================================


class RLMAgentState:
    """
    State for the RLM Navigator Agent.
    
    Extends the base AgentState with RLM-specific fields.
    """
    
    def __init__(
        self,
        question: str,
        root_node_id: str,
        config: RLMConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.question = question
        self.config = config or RLMConfig()
        self.metadata = metadata or {}
        
        # Navigation state
        self.current_node_id: str | None = root_node_id
        self.visited_nodes: list[str] = []
        self.navigation_path: list[str] = [root_node_id]
        self.nodes_to_visit: list[str] = []
        self.dead_ends: list[str] = []
        self.backtrack_stack: list[str] = []
        
        # Variable stitching
        self.variables: list[str] = []
        self.context: str = ""
        
        # Sub-questions (RLM decomposition)
        self.sub_questions: list[str] = []
        self.pending_questions: list[str] = []
        self.current_sub_question: str | None = None
        
        # Pre-filtering state
        self.extracted_keywords: list[str] = []
        self.pre_filtered_nodes: dict[str, list[str]] = {}  # node_id -> matched keywords
        
        # Recursion tracking
        self.current_recursion_depth: int = 0
        self.recursion_call_count: int = 0
        
        # Output
        self.answer: str | None = None
        self.confidence: float = 0.0
        self.verification_result: dict[str, Any] | None = None
        
        # Traceability
        self.trace: list[dict[str, Any]] = []
        self.iteration: int = 0
    
    def add_trace(
        self,
        node_type: str,
        action: str,
        details: dict | None = None,
    ) -> None:
        """Add a trace entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_type": node_type,
            "action": action,
            "details": details or {},
            "iteration": self.iteration,
        }
        self.trace.append(entry)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "variables": self.variables,
            "visited_nodes": self.visited_nodes,
            "iteration": self.iteration,
            "recursion_call_count": self.recursion_call_count,
            "verification_result": self.verification_result,
            "trace": self.trace,
        }


# =============================================================================
# RLM Navigator - Main Class
# =============================================================================


class RLMNavigator:
    """
    The RLM Navigator combines:
    1. PageIndex-style tree search with reasoning
    2. RLM-style REPL environment with code execution
    3. RNSR-style variable stitching and skeleton indexing
    
    This is the unified, state-of-the-art document retrieval agent.
    """
    
    def __init__(
        self,
        skeleton: dict[str, SkeletonNode],
        kv_store: KVStore,
        config: RLMConfig | None = None,
    ):
        self.skeleton = skeleton
        self.kv_store = kv_store
        self.config = config or RLMConfig()
        
        # Initialize components
        self.variable_store = VariableStore()
        self.pre_filter = PreFilterEngine(self.config)
        self.recursive_engine = RecursiveSubLLMEngine(self.config)
        self.verification_engine = AnswerVerificationEngine(self.config)
        
        # LLM function
        self._llm_fn: Callable[[str], str] | None = None
        
        # Find root node
        self.root_id = self._find_root_id()
    
    def _find_root_id(self) -> str:
        """Find the root node ID."""
        for node in self.skeleton.values():
            if node.level == 0:
                return node.node_id
        raise ValueError("No root node found in skeleton")
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Configure the LLM function for all components."""
        self._llm_fn = llm_fn
        self.recursive_engine.set_llm_function(llm_fn)
        self.verification_engine.set_llm_function(llm_fn)
    
    def navigate(
        self,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Navigate the document tree to answer a question.
        
        This is the main entry point for the RLM Navigator.
        
        Args:
            question: The user's question.
            metadata: Optional metadata (e.g., multiple choice options).
            
        Returns:
            Dict with answer, confidence, trace, etc.
        """
        # Initialize state
        state = RLMAgentState(
            question=question,
            root_node_id=self.root_id,
            config=self.config,
            metadata=metadata,
        )
        
        # Ensure LLM is configured
        if self._llm_fn is None:
            self._configure_default_llm()
        
        logger.info("rlm_navigation_started", question=question[:100])
        
        try:
            # Phase 1: Pre-filtering with keyword extraction
            state = self._phase_pre_filter(state)
            
            # Phase 2: Query decomposition
            state = self._phase_decompose(state)
            
            # Phase 3: Tree navigation with ToT
            state = self._phase_navigate(state)
            
            # Phase 4: Synthesis
            state = self._phase_synthesize(state)
            
            # Phase 5: Verification (if enabled)
            if self.config.enable_verification:
                state = self._phase_verify(state)
            
            logger.info(
                "rlm_navigation_complete",
                confidence=state.confidence,
                variables=len(state.variables),
                iterations=state.iteration,
            )
            
            return state.to_dict()
            
        except Exception as e:
            logger.error("rlm_navigation_failed", error=str(e))
            state.answer = f"Error during navigation: {str(e)}"
            state.confidence = 0.0
            return state.to_dict()
    
    def _configure_default_llm(self) -> None:
        """Configure the default LLM if none set."""
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
            self.set_llm_function(lambda p: str(llm.complete(p)))
        except Exception as e:
            logger.warning("default_llm_config_failed", error=str(e))
    
    def _phase_pre_filter(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 1: Extract keywords and pre-filter nodes."""
        state.add_trace("pre_filter", "Extracting keywords from query")
        
        # Extract keywords
        keywords = self.pre_filter.extract_keywords(state.question)
        state.extracted_keywords = keywords
        
        if not keywords:
            state.add_trace("pre_filter", "No keywords extracted, skipping pre-filter")
            return state
        
        # Pre-filter all leaf nodes
        all_nodes = list(self.skeleton.values())
        matching, remaining = self.pre_filter.filter_nodes_by_keywords(all_nodes, keywords)
        
        # Store which nodes matched which keywords
        for node in matching:
            search_text = f"{node.header} {node.summary}".lower()
            matched_keywords = [kw for kw in keywords if kw in search_text]
            state.pre_filtered_nodes[node.node_id] = matched_keywords
        
        state.add_trace(
            "pre_filter",
            f"Pre-filtered {len(matching)}/{len(all_nodes)} nodes",
            {"keywords": keywords, "matching_nodes": len(matching)},
        )
        
        return state
    
    def _phase_decompose(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 2: Decompose query into sub-questions."""
        state.add_trace("decomposition", "Analyzing query for decomposition")
        
        if self._llm_fn is None:
            state.sub_questions = [state.question]
            state.pending_questions = [state.question]
            return state
        
        # Use LLM to decompose
        decomposition_prompt = f"""Analyze this query and decompose it into specific sub-tasks.

Query: {state.question}

Available document sections (pre-filtered matches):
{chr(10).join(f"- {self.skeleton[nid].header}" for nid in list(state.pre_filtered_nodes.keys())[:10])}

RULES:
1. Each sub-task should target a specific piece of information
2. For comparison queries, create one sub-task per item
3. Maximum 5 sub-tasks
4. If the query is simple, return just one sub-task

OUTPUT FORMAT (JSON):
{{
    "sub_tasks": ["task1", "task2", ...],
    "synthesis_plan": "how to combine results"
}}

Respond with JSON only:"""
        
        try:
            import json
            
            response = self._llm_fn(decomposition_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                result = json.loads(json_match.group())
                sub_tasks = result.get("sub_tasks", [state.question])
                state.sub_questions = sub_tasks
                state.pending_questions = sub_tasks.copy()
                state.current_sub_question = sub_tasks[0] if sub_tasks else state.question
                
                state.add_trace(
                    "decomposition",
                    f"Decomposed into {len(sub_tasks)} sub-tasks",
                    {"sub_tasks": sub_tasks},
                )
            else:
                state.sub_questions = [state.question]
                state.pending_questions = [state.question]
                
        except Exception as e:
            logger.warning("decomposition_failed", error=str(e))
            state.sub_questions = [state.question]
            state.pending_questions = [state.question]
        
        return state
    
    def _phase_navigate(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 3: Navigate the tree using ToT with pre-filtering."""
        state.add_trace("navigation", "Starting tree navigation")
        
        # Initialize navigation at root
        state.current_node_id = self.root_id
        
        while state.iteration < self.config.max_iterations:
            state.iteration += 1
            
            # Check termination conditions
            if state.current_node_id is None and not state.nodes_to_visit:
                break
            
            # Pop from queue if needed
            if state.current_node_id is None and state.nodes_to_visit:
                state.current_node_id = state.nodes_to_visit.pop(0)
            
            if state.current_node_id is None:
                break
            
            node = self.skeleton.get(state.current_node_id)
            if node is None:
                state.current_node_id = None
                continue
            
            # Already visited?
            if state.current_node_id in state.visited_nodes:
                state.current_node_id = None
                continue
            
            # Decide: expand or traverse
            action = self._decide_action(state, node)
            
            if action == "expand":
                state = self._do_expand(state, node)
            elif action == "traverse":
                state = self._do_traverse(state, node)
            elif action == "backtrack":
                state = self._do_backtrack(state)
            else:
                break
        
        state.add_trace(
            "navigation",
            f"Navigation complete after {state.iteration} iterations",
            {"variables_found": len(state.variables)},
        )
        
        return state
    
    def _decide_action(
        self,
        state: RLMAgentState,
        node: SkeletonNode,
    ) -> Literal["expand", "traverse", "backtrack", "done"]:
        """Decide what action to take at current node."""
        # Leaf node -> expand
        if not node.child_ids:
            if node.node_id in state.visited_nodes:
                return "done"
            return "expand"
        
        # Check unvisited children
        unvisited = [
            cid for cid in node.child_ids
            if cid not in state.visited_nodes and cid not in state.dead_ends
        ]
        
        if not unvisited:
            if state.backtrack_stack:
                return "backtrack"
            return "done"
        
        # Has unvisited children -> traverse
        return "traverse"
    
    def _do_expand(self, state: RLMAgentState, node: SkeletonNode) -> RLMAgentState:
        """Expand current node: fetch content and store as variable."""
        content = self.kv_store.get(node.node_id)
        
        if content:
            pointer = generate_pointer_name(node.header)
            self.variable_store.assign(pointer, content, node.node_id)
            state.variables.append(pointer)
            state.context += f"\nFound: {pointer} (from {node.header})"
            
            state.add_trace(
                "variable_stitching",
                f"Stored {pointer}",
                {"node": node.node_id, "chars": len(content)},
            )
        
        state.visited_nodes.append(node.node_id)
        state.current_node_id = None
        return state
    
    def _do_traverse(self, state: RLMAgentState, node: SkeletonNode) -> RLMAgentState:
        """Traverse to children using ToT with pre-filtering."""
        # Get children
        children = [self.skeleton.get(cid) for cid in node.child_ids]
        children = [c for c in children if c is not None]
        
        # Apply pre-filtering
        if state.extracted_keywords and self.config.enable_pre_filtering:
            matching, remaining = self.pre_filter.filter_nodes_by_keywords(
                children,
                state.extracted_keywords,
            )
            
            # If we have matching nodes, prioritize them
            if matching:
                selected = matching[:self.config.top_k]
                state.add_trace(
                    "navigation",
                    f"Pre-filter selected {len(selected)}/{len(children)} children",
                    {"selected": [n.node_id for n in selected]},
                )
            else:
                # Fall back to ToT evaluation
                selected = self._tot_evaluate_children(state, children)
        else:
            # Use ToT evaluation
            selected = self._tot_evaluate_children(state, children)
        
        # Queue selected children
        if selected:
            for child in selected:
                if child.node_id not in state.nodes_to_visit:
                    state.nodes_to_visit.append(child.node_id)
            
            # Push current node to backtrack stack
            state.backtrack_stack.append(node.node_id)
        else:
            # Dead end
            state.dead_ends.append(node.node_id)
        
        state.visited_nodes.append(node.node_id)
        state.current_node_id = None
        return state
    
    def _tot_evaluate_children(
        self,
        state: RLMAgentState,
        children: list[SkeletonNode],
    ) -> list[SkeletonNode]:
        """Use Tree of Thoughts to evaluate children."""
        if not self._llm_fn or not children:
            return children[:self.config.top_k]
        
        # Format children for evaluation
        children_text = "\n".join(
            f"  - [{c.node_id}] {c.header}: {c.summary[:150]}"
            for c in children
        )
        
        current_node = self.skeleton.get(state.current_node_id or self.root_id)
        current_summary = f"{current_node.header}: {current_node.summary}" if current_node else ""
        
        tot_prompt = f"""You are evaluating document sections for relevance.

Current location: {current_summary}

Children sections:
{children_text}

Query: {state.current_sub_question or state.question}

TASK: Evaluate each child's probability (0.0-1.0) of containing relevant information.

OUTPUT FORMAT (JSON):
{{
    "evaluations": [
        {{"node_id": "...", "probability": 0.85, "reasoning": "..."}}
    ],
    "selected_nodes": ["node_id_1", "node_id_2"],
    "is_dead_end": false
}}

JSON only:"""
        
        try:
            import json
            
            response = self._llm_fn(tot_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                result = json.loads(json_match.group())
                selected_ids = result.get("selected_nodes", [])
                
                # Map back to nodes
                selected = [c for c in children if c.node_id in selected_ids]
                
                if not selected and not result.get("is_dead_end", False):
                    # Fallback: take top-k by probability
                    evaluations = result.get("evaluations", [])
                    sorted_evals = sorted(
                        evaluations,
                        key=lambda x: x.get("probability", 0),
                        reverse=True,
                    )
                    top_ids = [e["node_id"] for e in sorted_evals[:self.config.top_k]]
                    selected = [c for c in children if c.node_id in top_ids]
                
                return selected
                
        except Exception as e:
            logger.warning("tot_evaluation_failed", error=str(e))
        
        # Fallback: return first top_k children
        return children[:self.config.top_k]
    
    def _do_backtrack(self, state: RLMAgentState) -> RLMAgentState:
        """Backtrack to previous node."""
        if state.backtrack_stack:
            parent_id = state.backtrack_stack.pop()
            state.dead_ends.append(state.current_node_id or "")
            state.current_node_id = parent_id
            
            state.add_trace(
                "navigation",
                f"Backtracked to {parent_id}",
                {"from": state.current_node_id},
            )
        else:
            state.current_node_id = None
        
        return state
    
    def _phase_synthesize(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 4: Synthesize answer from variables."""
        state.add_trace("synthesis", "Synthesizing answer from variables")
        
        if not state.variables:
            state.answer = "No relevant content found in the document."
            state.confidence = 0.0
            return state
        
        # Collect all variable content
        contents = []
        for pointer in state.variables:
            content = self.variable_store.resolve(pointer)
            if content:
                contents.append(f"=== {pointer} ===\n{content}")
        
        context_text = "\n\n".join(contents)
        
        if not self._llm_fn:
            state.answer = context_text
            state.confidence = 0.5
            return state
        
        # Handle multiple choice
        options = state.metadata.get("options")
        if options:
            options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            synthesis_prompt = f"""Based on the context, answer this multiple-choice question.

Question: {state.question}

Options:
{options_text}

Context:
{context_text}

Respond with ONLY the letter and full option text (e.g., "A. [option text]"):"""
        else:
            synthesis_prompt = f"""Based on the context, answer the question concisely.

Question: {state.question}

Context:
{context_text}

Answer:"""
        
        try:
            answer = self._llm_fn(synthesis_prompt)
            state.answer = answer.strip()
            state.confidence = min(1.0, len(state.variables) * 0.25)
            
            # Normalize multiple choice answer
            if options:
                state.answer = self._normalize_mc_answer(state.answer, options)
                
        except Exception as e:
            logger.error("synthesis_failed", error=str(e))
            state.answer = f"Error during synthesis: {str(e)}"
            state.confidence = 0.0
        
        return state
    
    def _normalize_mc_answer(self, answer: str, options: list) -> str:
        """Normalize multiple choice answer to match option text."""
        answer_lower = answer.lower().strip()
        
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            opt_lower = opt.lower()
            
            if (answer_lower.startswith(f"{letter.lower()}.") or
                answer_lower.startswith(f"{letter.lower()})") or
                opt_lower in answer_lower):
                return opt
        
        return answer
    
    def _phase_verify(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 5: Verify the answer."""
        state.add_trace("verification", "Verifying answer")
        
        # Collect evidence
        evidence = [
            self.variable_store.resolve(p) or ""
            for p in state.variables
        ]
        
        result = self.verification_engine.verify_answer(
            state.question,
            state.answer or "",
            evidence,
        )
        
        state.verification_result = result
        
        if result.get("improved_answer"):
            state.answer = result["improved_answer"]
        
        state.confidence = result.get("confidence", state.confidence)
        
        state.add_trace(
            "verification",
            f"Verification complete: valid={result.get('is_valid', True)}",
            {"issues": result.get("issues", [])},
        )
        
        return state


# =============================================================================
# Factory Function
# =============================================================================


def create_rlm_navigator(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    config: RLMConfig | None = None,
) -> RLMNavigator:
    """
    Create an RLM Navigator instance.
    
    Args:
        skeleton: Skeleton index.
        kv_store: KV store with full content.
        config: Optional configuration.
        
    Returns:
        Configured RLMNavigator.
        
    Example:
        from rnsr import ingest_document, build_skeleton_index
        from rnsr.agent.rlm_navigator import create_rlm_navigator, RLMConfig
        
        result = ingest_document("contract.pdf")
        skeleton, kv_store = build_skeleton_index(result.tree)
        
        # With custom config
        config = RLMConfig(
            max_recursion_depth=3,
            enable_pre_filtering=True,
            enable_verification=True,
        )
        
        navigator = create_rlm_navigator(skeleton, kv_store, config)
        result = navigator.navigate("What are the liability terms?")
        print(result["answer"])
    """
    nav = RLMNavigator(skeleton, kv_store, config)
    
    # Configure LLM
    try:
        from rnsr.llm import get_llm
        llm = get_llm()
        nav.set_llm_function(lambda p: str(llm.complete(p)))
    except Exception as e:
        logger.warning("llm_config_failed", error=str(e))
    
    return nav


def run_rlm_navigator(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    config: RLMConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the RLM Navigator on a question.
    
    Convenience function that creates and runs the navigator.
    
    Args:
        question: The user's question.
        skeleton: Skeleton index.
        kv_store: KV store.
        config: Optional configuration.
        metadata: Optional metadata.
        
    Returns:
        Dict with answer, confidence, trace.
    """
    navigator = create_rlm_navigator(skeleton, kv_store, config)
    return navigator.navigate(question, metadata)
