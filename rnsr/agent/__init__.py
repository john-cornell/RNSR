"""
Agent Module - Recursive Navigator with Full RLM Support

Implements the state-of-the-art hybrid retrieval system combining:
- PageIndex: Vectorless, reasoning-based tree search
- RLMs: REPL environment with recursive sub-LLM calls  
- RNSR: Latent hierarchy reconstruction + variable stitching

Key Features:
1. RLM Navigator - Full recursive language model with pre-filtering
2. REPLEnvironment - Python REPL with DOC_VAR and code execution
3. Variable Store - Pointer-based stitching to prevent context pollution
4. Tree of Thoughts (ToT) - LLM-based navigation decisions
5. Pre-filtering - Keyword/regex filtering before LLM calls
6. Deep Recursion - Multi-level recursive sub-LLM calls
7. Answer Verification - Sub-LLM validation of answers
8. Async Processing - Parallel sub-LLM execution

Inspired by:
- PageIndex (VectifyAI): https://github.com/VectifyAI/PageIndex
- Recursive Language Models: https://arxiv.org/html/2512.24601v1
"""

from rnsr.agent.graph import (
    AgentState,
    build_navigator_graph,
    create_initial_state,
    create_navigator_tools,
    run_navigator,
    # Tree of Thoughts (Section 7.2)
    evaluate_children_with_tot,
    backtrack_to_parent,
    TOT_SYSTEM_PROMPT,
    # RLM Recursive Execution (Section 2.2)
    execute_sub_task_with_llm,
    batch_execute_sub_tasks,
    process_pending_questions,
    DECOMPOSITION_PROMPT,
)
from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.agent.navigator_api import (
    NavigatorAPI,
    create_navigator,
    execute_rap_query,
)
from rnsr.agent.repl_env import (
    REPLEnvironment,
    create_repl_environment,
    RLM_SYSTEM_PROMPT,
    batch_process_async,
)
from rnsr.agent.rlm_navigator import (
    RLMNavigator,
    RLMConfig,
    RLMAgentState,
    PreFilterEngine,
    RecursiveSubLLMEngine,
    AnswerVerificationEngine,
    create_rlm_navigator,
    run_rlm_navigator,
)

__all__ = [
    # RLM Navigator (State-of-the-Art)
    "RLMNavigator",
    "RLMConfig",
    "RLMAgentState",
    "PreFilterEngine",
    "RecursiveSubLLMEngine",
    "AnswerVerificationEngine",
    "create_rlm_navigator",
    "run_rlm_navigator",
    # REPL Environment (Section 2.1 - Prompt-as-Environment)
    "REPLEnvironment",
    "create_repl_environment",
    "RLM_SYSTEM_PROMPT",
    "batch_process_async",
    # Navigator API (Section 5.1 Phase III)
    "NavigatorAPI",
    "create_navigator",
    "execute_rap_query",
    # Variable Store
    "VariableStore",
    "generate_pointer_name",
    # Agent Graph
    "AgentState",
    "build_navigator_graph",
    "create_initial_state",
    "create_navigator_tools",
    "run_navigator",
    # Tree of Thoughts (Section 7.2)
    "evaluate_children_with_tot",
    "backtrack_to_parent",
    "TOT_SYSTEM_PROMPT",
    # RLM Recursive Execution (Section 2.2)
    "execute_sub_task_with_llm",
    "batch_execute_sub_tasks",
    "process_pending_questions",
    "DECOMPOSITION_PROMPT",
]
