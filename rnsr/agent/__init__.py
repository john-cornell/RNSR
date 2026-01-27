"""
Agent Module - Recursive Navigator with RLM Support

Implements Phase III from Section 5.1 of the research paper:
"The Recursive REPL Agent (The Navigator)"

And Section 2 "The Recursive Language Model (RLM)":
- Prompt-as-Environment abstraction (DOC_VAR)
- Recursive sub-LLM invocation for sub-task decomposition
- Variable Stitching to prevent Context Rot
- Batched parallel processing for efficiency

Provides:
1. REPLEnvironment - Python REPL with DOC_VAR and code execution
2. NavigatorAPI - REPL environment with list_children, read_node, search_index
3. Variable Store - Pointer-based stitching to prevent context pollution
4. LangGraph state machine - Cyclic control flow for RAP
5. RAP execution - Reasoning via Planning query loop
6. Tree of Thoughts (ToT) Prompting - Section 7.2
7. Recursive Decomposition - LLM-based sub-task generation
8. Batch Processing - Parallel sub-LLM calls

ToT Features:
- LLM-based child node evaluation with probability scoring
- Top-k selection of most promising navigation paths
- Automatic backtracking from dead ends
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

__all__ = [
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
