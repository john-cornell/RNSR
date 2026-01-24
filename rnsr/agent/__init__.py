"""
Agent Module - Recursive Navigator

Responsible for:
1. Variable Store (pointer-based stitching)
2. LangGraph state machine
3. Navigation tools
4. RAP & synthesis
"""

from rnsr.agent.graph import (
    AgentState,
    build_navigator_graph,
    create_initial_state,
    create_navigator_tools,
    run_navigator,
)
from rnsr.agent.variable_store import VariableStore, generate_pointer_name

__all__ = [
    # Variable Store
    "VariableStore",
    "generate_pointer_name",
    # Agent Graph
    "AgentState",
    "build_navigator_graph",
    "create_initial_state",
    "create_navigator_tools",
    "run_navigator",
]
