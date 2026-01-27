"""
Tests for RLM (Recursive Language Model) components.

Tests the REPL environment and recursive execution from Section 2.
"""

import pytest

from rnsr.agent.repl_env import REPLEnvironment, create_repl_environment, RLM_SYSTEM_PROMPT
from rnsr.agent.variable_store import VariableStore
from rnsr.agent.graph import (
    decompose_query,
    execute_sub_task_with_llm,
    batch_execute_sub_tasks,
    _simple_decompose,
    create_initial_state,
    DECOMPOSITION_PROMPT,
)
from rnsr.models import SkeletonNode
from rnsr.indexing.kv_store import InMemoryKVStore


# =============================================================================
# REPL Environment Tests
# =============================================================================


class TestREPLEnvironment:
    """Test the REPL environment from Section 2.1."""
    
    @pytest.fixture
    def sample_document(self):
        """Sample document text for DOC_VAR."""
        return """
Chapter 1: Introduction
This is the introduction to the document. It contains background information.

Chapter 2: Methodology
This chapter describes the methodology used in the research.

Chapter 3: Results
The results of the experiment are presented here.
""".strip()
    
    @pytest.fixture
    def sample_skeleton(self):
        """Sample skeleton index."""
        return {
            "root": SkeletonNode(
                node_id="root",
                parent_id=None,
                level=0,
                header="Document Title",
                summary="A research document",
                child_ids=["ch1", "ch2", "ch3"],
            ),
            "ch1": SkeletonNode(
                node_id="ch1",
                parent_id="root",
                level=1,
                header="Introduction",
                summary="Background and context",
                child_ids=[],
            ),
            "ch2": SkeletonNode(
                node_id="ch2",
                parent_id="root",
                level=1,
                header="Methodology",
                summary="Research methods",
                child_ids=[],
            ),
            "ch3": SkeletonNode(
                node_id="ch3",
                parent_id="root",
                level=1,
                header="Results",
                summary="Experimental results",
                child_ids=[],
            ),
        }
    
    @pytest.fixture
    def sample_kv_store(self, sample_document):
        """KV store with content."""
        store = InMemoryKVStore()
        store.put("root", sample_document)
        store.put("ch1", "This is the introduction to the document.")
        store.put("ch2", "This chapter describes the methodology.")
        store.put("ch3", "The results of the experiment are presented here.")
        return store
    
    def test_repl_initialization(self, sample_document, sample_skeleton, sample_kv_store):
        """Test REPL environment initializes correctly."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        assert env.document_text == sample_document
        assert len(env.skeleton) == 4
        assert "DOC_VAR" in env._namespace
        assert "DOC_TREE" in env._namespace
    
    def test_doc_var_length(self, sample_document, sample_skeleton, sample_kv_store):
        """Test len(DOC_VAR) works."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        result = env.execute("len(DOC_VAR)")
        
        assert result["success"] is True
        assert str(len(sample_document)) in str(result["output"])
    
    def test_doc_var_slicing(self, sample_document, sample_skeleton, sample_kv_store):
        """Test DOC_VAR[i:j] slicing works."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        result = env.execute("DOC_VAR[0:20]")
        
        assert result["success"] is True
        assert "Chapter 1" in result["output"]
    
    def test_list_children(self, sample_document, sample_skeleton, sample_kv_store):
        """Test list_children() function."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        result = env.execute("list_children('root')")
        
        assert result["success"] is True
        assert "Introduction" in str(result["output"]) or "ch1" in str(result["output"])
    
    def test_read_node(self, sample_document, sample_skeleton, sample_kv_store):
        """Test read_node() function."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        result = env.execute("read_node('ch1')")
        
        assert result["success"] is True
        assert "introduction" in result["output"].lower()
    
    def test_store_and_get_variable(self, sample_document, sample_skeleton, sample_kv_store):
        """Test variable stitching - store and retrieve."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        # Store a variable
        result = env.execute("store_variable('TEST_VAR', 'Hello World')")
        assert result["success"] is True
        assert "$TEST_VAR" in result["output"]
        
        # Retrieve the variable
        result = env.execute("get_variable('TEST_VAR')")
        assert result["success"] is True
        assert "Hello World" in result["output"]
    
    def test_search_text(self, sample_document, sample_skeleton, sample_kv_store):
        """Test search_text() regex function."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        result = env.execute("search_text('Chapter \\\\d+')")
        
        assert result["success"] is True
        assert "Chapter" in result["output"]
    
    def test_execution_error_handling(self, sample_document, sample_skeleton, sample_kv_store):
        """Test that errors are captured properly."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        result = env.execute("undefined_variable")
        
        assert result["success"] is False
        assert result["error"] is not None
        assert "NameError" in result["error"]
    
    def test_markdown_code_cleaning(self, sample_document, sample_skeleton, sample_kv_store):
        """Test that markdown code blocks are cleaned."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        # Simulate LLM output with markdown
        code = """```python
len(DOC_VAR)
```"""
        
        result = env.execute(code)
        
        assert result["success"] is True
    
    def test_get_state_summary(self, sample_document, sample_skeleton, sample_kv_store):
        """Test state summary retrieval."""
        env = REPLEnvironment(
            document_text=sample_document,
            skeleton=sample_skeleton,
            kv_store=sample_kv_store,
        )
        
        summary = env.get_state_summary()
        
        assert "doc_length" in summary
        assert "num_nodes" in summary
        assert summary["doc_length"] == len(sample_document)
        assert summary["num_nodes"] == 4


class TestCreateREPLEnvironment:
    """Test the factory function."""
    
    def test_create_with_defaults(self):
        """Test creation with minimal args."""
        env = create_repl_environment()
        
        assert env is not None
        assert env.document_text == ""
    
    def test_create_with_document(self):
        """Test creation with document text."""
        env = create_repl_environment(document_text="Test document content")
        
        assert env.document_text == "Test document content"


class TestRLMSystemPrompt:
    """Test the RLM system prompt."""
    
    def test_system_prompt_contains_key_elements(self):
        """Verify system prompt has required elements."""
        assert "DOC_VAR" in RLM_SYSTEM_PROMPT
        assert "list_children" in RLM_SYSTEM_PROMPT
        assert "read_node" in RLM_SYSTEM_PROMPT
        assert "store_variable" in RLM_SYSTEM_PROMPT
        assert "sub_llm" in RLM_SYSTEM_PROMPT
        assert "batch_sub_llm" in RLM_SYSTEM_PROMPT


# =============================================================================
# Recursive Decomposition Tests
# =============================================================================


class TestSimpleDecomposition:
    """Test pattern-based query decomposition."""
    
    def test_compare_query(self):
        """Test 'compare X and Y' pattern."""
        result = _simple_decompose("Compare the 2023 and 2024 agreements")
        
        assert len(result) == 2
        assert "2023" in result[0]
        assert "2024" in result[1]
    
    def test_vs_query(self):
        """Test 'X vs Y' pattern."""
        result = _simple_decompose("Apple vs Microsoft market share")
        
        assert len(result) == 2
    
    def test_difference_query(self):
        """Test 'differences between X and Y' pattern."""
        result = _simple_decompose("What are the differences between plan A and plan B?")
        
        assert len(result) == 2
    
    def test_list_all_query(self):
        """Test 'list all X' pattern."""
        result = _simple_decompose("List all risk factors in the document")
        
        assert len(result) >= 1
    
    def test_simple_query(self):
        """Test simple query returns as-is."""
        query = "What is the main conclusion?"
        result = _simple_decompose(query)
        
        assert len(result) == 1
        assert result[0] == query


class TestDecomposeQuery:
    """Test LLM-based decomposition (with fallback)."""
    
    def test_decompose_updates_state(self):
        """Test that decompose_query updates state correctly."""
        state = create_initial_state(
            question="Compare section A and section B",
            root_node_id="root",
        )
        
        new_state = decompose_query(state)
        
        assert len(new_state["sub_questions"]) >= 1
        assert len(new_state["pending_questions"]) >= 1
        assert new_state["current_sub_question"] is not None
        assert new_state["iteration"] == 1
    
    def test_decompose_adds_trace(self):
        """Test that decomposition adds trace entry."""
        state = create_initial_state(
            question="What are the terms?",
            root_node_id="root",
        )
        
        new_state = decompose_query(state)
        
        assert len(new_state["trace"]) > 0
        assert new_state["trace"][0]["node_type"] == "decomposition"


class TestDecompositionPrompt:
    """Test the decomposition prompt template."""
    
    def test_prompt_has_placeholders(self):
        """Verify prompt has required placeholders."""
        assert "{query}" in DECOMPOSITION_PROMPT
        assert "{structure}" in DECOMPOSITION_PROMPT
        assert "JSON" in DECOMPOSITION_PROMPT


# =============================================================================
# Sub-task Execution Tests
# =============================================================================


class TestExecuteSubTask:
    """Test sub-task execution."""
    
    def test_execute_without_llm(self):
        """Test execution when LLM not available."""
        result = execute_sub_task_with_llm(
            sub_task="Extract the clause",
            context="Some contract text",
            llm_fn=None,  # No LLM
        )
        
        # Should return error or attempt to use default
        assert isinstance(result, str)
    
    def test_execute_with_mock_llm(self):
        """Test execution with mock LLM."""
        mock_llm = lambda p: "Mock response"
        
        result = execute_sub_task_with_llm(
            sub_task="Summarize this",
            context="Test content",
            llm_fn=mock_llm,
        )
        
        assert result == "Mock response"
    
    def test_execute_handles_llm_error(self):
        """Test that LLM errors are handled."""
        def failing_llm(prompt):
            raise RuntimeError("API Error")
        
        result = execute_sub_task_with_llm(
            sub_task="Extract data",
            context="Content",
            llm_fn=failing_llm,
        )
        
        assert "[Error" in result


class TestBatchExecuteSubTasks:
    """Test batch execution."""
    
    def test_batch_length_mismatch(self):
        """Test error when lists have different lengths."""
        with pytest.raises(ValueError):
            batch_execute_sub_tasks(
                sub_tasks=["Task 1", "Task 2"],
                contexts=["Context 1"],  # Mismatch!
            )
    
    def test_batch_empty_lists(self):
        """Test empty input returns empty output."""
        result = batch_execute_sub_tasks(
            sub_tasks=[],
            contexts=[],
        )
        
        assert result == []
