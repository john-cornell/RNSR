# Recursive Neural-Symbolic Retriever (RNSR) - System Overview

**Current State as of January 26, 2026**

The RNSR (Recursive Neural-Symbolic Retriever) is a next-generation Document Indexing and RAG (Retrieval-Augmented Generation) system. Unlike traditional RAG which flattens documents into chunks, RNSR preserves the **hierarchical structure** of documents and uses a **recursive agent** to navigate them.

## 1. Core Philosophy: "Prompt-as-Environment"

The system implements the "Prompt-as-Environment" abstraction (Zhang et al., 2025). 
- **The Problem**: Large documents don't fit in context windows, and "chunking" destroys global context.
- **The Solution**: The document is NOT passed to the LLM. Instead, the document exists as a variable (`DOC_VAR`) in a persistent Python REPL environment. 
- **The Interaction**: The LLM writes Python code to query, slice, and read the document, acting like a human researcher using a tool.

## 2. Key Capabilities

### A. Component-Based Architecture
1.  **Skeleton Indexing (The Map)**
    - Parses PDFs and detects visual layout (headers, fonts).
    - Builds a hierarchical `SkeletonNode` tree representing the document structure (Sections > Sub-sections > Paragraphs).
    - **Benefit**: The agent sees "Section 5: Liability" and knows it contains "Section 5.1", without needing to read the text yet.

2.  **Recursive Agent (The Navigator)**
    - Uses a **Tree of Thoughts (ToT)** reasoning engine.
    - **Step 1: Inspect**: Looks at the current level of the document tree.
    - **Step 2: Evaluate**: Scores child nodes based on relevance to the user's query (Probability scores).
    - **Step 3: Decide**: 
        - If a node is highly relevant (`> threshold`), it descends into it.
        - If a node is a dead end (`< dead_end_threshold`), it prunes it.
    - **Step 4: Expand**: When it finds the specific leaf node containing the answer, it reads the full text.

3.  **The REPL Environment (The Workspace)**
    - A sandboxed Python environment where the agent operates.
    - **Persistent Memory**: Findings are stored as variables (e.g., `$LIABILITY_CLAUSE`). This prevents "context rot" where the LLM forgets previous findings as the conversation gets too long.
    - **Sub-Agents**: The main agent can spawn "Sub-LLMs" to handle specific sub-tasks in parallel (e.g., "Extract the date from this specific clause").

## 3. "Now" - Recent Enhancements & Configurations

The system has been hardened for reliability and flexible experimentation.

### A. Configurable Navigation (Tree of Thoughts)
The strictness of the agent's navigation is now fully configurable via CLI arguments. This allows tuning between "Precision" (strict) and "Recall" (exploration).

- `--tot-threshold`: (Default: `0.4`) The confidence score required to explore a branch. Lowering this makes the agent more curious.
- `--tot-dead-end`: (Default: `0.1`) The score below which a branch is completely ignored. 

### B. Resilience & Stability
- **Exponential Backoff**: The LLM communication layer (specifically for Gemini) now includes robust retry logic. It handles `503 Service Unavailable` and rate limits by waiting exponentially (2s, 4s, 8s...) before retrying. 
- This ensures the "bursty" nature of the recursive agent (which might fire 10 queries at once) doesn't crash the application.

### C. Evaluation Suite
A built-in benchmarking tool (`rnsr.benchmarks.evaluation_suite`) automates testing against datasets like **FinanceBench**. It can run local cached evaluations to save API costs and time.

## 4. Current Workflow Example

1.  **User Query**: "What is the termination fee?"
2.  **Root Layout**: Agent sees headers: [1. Intro, 2. Definitions, ..., 9. Termination].
3.  **ToT Scoring**: 
    - "1. Intro": 0.05 (Dead End) -> Prune.
    - "9. Termination": 0.85 (High Relevance) -> Select.
4.  **Descend**: Agent "enters" Node 9. Sees [9.1 Notice Period, 9.2 Fees].
5.  **Refine**: 
    - "9.1": 0.2 (Low)
    - "9.2": 0.95 (High) -> Select.
6.  **Read**: Agent calls `read_node('9.2')`.
7.  **Answer**: Agent synthesizes the text from 9.2 to answer the user request.

## 5. Technical Stack

- **Framework**: `LlamaIndex`, `LangGraph` (State Machine).
- **PDF Processing**: `PyMuPDF` (aka `fitz`).
- **LLM Providers**: Google Gemini (Flash 2.5), OpenAI, Anthropic.
- **Resilience**: `Tenacity` library for retries.
