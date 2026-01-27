# System Architecture: Recursive Neural-Symbolic Retriever (RNSR)

**Target Audience:** Computer Science / Engineering Team

## 1. Architectural Paradigm: Why RNSR?

We are solving the **Long-Context Retrieval** problem. Traditional approaches occupy two extremes of the trade-off curve between *latency* and *contextual coherence*. RNSR is a hybrid approach designed to optimize this curve.

### A. The Baseline: Naive RAG ("Page Indexing")
*   **Mechanism:** Flat vector search (KNN) over arbitrary chunks (e.g., 512 tokens). Effectively a $O(1)$ lookup complexity.
*   **The Failure Mode (Context Fragmentation):** "Shredding" the document destroys global structure.
    *   *Example:* If a chunk says "The fee is 5%", but the "Termination" header was in the previous chunk, the vector embedding loses semantic binding.
    *   *Result:* High retrieval speed, high hallucination rate, loss of dependency resolution.

### B. The Alternative: Recursive Summarization (RLM)
*   **Mechanism:** Sequential Map-Reduce. Read Chunk 1 -> Summarize -> Pass to Chunk 2. Complexity is $O(N)$ where $N$ is document length.
*   **The Failure Mode (Latency & Drift):**
    *   *Latency:* Strictly serial execution blocks on LLM inference time.
    *   *Information Bottleneck:* The "summary state" loses resolution as $N$ increases ("Context Rot").

### C. Our Solution: RNSR (The "RLM" + "PageIndex" Synthesis)
*   **Architecture**: We combine two cutting-edge paradigms:
    1.  **PageIndex Philosophy** (from VectifyAI):
        *   Rejects "shredding" documents into vectors.
        *   Retains the **Document Tree** (Table of Contents) as the primary data structure.
    2.  **Recursive Language Models** (arXiv:2512.24601):
        *   Treats the prompt as an "External Environment" (a REPL).
        *   Allows the LLM to inspect the tree and recursively call itself (`sub_questions` in our graph) to solve sub-problems.
*   **Mechanism:** We treat document retrieval as a **State Space Search** problem over a hierarchical tree.
*   **Complexity:** Approximates $O(log_k N)$ where $k$ is the branching factor of the document headers.
*   **Visual-Symbolic Hybrid:** We reconstruct the document's latent hierarchy (DOM) using visual layout analysis, then use a Neural Agent (LLM) to traverse this symbolic structure.

---

## 2. Data Structures: The "Document Model"

We implement a custom "Document Object Model" (DOM) for unstructured data, decoupled from the inference engine.

### The `DocumentTree`
Instead of a flat list of text, we build a persistent N-ary tree:

```python
class DocumentNode:
    id: UUID
    level: int          # Depth (H1, H2, H3)
    header: str         # Symbolic anchor (e.g., "Section 9.2")
    content: str        # The raw tokens (Leaf nodes only)
    embedding: Vector   # Dense vector representation of the summary
    children: List[DocumentNode]
```

### Ingestion Pipeline (Heuristic Layout Analysis)
We do not rely on LLMs for parsing (too expensive). We use Computer Vision heuristics:
1.  **Font Histogram Analysis**: Compute distributions of (Font, Size, Weight). Outliers > $\sigma$ are classified as Header candidates.
2.  **XY-Cut Algorithm**: Recursive segmentation of the page layout into geometric blocks.
3.  **Result**: A layout-aware tree that preserves the author's intended logical grouping.

---

## 3. Execution Engine: Tree of Thoughts (ToT)

The Agent is a **Finite State Machine** (implemented via `LangGraph`) that executes a probabilistic search algorithm.

1.  **State Initialization**: $S_0 = N_{root}$
2.  **Lookahead (Expansion)**: The agent observes all edges $E = \{child_1, ..., child_k\}$ from current node.
3.  **Evaluation (Discriminator)**:
    A lightweight LLM call computes $P(relevant | query, header_i, summary_i)$ for all $i$.
    *   This is the "Neural" component guiding the symbolic traversal.
4.  **Selection Policy**:
    *   Prune if $P < \theta_{dead\_end}$ (e.g., 0.1).
    *   Traverse if $P > \theta_{select}$ (e.g., 0.4).
5.  **Backtracking**: If a path yields $P < \epsilon$, the agent backtracks to the parent node (DFS/BFS hybrid).

---

## 4. The Role of LlamaIndex

We utilize **LlamaIndex** as our Indexing and Retrieval Middleware Layer, but heavily customized.

#### How we use it:
1.  **Vector Store Abstraction**: LlamaIndex manages the interface to the underlying vector database (e.g., pgvector, Chroma).
2.  **`IndexNode` Utilization**:
    Standard LlamaIndex nodes map `Embedding <-> Text Chunk`.
    We implement **Skeleton Nodes**:
    *   **Text Field**: Contains *only* the summary (50-100 words).
    *   **Embedding**: Vector of the *summary*.
    *   **Metadata**: Contains the graph pointers (`child_ids`, `parent_id`).
    *   **Result**: The expensive "Full Content" is stored in a separate Key-Value (KV) store, not in the vector index. The Vector Index is purely a navigation map.

#### Why not simpler LlamaIndex pipelines?
Standard `RecursiveRetriever` in LlamaIndex follows hard-coded depth rules. Our implementation injects a **Reasoning Loop** (The Agent) at every node, allowing dynamic decision making ("This section looks promising, let me verify") rather than static graph traversal.
