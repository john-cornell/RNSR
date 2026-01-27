# Comprehensive Technology Stack Analysis

The RNSR system is not just a "wrapper" around an LLM. It is a sophisticated pipeline composed of specialized subsystems for Computer Vision, Graph Theory, and Probabilistic Reasoning.

Here is the breakdown of every major technology used, **what** it does, and **why** we chose it.

---

## 1. Orchestration & Reasoning Layer

### **LangGraph** (by LangChain)
*   **What it does:** Manages the **Finite State Machine (FSM)** of the Agent. It defines the "Nodes" (Inspect, Decide, Read) and the "Edges" (Logic for transitioning/looping).
*   **Why we use it:**
    *   **Cycles:** Standard chains (LangChain/LlamaIndex pipelines) are Directed Acyclic Graphs (DAGs)—they go A->B->C. Our agent needs **Loops** (Inspect -> Decide -> Backtrack -> Inspect). LangGraph is specifically designed for cyclic, stateful agentic flows.
    *   **State Persistence:** It maintains the `AgentState` (the "Memory" of the current navigation path) automatically across steps.

### **LlamaIndex**
*   **What it does:** The **Data Access Layer**. It handles the dirty work of chunking, embedding, vector storage, and retrieving nodes.
*   **Why we use it:**
    *   **Abstractions:** It provides the cleanest interface for "Retriever" patterns. We don't want to write raw SQL/Vector DB queries.
    *   **`IndexNode`:** Its specific implementation of "Index Nodes" (nodes that point to other nodes) enables our **Skeleton Index** architecture, where a lightweight vector search redirects to a heavy content block.

---

## 2. Ingestion & Computer Vision Layer

### **PyMuPDF (fitz)**
*   **What it does:** Low-level PDF parsing. It extracts text *plus* metadata: font name, font size, bold/italic flags, and bounding box coordinates ($x, y$).
*   **Why we use it:**
    *   **Speed:** It is C-based and orders of magnitude faster than Python-native parsers like `pypdf`.
    *   **Visual Fidelity:** Unlike simple text extractors, it gives us the **Visual Layout**. We need to know that "Terms" is written in *Arial Bold Size 24* to know it's a Header, not just text.

### **Scikit-Learn (K-Means Clustering)**
*   **What it does:** Unsupervised Machine Learning. We feed it the list of all font sizes on a page.
*   **Why we use it:**
    *   **Dynamic Header Detection:** We can't hardcode "Header = Size 20" because every PDF is different. K-Means automatically groups fonts into clusters (e.g., Small, Medium, Large). We mathematically prove that the "Large" cluster contains the headers, without human rules.

### **Recursive XY-Cut Algorithm** (Custom Implementation)
*   **What it does:** A computer vision algorithm that "slices" whitespace. It looks for wide horizontal or vertical gaps to identify columns and paragraphs.
*   **Why we use it:**
    *   **Reading Order:** PDFs are just "bags of characters" with coordinates. They don't know that Column 1 ends and Column 2 begins. XY-Cut reconstructs the **correct reading order** for complex multi-column layouts so the LLM doesn't read across columns (which generates gibberish).

---

## 3. Storage & Data Modeling

### **Pydantic**
*   **What it does:** Strict data validation. Defines our `DocumentTree`, `SkeletonNode`, and `AgentState`.
*   **Why we use it:**
    *   **Type Safety:** In complex pipelines, passing uncontrolled dictionaries leads to silent failures. Pydantic ensures that if a Node is missing an ID or a Child, the system crashes *loudly* and *early* (Evaluation) rather than *silently* (Production).

### **The "Skeleton" Storage Pattern (Dual-Store)**
*   **Technology:** `VectorStore` (embeddings) + `KVStore` (Hashtable).
*   **Why we use it:**
    *   **Optimization:** Putting full text into a Vector Store is expensive and slow to search.
    *   **Architecture:** We put **Summaries** in the Vector Store (fast, low memory) and **Full Content** in the KV Store (disk-based). The Agent searches the index but reads from the KV store. This decouples "Finding" from "Reading."

---

## 4. Resilience & Reliability

### **Tenacity**
*   **What it does:** Decorator-based retry library.
*   **Why we use it:**
    *   **API Reality:** LLM APIs (Gemini/OpenAI) are unstable. They throw `503 Unavailable` or `429 Rate Limit` constantly under load.
    *   **Exponential Backoff:** Tenacity handles the math of "Wait 2s, then 4s, then 8s" automatically so our benchmark suite doesn't crash overnight.

### **Structlog**
*   **What it does:** Structured JSON logging.
*   **Why we use it:**
    *   **Observability:** Instead of printing "Error in file", it logs `{"event": "ingestion_failed", "file": "doc.pdf", "error": "timeout"}`. This allows us to parse logs programmatically to generate the benchmark reports.

---

## 5. Intelligence (The LLMs)

### **Google Gemini 2.5 Flash**
*   **Role:** The "Workhorse".
*   **Why:** Speed and Cost. It has a massive context window (1M+ tokens) and is incredibly cheap compared to GPT-4. We use it for the heavy lifting of evaluating hundreds of tree nodes.

### **OpenAI GPT-4o / Anthropic Claude 3.5 Sonnet**
*   **Role:** The "Reasoners".
*   **Why:** Higher logic adherence. We use them (optionally) for the complex `Final Synthesis` step where accuracy matters more than speed.
