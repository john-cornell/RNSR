# RNSR vs. VectifyAI PageIndex

## The Short Answer
You are correct: **Functionally, RNSR and VectifyAI's PageIndex are cousins.** They both solve the "Vector Search isn't smart enough" problem using the same core philosophy: **Hierarchical Agentic Navigation.**

Both systems reject the idea of "flattening" a document into chunks and betting on similarity. Instead, both build a **Tree** (Table of Contents) and let an LLM "browse" it to find the answer.

## Key Differences

While the *philosophy* is identical, the *implementation strategy* differs significantly.

### 1. Tree Construction (How the map is made)
*   **RNSR (This System)**: Uses a **Visual-First, Deterministic** approach.
    *   **Mechanism**: It extracts font statistics, indentation, and whitespace directly from the PDF instructions. It builds the tree based on *how the document looks* (e.g., "This line is Arial 24px Bold, so it's a Header").
    *   **Pro**: Extremely fast and cheap (no LLM calls needed to build the initial structure). It flawlessly captures the author's intended hierarchy in standard PDFs.
    *   **Fallback**: It only uses LLMs (Semantic Clustering) if the visual parsing fails.
    
*   **VectifyAI PageIndex**: Uses a **Reasoning/Model-First** approach.
    *   **Mechanism**: It typically uses an LLM or VLM (Vision Language Model) agent to "read" the document and strictly *generate* a semantic tree structure.
    *   **Pro**: Very robust on scanned documents or documents with weird visual layouts where font sizes don't match hierarchy.
    *   **Con**: Higher indexing cost (requires heavy inference to build the tree).

### 2. Retrieval Logic
*   **RNSR**: **Pathfinding Agent**.
    *   The retrieval is a "Navigator Graph" (in `rnsr/agent/navigator_api.py`). It specifically mimics a person opening a book: check TOC -> go to chapter -> scan headers -> read section.
    *   It treats the document as a *State Machine*.

*   **PageIndex**: **Reasoning Tree Search**.
    *   It formulates retrieval as a search problem over the generated JSON tree. It allows global reasoning over the entire tree structure to identify relevant nodes.

## Why build RNSR if PageIndex exists?
1.  **Control**: RNSR is a "White Box" implementation. You own every line of the navigation logic (`rnsr/agent/graph.py`). You can tune exactly how aggressive the agent is (`--tot-threshold`) or swap the underlying PDF parser.
2.  **Cost**: RNSR's visual parser is computationally free compared to LLM-based indexing.
3.  **Hybridization**: RNSR allows you to mix "Visual Heuristics" (fast) with "Semantic Fallback" (smart). PageIndex generally abstracts this away.

## Summary Comparison

| Feature | RNSR (This Repo) | VectifyAI PageIndex |
| :--- | :--- | :--- |
| **Core Philosophy** | Structure > Vectors | Structure > Vectors |
| **Indexing Method** | **Visual Heuristics** (Fonts/Layout) + Semantic Fallback | **LLM/VLM Reasoning** + Semantic Layout |
| **Indexing Cost** | Low (CPU-bound PDF parsing) | High (LLM-bound generation) |
| **Retrieval** | Explicit Agent Navigation (State Machine) | Reasoning Tree Search |
| **Best For** | Clean digital PDFs, Financial Reports, Standard Layouts | Complex layouts, Scanned docs, minimizing dev time |

**In short:** You are building a custom, highly-tuned version of what PageIndex offers as a product.
