# Head-to-Head: RNSR vs. "Page Indexing" (Standard RAG)

When we talk about **"Page Indexing"**, we refer to the standard industry practice of Retrieval-Augmented Generation (RAG): splitting a PDF into pages (or chunks of 500-1000 words), embedding them, and putting them into a vector database.

Here is why RNSR fundamentally outperforms this approach.

---

## 1. The Structural Difference

### 📄 Page Indexing (The "Bag of Chunks")
*   **Structure:** Flat List.
*   **Representation:** The system sees the document as a pile of 500 unconnected index cards.
*   **Blind Spot:** It has no concept of "Chapter 1" or "Section B". It only knows "Chunk 45" vs "Chunk 46".
*   **The Flaw:** **Context Fragmentation**.
    *   *Scenario:* A clause starts on the bottom of Page 10 and ends on the top of Page 11.
    *   *Result:* Page Indexing splits this clause into two separate vectors. When you search for the clause, neither vector matches perfectly, so **retrieval fails**.

### 🌳 RNSR (The "Semantic Tree")
*   **Structure:** Hierarchical Tree (DOM).
*   **Representation:** The system sees the document as a nested book: `Book` > `Chapter` > `Section` > `Paragraph`.
*   **Awareness:** It knows that "Section 1.1" is *inside* "Section 1".
*   **The Fix:** **Context Unity**.
    *   *Scenario:* That same clause is contained entirely within the `Section 1.1` object (the `DocumentNode`).
    *   *Result:* The agent retrieves the *entire node*, keeping the clause intact.

---

## 2. The Retrieval Difference

### 🔍 Page Indexing: Similarity Search (KNN)
*   **Method:** "Find me chunks that *look like* my question."
*   **The Flaw:** **Semantic Ambiguity (Hallucination)**.
    *   *Question:* "What are the termination rights for the *Employee*?"
    *   *Document:* Contains "Termination" clauses in both the "Employee" section and the "Vendor Contract" section.
    *   *Result:* Page Indexing grabs *both* because they both share the word "Termination". The LLM then gets confused and might tell you the Vendor rules apply to the Employee. **This is a classic RAG failure.**

### 🧠 RNSR: Probabilistic Navigation (Decision Making)
*   **Method:** "Navigate to the section about Employees, then look for Termination."
*   **The Fix:** **Scope Isolation**.
    *   *Action:* The Agent sees the Root headers: `1. Employees` and `2. Vendors`.
    *   *Reasoning:* "The user asked about Employees. I will enter Node 1 and **prune** (ignore) Node 2."
    *   *Result:* The Agent *never even sees* the Vendor termination clause. It is mathematically impossible for it to hallucinate information from the wrong section because it never retrieved it.

---

## 3. The "Needle in a Haystack" Difference

### Page Indexing
*   **Performance:** Degrades as document size increases.
*   **Why:** In a 10-page document, finding the right page is easy. In a 500-page document, there are 50 pages that "sound sort of like" the answer. The "Signal-to-Noise" ratio drops, and the Top-5 chunks are often irrelevant.

### RNSR
*   **Performance:** Constant / Logarithmic scaling.
*   **Why:** The agent doesn't scan 500 pages. It scans 10 Chapter headers. Then it scans 5 Section headers. It ignores 98% of the document essentially instantly. A 500-page document is just as easy to navigate as a 10-page one, provided the Table of Contents is good.

---

## Summary Table

| Feature | Page Indexing (Standard RAG) | RNSR (Recursive Agent) |
| :--- | :--- | :--- |
| **Data Shape** | Flat List (Chunks) | Hierarchical Tree |
| **Search Algo** | K-Nearest Neighbors (Math) | Tree of Thoughts (Reasoning) |
| **Context** | Broken at page/chunk boundaries | Preserved by Section boundaries |
| **Disambiguation** | Poor (confuses similar terms) | Excellent (isolates by section) |
| **Scaling** | Gets worse with doc size | Scales logarithmically |
