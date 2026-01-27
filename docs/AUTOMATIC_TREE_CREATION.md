# Automatic Document Tree Creation (Ingestion)

Yes! The system automatically creates a document tree "on the fly" when you ingest standard documents (PDFs). This is the **Ingestion** phase of the pipeline.

## How It Works: The "Ingestion Pipeline"

When you call `client.ask("document.pdf", ...)` or `client.ingest_document("document.pdf")`, the system runs a sophisticated multi-stage analysis to reconstruct the document's hierarchy from its layout.

It uses a **"Graceful Degradation"** strategy (defined in `rnsr/ingestion/pipeline.py`), trying the best method first and falling back if needed.

### 1. Pre-Analysis
It quickly scans the PDF to detect **Layout Complexity**.
- Is it single column or multi-column?
- Are there headers?
- Is it text-based or scanned images?

### 2. Tier 1: Visual-Geometric Analysis (The "Smart" Way)
This is the default for most digital PDFs.
- **Font Histogram Analysis**: It scans the fonts used on every page.
    - If it sees text in **Bold, Size 24**, it guesses "This is a Header 1".
    - If it sees text in *Italic, Size 12*, it guesses "This is a caption".
    - If it sees text in Normal, Size 10, it treats it as body text.
- **Tree Construction**: Based on these font cues, it automatically builds the tree:
    ```
    Big Font -> Node: "1. Introduction"
        Normal Font -> Content for "1. Introduction"
    Medium Font -> Node: "1.1 Background" 
        Normal Font -> Content for "1.1 Background"
    ```

### 3. Tier 2: Semantic Boundary Detection (The "Fallback")
If the PDF has no headers (e.g., a plain text dump saved as PDF), Tier 1 will produce a flat list. The system detects this failure and switches to **Semantic Splitting**.
- It reads the raw text stream.
- It uses embeddings to detect "topic shifts" (where the subject changes).
- It splits the text at these shifts and uses an LLM to **generate synthetic headers** for each chunk.

### 4. Tier 3: OCR (Optical Character Recognition)
If the PDF is just an image (a scan), Tier 1 and 2 will see "No Text". 
- The system automatically detects "No extractable text".
- It runs OCR (e.g., Tesseract) to convert the image to text, then goes back to Tier 1 or 2 to build the tree.

## User Experience
To you, this is invisible. You just give it a file:

```python
# The system handles parsing, layout analysis, header detection, 
# and tree construction automatically.
client.ask("my_complex_report.pdf", "What is the conclusion?")
```
