# Custom Tree Integration Guide

The RNSR system is designed to be modular. While it includes a powerful PDF parser (`ingest_document`), you can easily bypass it and provide your own document tree if you have a custom source (e.g., HTML parser, Notion export, JSON data).

## 1. The Data Structure

The core structure is the `DocumentTree`, defined in `rnsr/models.py`. It is a Pydantic model.

### JSON Schema

```json
{
  "id": "doc_123",
  "title": "My Custom Document",
  "root": {
    "id": "root",
    "level": 0,
    "header": "My Custom Document",
    "content": "",
    "children": [
      {
        "id": "node_1",
        "level": 1,
        "header": "1. Introduction",
        "content": "This is the full text of the introduction...",
        "children": [
            {
                "id": "node_1_1",
                "level": 2,
                "header": "1.1 Background",
                "content": "Deep details about background...",
                "children": []
            }
        ]
      },
      {
        "id": "node_2",
        "level": 1,
        "header": "2. Main Content",
        "content": "Content of section 2...",
        "children": []
      }
    ]
  }
}
```

## 2. Python Implementation

You can manually build the tree using the classes from `rnsr.models` and then index it.

### Example Script

```python
from rnsr.models import DocumentNode, DocumentTree
from rnsr.indexing import build_skeleton_index
from rnsr.agent import run_navigator

# 1. Build your custom tree
root = DocumentNode(
    id="root",
    level=0,
    header="Employee Handbook",
    content="", # Root usually has no content
    children=[
        DocumentNode(
            id="sec_1", 
            level=1, 
            header="1. Leave Policy",
            content="Employees are entitled to 20 days...",
            children=[
                DocumentNode(
                    id="sec_1_1",
                    level=2,
                    header="1.1 Sick Leave",
                    content="Sick leave requires a doctor's note after 3 days...",
                    children=[]
                )
            ]
        )
    ]
)

tree = DocumentTree(title="Employee Handbook", root=root)

# 2. Convert to RNSR Indices (Skeleton + KV Store)
# This generates summaries for the "Skeleton" and stores full text in "KV Store"
print("Indexing custom tree...")
skeleton, kv_store = build_skeleton_index(tree)

# 3. Run the Agent
print("Running agent...")
result = run_navigator(
    question="When do I need a doctor's note?",
    skeleton=skeleton,
    kv_store=kv_store
)

print(f"Answer: {result['answer']}")
```

## 3. Key Requirements for Custom Trees

1.  **Hierarchy**: Ensure `level` is correct (0 for root, 1 for main sections, 2 for subsections). The agent utilizes this to know how deep it is.
2.  **Headers**: Critical for the "Tree of Thoughts" agent. The agent decides where to go based on the `header`. Make them descriptive.
3.  **Content**: The `content` field should contain the *full text* of that section.
4.  **Summarization**: When you call `build_skeleton_index(tree)`, the system will automatically generate summaries for the skeleton nodes (using an LLM if configured, or truncation otherwise). You don't need to provide summaries yourself.
