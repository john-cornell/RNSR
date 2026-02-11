#!/usr/bin/env python3
"""
RNSR Interactive Demo

A Gradio-based web interface for testing RNSR with your own documents.

Usage:
    python demo.py

Then open http://localhost:7860 in your browser.
"""
from __future__ import annotations  # Enable Python 3.10+ type hints on Python 3.9

import sys
print("Starting demo.py...", flush=True)

import os
from pathlib import Path
from typing import Any

print("Importing gradio...", flush=True)
import gradio as gr
print("Gradio imported.", flush=True)

# =============================================================================
# Monkeypatch Gradio to fix the Pydantic v2 schema parsing bug
# Only needed for Gradio < 6.0 -- Gradio 6 handles Pydantic v2 natively.
# Kept as a guarded safety net for users who haven't upgraded yet.
# =============================================================================
_gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 0

if _gradio_major < 6:
    import gradio_client.utils as client_utils

    _original_json_schema_to_python_type = client_utils._json_schema_to_python_type
    _original_get_type = client_utils.get_type

    def _patched_json_schema_to_python_type(schema, defs=None):
        """Patched version that handles boolean schemas (e.g., additionalProperties: true)."""
        if isinstance(schema, bool):
            return "any" if schema else "never"
        if schema is None:
            return "any"
        return _original_json_schema_to_python_type(schema, defs)

    def _patched_get_type(schema):
        """Patched version that handles boolean schemas."""
        if isinstance(schema, bool):
            return "any" if schema else "never"
        if schema is None:
            return "any"
        return _original_get_type(schema)

    client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
    client_utils.get_type = _patched_get_type

    _original_json_schema_to_python_type_public = client_utils.json_schema_to_python_type
    def _patched_json_schema_to_python_type_public(schema):
        if isinstance(schema, bool):
            return "any" if schema else "never"
        if schema is None:
            return "any"
        return _original_json_schema_to_python_type_public(schema)
    client_utils.json_schema_to_python_type = _patched_json_schema_to_python_type_public

# =============================================================================

print("Importing RNSR modules...", flush=True)
from rnsr import RNSRClient, ingest_document, build_skeleton_index
from rnsr.models import DocumentTree, DocumentNode
from rnsr.indexing import KVStore
from rnsr.indexing.knowledge_graph import KnowledgeGraph, InMemoryKnowledgeGraph
from rnsr.ingestion.pipeline import extract_entities_from_tree
from rnsr.extraction.models import Entity, EntityType
print("RNSR modules imported.", flush=True)
print("Defining SessionState class...", flush=True)

# Global state for the current document
class SessionState:
    def __init__(self):
        self.tree: DocumentTree | None = None
        self.skeleton: dict[str, Any] | None = None
        self.kv_store: KVStore | None = None
        self.knowledge_graph: InMemoryKnowledgeGraph | None = None
        self.document_name: str = ""
        self.document_path: str | None = None  # Track full path for RNSRClient
        self.chat_history: list[tuple[str, str]] = []
        self.entities: list[Entity] = []
        self.extraction_stats: dict[str, Any] = {}


state = SessionState()

# Global RNSRClient instance (uses caching for performance)
client = RNSRClient(cache_dir=".rnsr_demo_cache")
print("SessionState and RNSRClient created.", flush=True)


def process_document(file, extract_entities: bool = True, progress=gr.Progress()):
    """Process an uploaded PDF document with live status updates."""
    import time
    
    if file is None:
        yield "❌ Please upload a PDF file."
        return
    
    try:
        # Get the file path
        file_path = file if isinstance(file, str) else file
        file_name = Path(file_path).name
        
        # Store document path for RNSRClient
        state.document_path = file_path
        state.document_name = file_name
        state.chat_history = []
        
        # Step 1: Ingest the document
        yield f"""⏳ **Processing: {file_name}**

📄 Step 1/4: Reading PDF and extracting text..."""
        
        progress(0.1, desc="📄 Reading PDF...")
        result = ingest_document(file_path)
        state.tree = result.tree
        
        # Get document stats early for time estimates
        node_count = state.tree.total_nodes if state.tree else 0
        root_sections = len(state.tree.root.children) if state.tree and state.tree.root else 0
        
        # Calculate time estimates
        base_time_sec = 10  # Base processing time
        entity_time_sec = node_count * 45 if extract_entities else 0  # ~45 sec per node for entity extraction
        total_est_sec = base_time_sec + entity_time_sec
        
        if extract_entities:
            time_str = f"~{total_est_sec // 60}-{(total_est_sec // 60) + 2} minutes"
        else:
            time_str = "~10-30 seconds"
        
        # Step 2: Build the skeleton index
        yield f"""⏳ **Processing: {file_name}**

📄 Step 1/4: ✅ Text extracted
🌳 Step 2/4: Building document hierarchy...

📊 Found **{node_count} sections** | ⏱️ Est. time: {time_str}"""
        
        progress(0.3, desc="🌳 Building hierarchy...")
        state.skeleton, state.kv_store = build_skeleton_index(state.tree)
        
        # Initialize knowledge graph
        state.knowledge_graph = InMemoryKnowledgeGraph()
        state.entities = []
        state.extraction_stats = {}
        
        # Step 3: Extract entities if enabled
        entity_info = ""
        if extract_entities and state.tree:
            start_time = time.time()
            
            yield f"""⏳ **Processing: {file_name}**

📄 Step 1/4: ✅ Text extracted
🌳 Step 2/4: ✅ Hierarchy built ({node_count} nodes)
🔍 Step 3/4: Extracting entities (0/{node_count} nodes)...

⏱️ **Estimated: {total_est_sec // 60}-{(total_est_sec // 60) + 2} minutes remaining**
*This step uses AI to identify people, organizations, dates, etc.*"""
            
            try:
                progress(0.4, desc=f"🔍 Extracting entities (0/{node_count})...")
                extraction = extract_entities_from_tree(state.tree)
                state.entities = extraction.get("entities", [])
                state.extraction_stats = extraction.get("stats", {})
                
                elapsed = int(time.time() - start_time)
                
                yield f"""⏳ **Processing: {file_name}**

📄 Step 1/4: ✅ Text extracted
🌳 Step 2/4: ✅ Hierarchy built ({node_count} nodes)
🔍 Step 3/4: ✅ Entities extracted ({len(state.entities)} found in {elapsed}s)
🔗 Step 4/4: Building knowledge graph...

⏱️ **Almost done...**"""
                
                progress(0.8, desc="🔗 Building knowledge graph...")
                # Store entities in knowledge graph
                for entity in state.entities:
                    state.knowledge_graph.add_entity(entity)
                
                # Store relationships
                for rel in extraction.get("relationships", []):
                    state.knowledge_graph.add_relationship(rel)
                
                # Format entity summary
                entity_counts = state.extraction_stats.get("entity_types", {})
                if entity_counts:
                    entity_parts = [f"{v} {k}s" for k, v in entity_counts.items()]
                    entity_info = f"\n🔍 **Entities:** {', '.join(entity_parts)}"
                    entity_info += f"\n🔗 **Relationships:** {state.extraction_stats.get('relationships_extracted', 0)}"
                
            except Exception as e:
                entity_info = f"\n⚠️ Entity extraction skipped: {str(e)[:50]}"
        else:
            yield f"""⏳ **Processing: {file_name}**

📄 Step 1/4: ✅ Text extracted
🌳 Step 2/4: ✅ Hierarchy built ({node_count} nodes)
⚡ Step 3/4: Preparing for Q&A...

⏱️ **Almost done...**"""
        
        # Step 4: Pre-warm the RNSRClient cache
        progress(0.9, desc="⚡ Preparing for questions...")
        try:
            client._get_or_create_index(Path(file_path), force_reindex=False)
        except Exception:
            pass  # Non-critical, will be created on first question
        
        progress(1.0, desc="✅ Ready!")
        
        # Final success message
        if extract_entities and state.entities:
            yield f"""✅ **Document ready for questions!**

📄 **File:** {state.document_name}
🌳 **Hierarchy nodes:** {node_count}
📊 **Root sections:** {root_sections}{entity_info}

💬 **Ask questions in the chat below.** Entity visualization available in the tabs."""
        else:
            yield f"""✅ **Document ready for questions!**

📄 **File:** {state.document_name}
🌳 **Hierarchy nodes:** {node_count}
📊 **Root sections:** {root_sections}

💬 **Ask questions in the chat below.**
*The AI will extract entities on-the-fly for accurate answers.*"""
        
    except Exception as e:
        import traceback
        yield f"❌ Error processing document: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def chat(message: str, history: list, progress=gr.Progress()) -> tuple[str, list]:
    """Handle a chat message and return the response."""
    if state.document_path is None:
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "❌ Please upload and process a document first using the 'Process Document' button."}
        ]
    
    if not message.strip():
        return "", history
    
    try:
        progress(0.2, desc="🔍 Searching document...")
        
        # Use RNSRClient.ask_advanced() with knowledge graph for best accuracy
        # This matches the benchmark's zero-hallucination performance
        progress(0.4, desc="🧠 Analyzing relevant sections...")
        result = client.ask_advanced(
            document=state.document_path,
            question=message,
            use_knowledge_graph=True,   # Entity extraction for better accuracy
            enable_verification=False,  # Avoid overly strict critic
        )
        
        progress(0.8, desc="✍️ Generating answer...")
        
        # Format the response
        answer = result.get("answer", str(result))
        confidence = result.get("confidence", None)
        
        # Build response with confidence if available
        if confidence is not None:
            response = f"{answer}\n\n*Confidence: {confidence:.0%}*"
        else:
            response = str(answer)
        
        # Add entity context from local knowledge graph if available
        if state.knowledge_graph and state.entities:
            mentioned_entities = []
            query_lower = message.lower()
            
            for entity in state.entities:
                if entity.canonical_name.lower() in query_lower:
                    mentioned_entities.append(entity)
                else:
                    for alias in entity.aliases:
                        if alias.lower() in query_lower:
                            mentioned_entities.append(entity)
                            break
            
            if mentioned_entities:
                context_parts = []
                for entity in mentioned_entities[:3]:
                    co_mentions = state.knowledge_graph.get_entities_mentioned_together(entity.id)
                    related = [e.canonical_name for e, _ in co_mentions[:3]]
                    context_parts.append(
                        f"- {entity.canonical_name} ({entity.type.value})"
                        + (f", related to: {', '.join(related)}" if related else "")
                    )
                response += "\n\n*Entity context:*\n" + "\n".join(context_parts)
        
        progress(1.0, desc="✅ Done")
        
        # Update history with Gradio 6.x message format
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        state.chat_history = new_history
        
        return "", new_history
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]


def clear_chat() -> tuple[str, list]:
    """Clear the chat history."""
    state.chat_history = []
    return "", []


def get_tree_visualization() -> str:
    """Generate a text visualization of the document tree."""
    if state.tree is None:
        return "No document loaded."
    
    lines = []
    _visualize_node(state.tree.root, lines, "", True)
    return "\n".join(lines)


def _visualize_node(node: DocumentNode, lines: list, prefix: str, is_last: bool):
    """Recursively build tree visualization."""
    connector = "└── " if is_last else "├── "
    
    # Truncate title for display
    title = node.header or "(untitled)"
    if len(title) > 60:
        title = title[:57] + "..."
    
    lines.append(f"{prefix}{connector}{title}")
    
    if node.children:
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            _visualize_node(child, lines, child_prefix, i == len(node.children) - 1)


def get_entities_visualization() -> str:
    """Generate a visualization of extracted entities."""
    if not state.entities:
        if state.document_path:
            return """**No entities extracted for visualization.**

To see entities here, re-process the document with "Extract entities" enabled.

*Note: Q&A still works - the AI extracts entities on-the-fly when answering questions.*"""
        return "No entities extracted. Process a document first."
    
    lines = []
    lines.append(f"## 📊 Extracted Entities ({len(state.entities)} total)\n")
    
    # Group by type
    by_type: dict[str, list[Entity]] = {}
    for entity in state.entities:
        type_name = entity.type.value
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(entity)
    
    # Format each type group
    type_icons = {
        "person": "👤",
        "organization": "🏢",
        "date": "📅",
        "event": "📌",
        "legal_concept": "⚖️",
        "location": "📍",
        "reference": "📎",
        "monetary": "💰",
        "document": "📄",
        "other": "🏷️",
    }
    
    for type_name, entities in sorted(by_type.items()):
        icon = type_icons.get(type_name, "•")
        display_name = type_name.replace('_', ' ').title()
        lines.append(f"\n### {icon} {display_name} ({len(entities)})\n")
        
        for entity in entities[:15]:  # Limit to 15 per type
            mentions = len(entity.mentions)
            aliases = f" (aka: {', '.join(entity.aliases[:2])})" if entity.aliases else ""
            
            # For OTHER type, show the original type if available
            original_type = ""
            if type_name == "other" and entity.metadata.get("original_type"):
                original_type = f" [{entity.metadata['original_type']}]"
            
            lines.append(f"- **{entity.canonical_name}**{original_type}{aliases} - {mentions} mention(s)")
        
        if len(entities) > 15:
            lines.append(f"  ... and {len(entities) - 15} more")
    
    return "\n".join(lines)


def get_relationships_visualization() -> str:
    """Generate a visualization of entity relationships."""
    if not state.knowledge_graph or not state.entities:
        if state.document_path:
            return """**No relationships extracted for visualization.**

To see relationships here, re-process the document with "Extract entities" enabled.

*Note: Q&A still works - the AI builds its own knowledge graph when answering questions.*"""
        return "No knowledge graph available. Process a document first."
    
    stats = state.knowledge_graph.get_stats()
    
    if stats.get("relationship_count", 0) == 0:
        return "No relationships extracted yet."
    
    lines = []
    lines.append(f"## 🔗 Entity Relationships\n")
    lines.append(f"**Total relationships:** {stats.get('relationship_count', 0)}\n")
    
    # Get some example relationships
    if state.entities:
        lines.append("\n### Key Connections\n")
        
        shown = 0
        for entity in state.entities[:10]:
            rels = state.knowledge_graph.get_entity_relationships(entity.id)
            for rel in rels[:3]:
                # Get target entity name
                target = state.knowledge_graph.get_entity(rel.target_id)
                target_name = target.canonical_name if target else rel.target_id
                
                rel_type = rel.type.value.replace("_", " ").title()
                lines.append(f"- {entity.canonical_name} → **{rel_type}** → {target_name}")
                shown += 1
                
                if shown >= 20:
                    break
            if shown >= 20:
                break
        
        if shown == 0:
            lines.append("*(No entity-to-entity relationships found)*")
    
    return "\n".join(lines)


def search_entities(query: str) -> str:
    """Search for entities matching a query."""
    if not state.knowledge_graph:
        return "No knowledge graph available."
    
    if not query.strip():
        return "Enter a search term."
    
    results = state.knowledge_graph.find_entities_by_name(query.strip(), fuzzy=True)
    
    if not results:
        return f"No entities found matching '{query}'."
    
    lines = [f"### Search Results for '{query}'\n"]
    
    for entity in results[:10]:
        type_name = entity.type.value.replace("_", " ").title()
        mentions = len(entity.mentions)
        
        lines.append(f"**{entity.canonical_name}** ({type_name})")
        lines.append(f"  - Mentions: {mentions}")
        
        if entity.aliases:
            lines.append(f"  - Also known as: {', '.join(entity.aliases[:3])}")
        
        # Get related entities
        co_mentions = state.knowledge_graph.get_entities_mentioned_together(entity.id)
        if co_mentions:
            related = [e.canonical_name for e, _ in co_mentions[:3]]
            lines.append(f"  - Related to: {', '.join(related)}")
        
        lines.append("")
    
    return "\n".join(lines)


def get_tables_list() -> str:
    """List all detected tables in the document."""
    if not state.document_path:
        return "No document loaded. Upload and process a document first."
    
    try:
        tables = client.list_tables(state.document_path)
        
        if not tables:
            return """**No tables detected in this document.**

Tables are automatically detected during document processing. This document may not contain any structured tables, or the tables may not be in a recognizable format (markdown/ASCII).

*Tip: Tables work best when they use consistent formatting with headers and delimiters.*"""
        
        lines = [f"## 📊 Detected Tables ({len(tables)} found)\n"]
        
        for i, table in enumerate(tables, 1):
            headers_str = ", ".join(table["headers"][:5])
            if len(table["headers"]) > 5:
                headers_str += f" ... (+{len(table['headers']) - 5} more)"
            
            table_label = table.get('title') or f"Table {table['id']}"
            lines.append(f"### {i}. {table_label}")
            lines.append(f"- **ID:** `{table['id']}`")
            lines.append(f"- **Rows:** {table['num_rows']} | **Columns:** {table['num_cols']}")
            lines.append(f"- **Headers:** {headers_str}")
            if table.get("page_num"):
                lines.append(f"- **Page:** {table['page_num']}")
            lines.append("")
        
        lines.append("\n*Use the Table Query tab to run SQL-like queries on these tables.*")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error listing tables: {str(e)}"


def get_table_dropdown_choices() -> list[str]:
    """Get table IDs for dropdown."""
    if not state.document_path:
        return []
    
    try:
        tables = client.list_tables(state.document_path)
        return [f"{t['id']} - {t.get('title') or 'Untitled'} ({t['num_rows']} rows)" for t in tables]
    except Exception:
        return []


def query_table_ui(
    table_selection: str,
    columns: str,
    where_column: str,
    where_op: str,
    where_value: str,
    order_by: str,
    limit: int,
) -> str:
    """Run a SQL-like query on a table."""
    if not state.document_path:
        return "No document loaded. Upload and process a document first."
    
    if not table_selection:
        return "Please select a table from the dropdown."
    
    try:
        # Extract table ID from selection (format: "table_001 - Title (N rows)")
        table_id = table_selection.split(" - ")[0].strip()
        
        # Parse columns
        cols = None
        if columns.strip():
            cols = [c.strip() for c in columns.split(",") if c.strip()]
        
        # Parse where clause
        where = None
        if where_column.strip() and where_value.strip():
            if where_op in ("==", "!=", ">", ">=", "<", "<="):
                # Try numeric conversion
                try:
                    val = float(where_value.replace(",", "").replace("$", "").strip())
                    where = {where_column.strip(): {"op": where_op, "value": val}}
                except ValueError:
                    where = {where_column.strip(): {"op": where_op, "value": where_value}}
            elif where_op == "contains":
                where = {where_column.strip(): {"op": "contains", "value": where_value}}
            else:
                where = {where_column.strip(): where_value}
        
        # Parse order by
        order = None
        if order_by.strip():
            order = order_by.strip()
        
        # Parse limit
        lim = limit if limit > 0 else None
        
        # Run query
        results = client.query_table(
            document=state.document_path,
            table_id=table_id,
            columns=cols,
            where=where,
            order_by=order,
            limit=lim,
        )
        
        if not results:
            return f"**Query returned 0 rows.**\n\nTable: `{table_id}`"
        
        # Format as markdown table
        lines = [f"### Query Results ({len(results)} rows)\n"]
        
        # Get headers from first row
        headers = list(results[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in results:
            values = [str(row.get(h, ""))[:50] for h in headers]  # Truncate long values
            lines.append("| " + " | ".join(values) + " |")
        
        return "\n".join(lines)
        
    except Exception as e:
        import traceback
        return f"**Query Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def run_aggregation(table_selection: str, column: str, operation: str) -> str:
    """Run an aggregation on a table column."""
    if not state.document_path:
        return "No document loaded."
    
    if not table_selection or not column.strip() or not operation:
        return "Please select a table, column, and operation."
    
    try:
        table_id = table_selection.split(" - ")[0].strip()
        
        result = client.aggregate_table(
            document=state.document_path,
            table_id=table_id,
            column=column.strip(),
            operation=operation,
        )
        
        # Format nicely based on operation
        op_names = {
            "sum": "Sum",
            "avg": "Average",
            "count": "Count",
            "min": "Minimum",
            "max": "Maximum",
        }
        
        if operation == "avg":
            formatted = f"{result:,.2f}"
        elif isinstance(result, float) and result == int(result):
            formatted = f"{int(result):,}"
        else:
            formatted = f"{result:,.2f}" if isinstance(result, float) else str(result)
        
        return f"### {op_names.get(operation, operation)} of `{column}`\n\n# {formatted}"
        
    except Exception as e:
        return f"**Aggregation Error:** {str(e)}"


print("All functions defined.", flush=True)

# Example questions users can click
EXAMPLE_QUESTIONS = [
    "What are the main sections of this document?",
    "Who are the key people mentioned?",
    "Summarize the key findings",
    "What dates are mentioned?",
]


# Create the Gradio interface
def create_demo():
    print("Creating demo...", flush=True)
    print("Creating gr.Blocks...", flush=True)
    
    # Use a modern theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        block_title_text_weight="600",
        block_border_width="1px",
        block_shadow="0 1px 3px rgba(0,0,0,0.1)",
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
    )
    
    custom_css = """
        .main-header { text-align: center; margin-bottom: 1rem; }
        .status-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1rem;
            color: white;
        }
        .example-btn { font-size: 0.85rem !important; }
        footer { text-align: center; opacity: 0.7; font-size: 0.85rem; margin-top: 2rem; }
    """
    
    # Gradio 6 moved theme/css to launch(); Gradio 4/5 takes them in the constructor
    blocks_kwargs = {"title": "RNSR - Document Q&A"}
    if _gradio_major < 6:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = custom_css
    
    with gr.Blocks(**blocks_kwargs) as demo:
        print("Inside gr.Blocks context...", flush=True)
        
        # Header
        gr.Markdown("""
        <div class="main-header">
        
        # 🔍 RNSR - Document Q&A
        ### Recursive Neural-Symbolic Retriever with Knowledge Graph
        
        </div>
        """)
        
        # Quick start steps
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Step 1:** Upload PDF")
            with gr.Column(scale=1):
                gr.Markdown("**Step 2:** Click Process")
            with gr.Column(scale=1):
                gr.Markdown("**Step 3:** Ask Questions")
        
        gr.Markdown("---")
        
        with gr.Row(equal_height=False):
            # Left column - Document & Settings
            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    gr.Markdown("### 📄 Document")
                    file_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath",
                        file_count="single",
                    )
                    process_btn = gr.Button(
                        "🚀 Process Document", 
                        variant="primary",
                        size="lg",
                    )
                
                # Status card
                status_output = gr.Markdown("""
**Status:** Waiting for document...

Upload a PDF above and click **Process Document** to begin.
""")
                
                # Advanced options in accordion
                with gr.Accordion("⚙️ Advanced Options", open=False):
                    extract_entities_cb = gr.Checkbox(
                        label="Extract entities for visualization",
                        value=False,
                        info="Adds 5-10 min. Not needed for Q&A."
                    )
                    gr.Markdown("*Entity extraction runs AI on each section for the visualization tabs. Q&A works without this.*")
                
                # Document info tabs
                with gr.Tabs():
                    with gr.TabItem("🌳 Structure"):
                        tree_output = gr.Textbox(
                            label="Document Hierarchy",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            show_label=False,
                            placeholder="Process a document to see its structure..."
                        )
                        refresh_tree_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    with gr.TabItem("🔍 Entities"):
                        entities_output = gr.Markdown("*Process a document with entity extraction enabled to see entities*")
                        refresh_entities_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    with gr.TabItem("🔗 Relationships"):
                        relationships_output = gr.Markdown("*Process a document with entity extraction enabled to see relationships*")
                        refresh_rels_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    with gr.TabItem("🔎 Search"):
                        entity_search = gr.Textbox(
                            label="Search entities",
                            placeholder="Enter name to search...",
                            show_label=False,
                        )
                        search_btn = gr.Button("🔍 Search", size="sm")
                        search_results = gr.Markdown("")
                    
                    with gr.TabItem("📊 Tables"):
                        tables_output = gr.Markdown("*Process a document to detect tables*")
                        refresh_tables_btn = gr.Button("🔄 Refresh Tables", size="sm")
            
            # Right column - Chat & Tables
            with gr.Column(scale=2, min_width=400):
                with gr.Tabs():
                    # Chat Tab
                    with gr.TabItem("💬 Chat"):
                        chatbot_kwargs = dict(
                            label="Conversation",
                            height=350,
                            show_label=False,
                            placeholder="📄 Upload and process a document first, then ask questions here...",
                            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=rnsr"),
                        )
                        if _gradio_major >= 6:
                            chatbot_kwargs["buttons"] = ["copy"]
                        else:
                            chatbot_kwargs["show_copy_button"] = True
                        chatbot = gr.Chatbot(**chatbot_kwargs)
                
                # Example questions as clickable buttons
                gr.Markdown("**Try an example:**")
                with gr.Row():
                    example_btns = []
                    for i, q in enumerate(EXAMPLE_QUESTIONS):
                        btn = gr.Button(
                            q[:35] + "..." if len(q) > 35 else q,
                            size="sm",
                            variant="secondary",
                            elem_classes=["example-btn"]
                        )
                        example_btns.append((btn, q))
                
                # Chat input
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your question",
                        placeholder="Type your question here...",
                        scale=5,
                        show_label=False,
                        container=False,
                    )
                    submit_btn = gr.Button("➤ Send", variant="primary", scale=1, min_width=100)
                
                    with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Chat", size="sm", variant="secondary")
                    
                    # Table Query Tab
                    with gr.TabItem("📊 Table Query"):
                        gr.Markdown("### SQL-like Table Queries")
                        gr.Markdown("*Query detected tables with SELECT, WHERE, ORDER BY, and aggregations.*")
                        
                        with gr.Row():
                            table_dropdown = gr.Dropdown(
                                label="Select Table",
                                choices=[],
                                interactive=True,
                                scale=2,
                            )
                            refresh_dropdown_btn = gr.Button("🔄", size="sm", scale=0)
                        
                        with gr.Accordion("Query Options", open=True):
                            columns_input = gr.Textbox(
                                label="Columns (comma-separated, leave empty for all)",
                                placeholder="Name, Amount, Date",
                            )
                            
                            with gr.Row():
                                where_col = gr.Textbox(label="Where Column", placeholder="Amount", scale=1)
                                where_op = gr.Dropdown(
                                    label="Operator",
                                    choices=["==", "!=", ">", ">=", "<", "<=", "contains"],
                                    value=">=",
                                    scale=1,
                                )
                                where_val = gr.Textbox(label="Value", placeholder="1000", scale=1)
                            
                            with gr.Row():
                                order_input = gr.Textbox(
                                    label="Order By (prefix with - for DESC)",
                                    placeholder="-Amount",
                                    scale=2,
                                )
                                limit_input = gr.Number(label="Limit", value=10, precision=0, scale=1)
                        
                        query_btn = gr.Button("▶ Run Query", variant="primary")
                        query_results = gr.Markdown("*Select a table and run a query to see results*")
                        
                        gr.Markdown("---")
                        gr.Markdown("### Aggregations")
                        
                        with gr.Row():
                            agg_col = gr.Textbox(label="Column", placeholder="Revenue", scale=2)
                            agg_op = gr.Dropdown(
                                label="Operation",
                                choices=["sum", "avg", "count", "min", "max"],
                                value="sum",
                                scale=1,
                            )
                            agg_btn = gr.Button("▶ Calculate", variant="secondary", scale=1)
                        
                        agg_result = gr.Markdown("")
        
        # Footer
        gr.Markdown("""
        <footer>
        
        ---
        **RNSR** - Recursive Neural-Symbolic Retriever | 
        [GitHub](https://github.com/theeufj/RNSR) | 
        Built with knowledge graph extraction for accurate, grounded answers
        
        </footer>
        """)
        
        # Event handlers
        
        # Process document and auto-refresh structure
        def process_and_refresh(file, extract_entities):
            """Process document and return status, tree, tables, and dropdown choices."""
            for status in process_document(file, extract_entities):
                tables_list = get_tables_list()
                dropdown_choices = get_table_dropdown_choices()
                if _gradio_major >= 6:
                    dropdown_update = gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None)
                else:
                    dropdown_update = gr.update(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None)
                yield status, get_tree_visualization(), tables_list, dropdown_update
        
        process_btn.click(
            fn=process_and_refresh,
            inputs=[file_input, extract_entities_cb],
            outputs=[status_output, tree_output, tables_output, table_dropdown]
        )
        
        refresh_tree_btn.click(
            fn=get_tree_visualization,
            inputs=[],
            outputs=[tree_output]
        )
        
        refresh_entities_btn.click(
            fn=get_entities_visualization,
            inputs=[],
            outputs=[entities_output]
        )
        
        refresh_rels_btn.click(
            fn=get_relationships_visualization,
            inputs=[],
            outputs=[relationships_output]
        )
        
        search_btn.click(
            fn=search_entities,
            inputs=[entity_search],
            outputs=[search_results]
        )
        
        entity_search.submit(
            fn=search_entities,
            inputs=[entity_search],
            outputs=[search_results]
        )
        
        refresh_tables_btn.click(
            fn=get_tables_list,
            inputs=[],
            outputs=[tables_output]
        )
        
        # Table query handlers
        def update_table_dropdown():
            choices = get_table_dropdown_choices()
            if _gradio_major >= 6:
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            else:
                return gr.update(choices=choices, value=choices[0] if choices else None)
        
        refresh_dropdown_btn.click(
            fn=update_table_dropdown,
            inputs=[],
            outputs=[table_dropdown]
        )
        
        query_btn.click(
            fn=query_table_ui,
            inputs=[table_dropdown, columns_input, where_col, where_op, where_val, order_input, limit_input],
            outputs=[query_results]
        )
        
        agg_btn.click(
            fn=run_aggregation,
            inputs=[table_dropdown, agg_col, agg_op],
            outputs=[agg_result]
        )
        
        # Example question buttons - fill input and submit
        def use_example(question):
            """Put example question in input box."""
            return question
        
        for btn, question in example_btns:
            btn.click(
                fn=lambda q=question: q,
                inputs=[],
                outputs=[msg_input]
            )
        
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        submit_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[msg_input, chatbot]
        )
    
    # Store theme and css on the demo object so launch() can use them
    demo._rnsr_theme = theme
    demo._rnsr_css = custom_css
    print("Demo created, returning...", flush=True)
    return demo


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: No API key found. Set GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
        print("   Example: export ANTHROPIC_API_KEY='your-key-here'")
        print()
    
    print("🚀 Starting RNSR Demo...")
    print("   Open http://localhost:7860 in your browser")
    print()
    
    demo = create_demo()
    print("Launching demo server...", flush=True)
    launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Disabled to avoid slow share link creation
    )
    # Gradio 6 accepts theme/css in launch(); older versions already got them in Blocks()
    if _gradio_major >= 6:
        launch_kwargs["theme"] = demo._rnsr_theme
        launch_kwargs["css"] = demo._rnsr_css
    demo.launch(**launch_kwargs)
