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
# The bug occurs because Pydantic v2 uses `additionalProperties: true` (a boolean)
# but Gradio's _json_schema_to_python_type expects a dict schema
# =============================================================================
import gradio_client.utils as client_utils

# Store original functions
_original_json_schema_to_python_type = client_utils._json_schema_to_python_type
_original_get_type = client_utils.get_type

def _patched_json_schema_to_python_type(schema, defs=None):
    """Patched version that handles boolean schemas (e.g., additionalProperties: true)."""
    # Handle boolean schemas - these mean "any" in JSON Schema
    if isinstance(schema, bool):
        return "any" if schema else "never"
    # Handle None schema
    if schema is None:
        return "any"
    # Call original function
    return _original_json_schema_to_python_type(schema, defs)

def _patched_get_type(schema):
    """Patched version that handles boolean schemas."""
    if isinstance(schema, bool):
        return "any" if schema else "never"
    if schema is None:
        return "any"
    return _original_get_type(schema)

# Apply the patches
client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
client_utils.get_type = _patched_get_type

# Also patch json_schema_to_python_type to handle booleans at the top level
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
from rnsr import ingest_document, build_skeleton_index, run_navigator
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
        self.chat_history: list[tuple[str, str]] = []
        self.entities: list[Entity] = []
        self.extraction_stats: dict[str, Any] = {}


state = SessionState()
print("SessionState created.", flush=True)


def process_document(file, extract_entities: bool = True) -> str:
    """Process an uploaded PDF document."""
    if file is None:
        return "❌ Please upload a PDF file."
    
    try:
        # Get the file path
        file_path = file if isinstance(file, str) else file
        
        # Ingest the document
        result = ingest_document(file_path)
        state.tree = result.tree
        
        # Build the skeleton index
        state.skeleton, state.kv_store = build_skeleton_index(state.tree)
        state.document_name = Path(file_path).name
        state.chat_history = []
        
        # Initialize knowledge graph
        state.knowledge_graph = InMemoryKnowledgeGraph()
        state.entities = []
        state.extraction_stats = {}
        
        # Get document stats
        node_count = state.tree.total_nodes if state.tree else 0
        root_sections = len(state.tree.root.children) if state.tree and state.tree.root else 0
        
        # Extract entities if enabled
        entity_info = ""
        if extract_entities and state.tree:
            try:
                extraction = extract_entities_from_tree(state.tree)
                state.entities = extraction.get("entities", [])
                state.extraction_stats = extraction.get("stats", {})
                
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
        
        return f"""✅ **Document processed successfully!**

📄 **File:** {state.document_name}
🌳 **Hierarchy nodes:** {node_count}
📊 **Root sections:** {root_sections}{entity_info}

You can now ask questions about this document in the chat below."""
        
    except Exception as e:
        import traceback
        return f"❌ Error processing document: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def chat(message: str, history: list) -> tuple[str, list]:
    """Handle a chat message and return the response."""
    if state.skeleton is None or state.kv_store is None:
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "❌ Please upload a document first before asking questions."}
        ]
    
    if not message.strip():
        return "", history
    
    try:
        # Check if this is an entity-related query and we have entities
        entity_context = ""
        if state.knowledge_graph and state.entities:
            # Search for mentioned entities in the query
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
            
            # If we found entities, add context
            if mentioned_entities:
                context_parts = []
                for entity in mentioned_entities[:3]:
                    # Get related entities
                    co_mentions = state.knowledge_graph.get_entities_mentioned_together(entity.id)
                    related = [e.canonical_name for e, _ in co_mentions[:3]]
                    
                    context_parts.append(
                        f"- {entity.canonical_name} ({entity.type.value})"
                        + (f", related to: {', '.join(related)}" if related else "")
                    )
                
                entity_context = "\n\n*Entity context:*\n" + "\n".join(context_parts)
        
        # Run the navigator agent
        answer = run_navigator(
            question=message,
            skeleton=state.skeleton,
            kv_store=state.kv_store
        )
        
        # Format the response
        response = str(answer)
        
        # Append entity context if available
        if entity_context:
            response += entity_context
        
        # Update history with Gradio 6.x message format
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        state.chat_history = new_history
        
        return "", new_history
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}"
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
    if not state.knowledge_graph:
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


print("All functions defined.", flush=True)

# Create the Gradio interface
def create_demo():
    print("Creating demo...", flush=True)
    print("Creating gr.Blocks...", flush=True)
    with gr.Blocks(title="RNSR - Document Q&A") as demo:
        print("Inside gr.Blocks context...", flush=True)
        gr.Markdown("""
        # 🔍 RNSR - Recursive Neural-Symbolic Retriever
        
        Upload a PDF document and ask questions about it. RNSR preserves the document's 
        hierarchical structure and extracts entities for better context understanding.
        
        **Features:**
        - 📊 Hierarchical document structure extraction
        - 🔍 Entity extraction (people, organizations, dates, etc.)
        - 🔗 Relationship mapping between entities
        - 💬 Natural language Q&A with structural context
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Upload")
                file_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                extract_entities_cb = gr.Checkbox(
                    label="Extract entities",
                    value=True,
                    info="Enable entity and relationship extraction"
                )
                process_btn = gr.Button("🔄 Process Document", variant="primary")
                status_output = gr.Markdown("*No document loaded*")
                
                with gr.Tabs():
                    with gr.TabItem("🌳 Structure"):
                        tree_output = gr.Textbox(
                            label="Document Hierarchy",
                            lines=12,
                            max_lines=25,
                            interactive=False
                        )
                        refresh_tree_btn = gr.Button("Refresh", size="sm")
                    
                    with gr.TabItem("🔍 Entities"):
                        entities_output = gr.Markdown("*Process a document to see entities*")
                        refresh_entities_btn = gr.Button("Refresh", size="sm")
                    
                    with gr.TabItem("🔗 Relationships"):
                        relationships_output = gr.Markdown("*Process a document to see relationships*")
                        refresh_rels_btn = gr.Button("Refresh", size="sm")
                    
                    with gr.TabItem("🔎 Search"):
                        entity_search = gr.Textbox(
                            label="Search entities",
                            placeholder="Enter name to search...",
                        )
                        search_btn = gr.Button("Search", size="sm")
                        search_results = gr.Markdown("")
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with your Document")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=450,
                    show_label=False,
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask a question about the document...",
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ Clear Chat")
        
        gr.Markdown("""
        ---
        ### Example Questions
        - "What are the main sections of this document?"
        - "Who are the key people mentioned?"
        - "What happened on [specific date]?"
        - "What is the relationship between [Person A] and [Organization B]?"
        - "Summarize the key findings"
        
        ---
        *Powered by RNSR with Ontological Entity Understanding - [GitHub](https://github.com/theeufj/RNSR)*
        """)
        
        # Event handlers
        process_btn.click(
            fn=process_document,
            inputs=[file_input, extract_entities_cb],
            outputs=[status_output]
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
    # Disable API docs to workaround the Gradio/Pydantic TypeError
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Disabled to avoid slow share link creation
    )
