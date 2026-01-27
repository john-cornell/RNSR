#!/usr/bin/env python3
"""
RNSR Interactive Demo

A Gradio-based web interface for testing RNSR with your own documents.

Usage:
    python demo.py

Then open http://localhost:7860 in your browser.
"""

import os
from pathlib import Path
from typing import Any

import gradio as gr

from rnsr import ingest_document, build_skeleton_index, run_navigator
from rnsr.models import DocumentTree, DocumentNode
from rnsr.indexing import KVStore


# Global state for the current document
class SessionState:
    def __init__(self):
        self.tree: DocumentTree | None = None
        self.skeleton: dict[str, Any] | None = None
        self.kv_store: KVStore | None = None
        self.document_name: str = ""
        self.chat_history: list[tuple[str, str]] = []


state = SessionState()


def process_document(file) -> str:
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
        
        # Get document stats
        node_count = state.tree.total_nodes if state.tree else 0
        root_sections = len(state.tree.root.children) if state.tree and state.tree.root else 0
        
        return f"""✅ **Document processed successfully!**

📄 **File:** {state.document_name}
🌳 **Hierarchy nodes:** {node_count}
📊 **Root sections:** {root_sections}

You can now ask questions about this document in the chat below."""
        
    except Exception as e:
        import traceback
        return f"❌ Error processing document: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def chat(message: str, history: list) -> tuple[str, list]:
    """Handle a chat message and return the response."""
    if state.skeleton is None or state.kv_store is None:
        return "", history + [
            (message, "❌ Please upload a document first before asking questions.")
        ]
    
    if not message.strip():
        return "", history
    
    try:
        # Run the navigator agent
        answer = run_navigator(
            question=message,
            skeleton=state.skeleton,
            kv_store=state.kv_store
        )
        
        # Format the response
        response = str(answer)
        
        # Update history
        new_history = history + [(message, response)]
        state.chat_history = new_history
        
        return "", new_history
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}"
        return "", history + [(message, error_msg)]


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


# Create the Gradio interface
def create_demo():
    with gr.Blocks(title="RNSR - Document Q&A") as demo:
        gr.Markdown("""
        # 🔍 RNSR - Recursive Neural-Symbolic Retriever
        
        Upload a PDF document and ask questions about it. RNSR preserves the document's 
        hierarchical structure for better context understanding.
        
        **How it works:**
        1. Upload a PDF document
        2. RNSR extracts the document hierarchy using font analysis
        3. Ask questions in natural language
        4. Get accurate answers with structural context
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Upload")
                file_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                process_btn = gr.Button("🔄 Process Document", variant="primary")
                status_output = gr.Markdown("*No document loaded*")
                
                with gr.Accordion("🌳 Document Structure", open=False):
                    tree_output = gr.Textbox(
                        label="Hierarchy",
                        lines=15,
                        max_lines=30,
                        interactive=False
                    )
                    refresh_tree_btn = gr.Button("Refresh Tree View")
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with your Document")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=False
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
        - "Summarize the key points"
        - "What does section 3.2 discuss?"
        - "Find information about [specific topic]"
        
        ---
        *Powered by RNSR - [GitHub](https://github.com/theeufj/RNSR)*
        """)
        
        # Event handlers
        process_btn.click(
            fn=process_document,
            inputs=[file_input],
            outputs=[status_output]
        )
        
        refresh_tree_btn.click(
            fn=get_tree_visualization,
            inputs=[],
            outputs=[tree_output]
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
    
    return demo


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY")
        print("   Example: export GOOGLE_API_KEY='your-key-here'")
        print()
    
    print("🚀 Starting RNSR Demo...")
    print("   Open http://localhost:7860 in your browser")
    print()
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
