"""
Node 6: RAG Debugging
Queries the FAISS documentation knowledge base with the error summary
to retrieve relevant documentation chunks for code repair.
"""

from __future__ import annotations
from rich.console import Console
from state import AgentState

console = Console()


def rag_debug_node(state: AgentState, retriever) -> AgentState:
    """
    Retrieve relevant documentation to help debug the code error.
    Uses the error_summary as the query to the FAISS retriever.
    """
    console.print("\n[bold cyan]📚 Node 6: RAG Debugging[/bold cyan]")

    error_summary = state.get("error_summary", state.get("execution_error", ""))

    # Build a concise retrieval query from the error
    query = _build_query(error_summary)
    console.print(f"   Query: [yellow]{query}[/yellow]")

    # Retrieve top-k docs
    docs = retriever.invoke(query)

    if not docs:
        rag_context = "No relevant documentation found. Check syntax and imports manually."
        console.print("   [yellow]⚠ No docs retrieved, using fallback[/yellow]")
    else:
        rag_context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'docs')}]\n{doc.page_content}"
            for doc in docs
        )
        console.print(
            f"   [green]✔ Retrieved {len(docs)} documentation chunks[/green]"
        )
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "docs")
            preview = doc.page_content[:80].replace("\n", " ")
            console.print(f"   [{i+1}] {src}: [dim]{preview}...[/dim]")

    return {**state, "rag_context": rag_context}


def _build_query(error_summary: str) -> str:
    """Extract a concise query string from the error summary."""
    lines = error_summary.splitlines()
    # Prefer "Exception Type" and "Message" lines
    important = [l for l in lines if l.startswith(("Exception", "Message", "Hint"))]
    if important:
        return " ".join(important[:3])
    return error_summary[:200]
