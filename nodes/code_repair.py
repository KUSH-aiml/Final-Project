"""
Node 7: Code Repair
Uses the LLM + RAG context to fix the failing code.
"""

from __future__ import annotations
from rich.console import Console
from state import AgentState
from prompts import CODE_REPAIR_PROMPT

console = Console()


def code_repair_node(state: AgentState, llm) -> AgentState:
    """Repair failing code using LLM guided by RAG documentation."""
    console.print("\n[bold cyan]🔧 Node 7: Code Repair[/bold cyan]")

    original_code = state.get("generated_code", "")
    error_message = state.get("error_summary", state.get("execution_error", ""))
    rag_context = state.get("rag_context", "")

    chain = CODE_REPAIR_PROMPT | llm

    response = chain.invoke({
        "dataset_path": state["dataset_path"].replace("\\", "/"),
        "original_code": original_code,
        "error_message": error_message,
        "rag_context": rag_context,
    })

    raw_repaired = response.content if hasattr(response, "content") else str(response)
    repaired_code = _strip_code_fences(raw_repaired)

    console.print(
        f"   [green]✔ Repaired code ready ({len(repaired_code.splitlines())} lines)[/green]"
    )

    # Update generated_code so the next repair cycle has the latest version
    return {
        **state,
        "repaired_code": repaired_code,
        "generated_code": repaired_code,
    }


def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
