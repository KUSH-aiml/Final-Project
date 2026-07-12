"""
Node 3: Code Generation
Uses the LLM to generate an executable Python analysis script.
"""

from __future__ import annotations
import json
from rich.console import Console
from state import AgentState
from prompts import CODE_GEN_PROMPT

console = Console()


def code_gen_node(state: AgentState, llm) -> AgentState:
    """Generate analysis Python code using LLM."""
    console.print("\n[bold cyan]⚙️  Node 3: Code Generation[/bold cyan]")

    df_info = state["df_info"]
    dataset_path = state["dataset_path"].replace("\\", "/")

    chain = CODE_GEN_PROMPT | llm

    response = chain.invoke({
        "dataset_path": dataset_path,
        "dataset_encoding": state.get("dataset_encoding", "utf-8"),
        "shape": f"{df_info['shape'][0]} rows × {df_info['shape'][1]} columns",
        "dtypes": json.dumps(df_info["dtypes"], indent=2),
        "missing": json.dumps(df_info["missing"], indent=2),
        "sample": df_info["sample"],
        "eda_summary": state.get("eda_summary", ""),
    })

    raw_code = response.content if hasattr(response, "content") else str(response)

    # Strip markdown fences if LLM wrapped code in them
    code = _strip_code_fences(raw_code)

    console.print(f"   [green]✔ Generated {len(code.splitlines())} lines of Python[/green]")

    return {**state, "generated_code": code}


def _strip_code_fences(text: str) -> str:
    """Remove ```python ... ``` fences if present."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
