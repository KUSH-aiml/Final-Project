"""
Node 2: Exploratory Data Analysis (EDA)
Calls the LLM with dataset metadata to generate a structured EDA plan.
"""

from __future__ import annotations
import json
from rich.console import Console
from state import AgentState
from prompts import EDA_PROMPT

console = Console()


def eda_node(state: AgentState, llm) -> AgentState:
    """Generate an EDA plan using the LLM based on dataset metadata."""
    console.print("\n[bold cyan]🔍 Node 2: Exploratory Data Analysis[/bold cyan]")

    df_info = state["df_info"]
    chain = EDA_PROMPT | llm

    response = chain.invoke({
        "shape": f"{df_info['shape'][0]} rows × {df_info['shape'][1]} columns",
        "dtypes": json.dumps(df_info["dtypes"], indent=2),
        "missing": json.dumps(df_info["missing"], indent=2),
        "sample": df_info["sample"],
    })

    eda_summary = response.content if hasattr(response, "content") else str(response)

    console.print("   [green]✔ EDA plan generated[/green]")
    console.print(f"   [dim]{eda_summary[:200]}...[/dim]")

    return {**state, "eda_summary": eda_summary}
