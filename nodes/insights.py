"""
Node 8: Insight Generation
Converts execution output and EDA results into human-readable business insights.
"""

from __future__ import annotations
from rich.console import Console
from state import AgentState
from prompts import INSIGHT_PROMPT

console = Console()


def insights_node(state: AgentState, llm) -> AgentState:
    """Generate business insights from execution results."""
    console.print("\n[bold cyan]💡 Node 8: Insight Generation[/bold cyan]")

    execution_output = state.get("execution_result", "")
    eda_summary = state.get("eda_summary", "")

    # Graceful fallback if execution produced no output
    if not execution_output.strip():
        execution_output = (
            "Note: Code executed but stdout was empty. "
            "Insights are based on EDA plan and dataset overview."
        )

    chain = INSIGHT_PROMPT | llm

    response = chain.invoke({
        "eda_summary": eda_summary,
        "execution_output": execution_output[:4000],  # cap to avoid token overflow
    })

    insights = response.content if hasattr(response, "content") else str(response)

    console.print("   [green]✔ Insights generated[/green]")

    return {**state, "insights": insights}
