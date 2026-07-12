"""
Node 9: Report Generation
Produces a structured Markdown analysis report and saves it to output/report.md.
"""

from __future__ import annotations
import os
import glob
from rich.console import Console
from state import AgentState
from prompts import REPORT_PROMPT

console = Console()
OUTPUT_DIR = "output"


def report_node(state: AgentState, llm) -> AgentState:
    """Generate and save the final analysis report."""
    console.print("\n[bold cyan]📝 Node 9: Report Generation[/bold cyan]")

    df_info = state.get("df_info", {})
    dataset_path = state.get("dataset_path", "")

    # List generated charts
    chart_paths = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
    chart_list = "\n".join(
        f"- {os.path.basename(p)}" for p in sorted(chart_paths)
    ) or "- No charts generated"

    import json
    chain = REPORT_PROMPT | llm

    response = chain.invoke({
        "dataset_path": dataset_path,
        "shape": f"{df_info.get('shape', ['?','?'])[0]} rows × {df_info.get('shape', ['?','?'])[1]} columns",
        "dtypes": json.dumps(df_info.get("dtypes", {}), indent=2),
        "eda_summary": state.get("eda_summary", ""),
        "insights": state.get("insights", ""),
        "charts": chart_list,
    })

    report = response.content if hasattr(response, "content") else str(response)

    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    console.print(f"   [green]✔ Report saved → {report_path}[/green]")
    console.print(f"   Charts referenced: {len(chart_paths)}")

    return {**state, "report": report}
