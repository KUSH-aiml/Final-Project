"""
main.py — Autonomous Data Science Co-Pilot
CLI entry point for running the agent on a CSV dataset.

Usage:
    python main.py --dataset path/to/data.csv
    python main.py --dataset demo_dataset.csv --max-retries 3
    python main.py --dataset demo_dataset.csv --inject-error   (tests self-correcting loop)
"""

from __future__ import annotations
import os
import sys
import argparse
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

console = Console()

# ─── Check API key early ────────────────────────────────────────────────────
def _check_env():
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("gsk_..."):
        console.print(
            "[bold red]ERROR:[/bold red] GROQ_API_KEY is not set.\n"
            "Copy [yellow].env.example[/yellow] → [yellow].env[/yellow] and add your key.\n"
            "Get a free key at: [link]https://console.groq.com/keys[/link]"
        )
        sys.exit(1)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Science Co-Pilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the input CSV dataset file",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("MAX_RETRIES", "3")),
        help="Maximum self-correction attempts (default: 3)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        help="Groq model to use (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--inject-error",
        action="store_true",
        help="Inject a deliberate bug into generated code to test the repair loop",
    )
    return parser.parse_args()


def _inject_bug(code: str) -> str:
    """
    Deliberately corrupt the generated code to force the self-correcting loop.
    Replaces 'pd.read_csv' with a misspelled 'pd.read_cvs' to trigger an AttributeError.
    """
    corrupted = code.replace("pd.read_csv", "pd.read_cvs", 1)
    if corrupted == code:
        # fallback: introduce an undefined variable reference
        corrupted = "undefined_variable_xyz.do_something()\n" + code
    return corrupted


def main():
    _check_env()
    args = _parse_args()

    console.print(
        Panel.fit(
            "[bold white]⚡ Autonomous Data Science Co-Pilot[/bold white]\n"
            "[dim]Powered by LangChain + LangGraph + Groq[/dim]",
            border_style="bright_cyan",
        )
    )
    console.print(f"   Dataset  : [yellow]{args.dataset}[/yellow]")
    console.print(f"   Model    : [yellow]{args.model}[/yellow] (Groq)")
    console.print(f"   Max Retries: [yellow]{args.max_retries}[/yellow]")
    if args.inject_error:
        console.print(
            "   [bold yellow]⚡ ERROR INJECTION MODE[/bold yellow]: "
            "A deliberate bug will be injected to test the repair loop."
        )

    # ── Validate dataset path ────────────────────────────────────────────────
    dataset_path = os.path.abspath(args.dataset)
    if not os.path.exists(dataset_path):
        console.print(f"[bold red]ERROR:[/bold red] Dataset not found: {dataset_path}")
        sys.exit(1)

    # ── Init LLM ─────────────────────────────────────────────────────────────
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=args.model,
        temperature=0.1,
        max_tokens=3072,
    )

    # ── Build RAG retriever ──────────────────────────────────────────────────
    from rag_pipeline import build_rag_retriever
    retriever = build_rag_retriever(k=4)

    # ── Build LangGraph ──────────────────────────────────────────────────────
    from graph import build_graph
    app = build_graph(llm, retriever)

    console.print(Rule("[bold cyan]Starting Agent Workflow[/bold cyan]"))

    # ── Initial state ────────────────────────────────────────────────────────
    initial_state = {
        "dataset_path": dataset_path,
        "retry_count": 0,
        "max_retries": args.max_retries,
        "execution_error": "",
        "repaired_code": "",
    }

    start_time = time.time()

    try:
        # ── Run the graph ─────────────────────────────────────────────────────
        for step_output in app.stream(initial_state, stream_mode="updates"):
            node_name = list(step_output.keys())[0]
            # If inject-error is active, corrupt the code after code_gen node
            if args.inject_error and node_name == "code_gen":
                step_state = step_output[node_name]
                if "generated_code" in step_state:
                    step_state["generated_code"] = _inject_bug(
                        step_state["generated_code"]
                    )
                    console.print(
                        "   [bold red]🐛 BUG INJECTED into generated code[/bold red]"
                    )

        elapsed = time.time() - start_time
        console.print(Rule("[bold green]Agent Completed[/bold green]"))
        console.print(f"\n   ⏱  Total time: [cyan]{elapsed:.1f}s[/cyan]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Agent interrupted by user.[/yellow]")
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]Agent failed with error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Print summary ────────────────────────────────────────────────────────
    output_dir = os.path.join(os.getcwd(), "output")
    report_path = os.path.join(output_dir, "report.md")

    console.print(
        Panel(
            f"[bold green]✅ Analysis Complete![/bold green]\n\n"
            f"📄 Report : [link={report_path}]{report_path}[/link]\n"
            f"🖼️  Charts : {output_dir}/*.png\n"
            f"🐍 Script : {output_dir}/analysis_script.py",
            border_style="green",
            title="Output",
        )
    )


if __name__ == "__main__":
    main()
