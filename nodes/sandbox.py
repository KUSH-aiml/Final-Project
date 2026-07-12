"""
Node 4: Sandbox Execution
Runs the generated Python script in a subprocess (sandboxed).
Captures stdout, stderr, return code.
"""

from __future__ import annotations
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from rich.console import Console
from state import AgentState

console = Console()
OUTPUT_DIR = "output"


def sandbox_node(state: AgentState) -> AgentState:
    """Execute the generated Python code in a subprocess sandbox."""
    console.print("\n[bold cyan]🚀 Node 4: Sandbox Execution[/bold cyan]")

    # Decide which code to run (repaired takes priority)
    code = state.get("repaired_code") or state.get("generated_code", "")

    if not code.strip():
        return {**state, "execution_error": "No code to execute."}

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write code to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            console.print("   [green]✔ Execution successful[/green]")
            if result.stdout:
                console.print(f"   [dim]{result.stdout[:300]}[/dim]")
            # Also save the generated code to output/
            _save_script(code, state.get("retry_count", 0))
            return {
                **state,
                "execution_result": result.stdout,
                "execution_error": "",
                "repaired_code": "",  # Clear repaired code after success
            }
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            console.print(f"   [red]✘ Execution failed (exit {result.returncode})[/red]")
            console.print(f"   [dim red]{error_msg[:400]}[/dim red]")
            return {**state, "execution_error": error_msg, "execution_result": ""}

    except subprocess.TimeoutExpired:
        err = "Execution timed out after 90 seconds."
        console.print(f"   [red]✘ {err}[/red]")
        return {**state, "execution_error": err, "execution_result": ""}

    except Exception as e:
        err = f"Subprocess error: {e}"
        console.print(f"   [red]✘ {err}[/red]")
        return {**state, "execution_error": err, "execution_result": ""}

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _save_script(code: str, retry: int) -> None:
    """Save the executed script for reference."""
    suffix = f"_attempt{retry}" if retry > 0 else ""
    path = os.path.join(OUTPUT_DIR, f"analysis_script{suffix}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
