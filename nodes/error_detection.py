"""
Node 5: Error Detection
Parses the execution error and constructs a structured error summary.
Increments the retry counter.
"""

from __future__ import annotations
import re
from rich.console import Console
from state import AgentState

console = Console()

# Map of common Python exception names to human descriptions
EXCEPTION_HINTS = {
    "AttributeError": "An object does not have the expected attribute or method.",
    "KeyError": "A dictionary key or DataFrame column does not exist.",
    "TypeError": "An operation was applied to an object of the wrong type.",
    "ValueError": "A function received an argument with an invalid value.",
    "ImportError": "A required module could not be imported.",
    "FileNotFoundError": "A file or path does not exist.",
    "IndexError": "A list or array index is out of range.",
    "NameError": "A variable or function name is not defined.",
    "ZeroDivisionError": "Division by zero occurred.",
    "PermissionError": "File permission denied.",
}


def error_detection_node(state: AgentState) -> AgentState:
    """Parse error traceback and build a structured error summary."""
    console.print("\n[bold cyan]🔎 Node 5: Error Detection[/bold cyan]")

    error_text = state.get("execution_error", "")
    retry_count = state.get("retry_count", 0) + 1
    max_retries = state.get("max_retries", 3)

    # Extract exception type
    exc_type = "Unknown"
    exc_match = re.search(r"(\w+Error|\w+Exception|SyntaxError):\s*(.*)", error_text)
    if exc_match:
        exc_type = exc_match.group(1)
        exc_msg = exc_match.group(2).strip()
    else:
        exc_msg = error_text.split("\n")[-1].strip()

    # Extract line number
    line_match = re.search(r'File ".*?", line (\d+)', error_text)
    line_no = int(line_match.group(1)) if line_match else "unknown"

    # Build hint
    hint = EXCEPTION_HINTS.get(exc_type, "Check the traceback for the root cause.")

    error_summary = (
        f"Exception Type: {exc_type}\n"
        f"Message: {exc_msg}\n"
        f"Line Number: {line_no}\n"
        f"Hint: {hint}\n\n"
        f"Full Traceback:\n{error_text}"
    )

    console.print(
        f"   [red]Error:[/red] {exc_type} at line {line_no} "
        f"(attempt {retry_count}/{max_retries})"
    )

    return {**state, "error_summary": error_summary, "retry_count": retry_count}
