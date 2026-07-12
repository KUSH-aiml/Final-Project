"""
Node 1: Dataset Ingestion
Loads the CSV, extracts structural metadata, and populates `df_info` in state.
"""

from __future__ import annotations
import pandas as pd
from rich.console import Console
from state import AgentState

console = Console()


def ingestion_node(state: AgentState) -> AgentState:
    """Load the dataset and extract structural metadata."""
    dataset_path = state["dataset_path"]
    console.print(f"\n[bold cyan]📂 Node 1: Dataset Ingestion[/bold cyan]")
    console.print(f"   Loading: [yellow]{dataset_path}[/yellow]")

    try:
        # Try UTF-8 first
        df = pd.read_csv(dataset_path)
        encoding_used = 'utf-8'
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1 / ISO-8859-1
            df = pd.read_csv(dataset_path, encoding='latin-1')
            encoding_used = 'latin-1'
        except UnicodeDecodeError:
            # Fallback to Windows-1252 (often from Excel)
            df = pd.read_csv(dataset_path, encoding='cp1252')
            encoding_used = 'cp1252'
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}") from e

    # Gather metadata
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    null_counts = df.isnull().sum()
    missing = {k: int(v) for k, v in null_counts.to_dict().items()}
    sample = df.head(5).to_string(index=False)

    # Numeric summary — single vectorized agg() across all numeric columns
    # instead of 5 separate reductions per column.
    numeric_summary = {}
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].agg(["min", "max", "mean", "std", "median"])
        for col in numeric_cols:
            numeric_summary[col] = {
                "min": float(stats.at["min", col]),
                "max": float(stats.at["max", col]),
                "mean": round(float(stats.at["mean", col]), 4),
                "std": round(float(stats.at["std", col]), 4),
                "median": float(stats.at["median", col]),
            }

    # Categorical summaries — reuse one value_counts() per column for both
    # the unique count and the top values instead of scanning twice.
    categorical_summary = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts()
        categorical_summary[col] = {
            "unique_count": int(len(vc)),
            "top_values": vc.head(5).to_dict(),
        }

    df_info = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing": missing,
        "sample": sample,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "total_missing": int(null_counts.sum()),
    }

    console.print(
        f"   [green]✔ Loaded {df.shape[0]} rows × {df.shape[1]} columns[/green]"
    )
    console.print(
        f"   Missing values: {df_info['total_missing']} total"
    )

    return {
        **state,
        "df_info": df_info,
        "dataset_encoding": encoding_used
    }
