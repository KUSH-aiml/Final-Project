"""
AgentState — shared state schema for the LangGraph workflow.
All nodes read from and write to this TypedDict.
"""

from __future__ import annotations
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────
    dataset_path: str
    dataset_encoding: str                  # Absolute path to the CSV file

    # ── Node 1: Ingestion ──────────────────────────────────────────────
    df_info: dict                      # Shape, dtypes, nulls, sample rows

    # ── Node 2: EDA ───────────────────────────────────────────────────
    eda_summary: str                   # Textual EDA plan from LLM

    # ── Node 3: Code Generation ───────────────────────────────────────
    generated_code: str                # Python script to execute

    # ── Node 4: Sandbox Execution ─────────────────────────────────────
    execution_result: str              # Captured stdout on success
    execution_error: str               # Captured stderr / traceback on failure

    # ── Node 5: Error Detection ───────────────────────────────────────
    error_summary: str                 # Structured error description

    # ── Node 6: RAG Debugging ─────────────────────────────────────────
    rag_context: str                   # Retrieved documentation chunks

    # ── Node 7: Code Repair ───────────────────────────────────────────
    repaired_code: str                 # Fixed Python script

    # ── Node 8: Insight Generation ────────────────────────────────────
    insights: str                      # Bullet-point findings

    # ── Node 9: Report Generation ─────────────────────────────────────
    report: str                        # Full markdown report text

    # ── Loop control ──────────────────────────────────────────────────
    retry_count: int                   # Number of self-correction attempts
    max_retries: int                   # Ceiling for the correction loop
