"""
LangGraph Workflow — Autonomous Data Science Co-Pilot
Defines the StateGraph with all 9 nodes and conditional edges.
"""

from __future__ import annotations
from functools import partial
from langgraph.graph import StateGraph, END
from state import AgentState


def _should_retry(state: AgentState) -> str:
    """
    Conditional router after Sandbox Execution.
    Returns:
      - 'error_detection' if code failed and retries remain
      - 'insights' if code succeeded or max retries exhausted
    """
    error = state.get("execution_error", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if error and retry_count < max_retries:
        return "error_detection"
    return "insights"


def build_graph(llm, retriever):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        llm: Initialized LangChain LLM (e.g., ChatOpenAI)
        retriever: FAISS VectorStoreRetriever from rag_pipeline

    Returns:
        Compiled LangGraph app
    """
    from nodes.ingestion import ingestion_node
    from nodes.eda import eda_node
    from nodes.code_gen import code_gen_node
    from nodes.sandbox import sandbox_node
    from nodes.error_detection import error_detection_node
    from nodes.rag_debug import rag_debug_node
    from nodes.code_repair import code_repair_node
    from nodes.insights import insights_node
    from nodes.report import report_node

    # Use partial to inject llm/retriever into nodes that need them
    graph = StateGraph(AgentState)

    # ── Add Nodes ───────────────────────────────────────────────────────────
    graph.add_node("ingestion",       ingestion_node)
    graph.add_node("eda",             partial(eda_node, llm=llm))
    graph.add_node("code_gen",        partial(code_gen_node, llm=llm))
    graph.add_node("sandbox",         sandbox_node)
    graph.add_node("error_detection", error_detection_node)
    graph.add_node("rag_debug",       partial(rag_debug_node, retriever=retriever))
    graph.add_node("code_repair",     partial(code_repair_node, llm=llm))
    graph.add_node("insight_gen",     partial(insights_node, llm=llm))
    graph.add_node("report_gen",      partial(report_node, llm=llm))

    # ── Define Edges ────────────────────────────────────────────────────────
    graph.set_entry_point("ingestion")

    graph.add_edge("ingestion",       "eda")
    graph.add_edge("eda",             "code_gen")
    graph.add_edge("code_gen",        "sandbox")

    # Conditional: success → insight_gen, failure → error loop
    graph.add_conditional_edges(
        "sandbox",
        _should_retry,
        {
            "error_detection": "error_detection",
            "insights": "insight_gen",
        }
    )

    # Self-correcting loop
    graph.add_edge("error_detection", "rag_debug")
    graph.add_edge("rag_debug",       "code_repair")
    graph.add_edge("code_repair",     "sandbox")   # ← loop back

    # Terminal path
    graph.add_edge("insight_gen",     "report_gen")
    graph.add_edge("report_gen",      END)

    return graph.compile()
