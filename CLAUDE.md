# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Autonomous Data Science Co-Pilot: a LangGraph agent that takes a raw CSV, plans an EDA, generates a pandas/seaborn analysis script, executes it in a subprocess sandbox, self-heals runtime errors via a local FAISS RAG lookup against curated Python/Pandas docs, and emits a markdown report + charts. Runs either as a CLI (`main.py`) or a FastAPI web app with SSE streaming (`app.py`).

## Commands

```bash
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env                 # then set GROQ_API_KEY

# Run web UI (SSE dashboard at http://localhost:8000)
uvicorn app:app --reload

# Run CLI
python main.py --dataset demo_dataset.csv
python main.py --dataset demo_dataset.csv --model llama-3.3-70b-versatile
python main.py --dataset demo_dataset.csv --max-retries 5
python main.py --dataset demo_dataset.csv --inject-error   # forces the self-heal loop for testing
```

There is no test suite, linter, or build step configured in this repo — `--inject-error` is the primary way to exercise/verify the self-correction path end to end.

Required env vars (`.env`, see `.env.example`): `GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.1-8b-instant`, optimized for speed; swap to `llama-3.3-70b-versatile` via `--model` or `.env` if code-gen quality/retry rate matters more than latency), `MAX_RETRIES` (default 3).

## Architecture

The whole system is a single `langgraph` `StateGraph` (`graph.py`) over one shared `AgentState` TypedDict (`state.py`), threaded through 9 node functions in `nodes/`. Both entry points (`main.py` CLI, `app.py` FastAPI) build the same graph via `build_graph(llm, retriever)` and drive it with `app.stream(initial_state, stream_mode="updates")` — they only differ in how they surface the per-node output (Rich console vs SSE events to the browser).

Node pipeline (linear, one node per file in `nodes/`):
1. `ingestion.py` — reads the CSV, computes shape/dtypes/nulls/sample → `df_info`
2. `eda.py` — LLM produces an analysis plan → `eda_summary`
3. `code_gen.py` — LLM turns the plan into a pandas/seaborn script → `generated_code`
4. `sandbox.py` — runs the script (`repaired_code` if present, else `generated_code`) as a subprocess with a 90s timeout, capturing stdout/stderr

After `sandbox`, `graph.py`'s `_should_retry` conditional edge branches:
- execution succeeded, or `retry_count >= max_retries` → skip to `insight_gen`
- execution failed and retries remain → `error_detection` → `rag_debug` → `code_repair` → back to `sandbox` (loop)

5. `error_detection.py` — classifies/summarizes the traceback → `error_summary`
6. `rag_debug.py` — queries the FAISS retriever (built in `rag_pipeline.py`) for relevant doc chunks → `rag_context`
7. `code_repair.py` — LLM rewrites the script using the error + retrieved docs → `repaired_code` (sandbox re-runs this next)
8. `insights.py` — LLM extracts business-facing bullet insights from execution output → `insights`
9. `report.py` — compiles insights + chart references into the final markdown → `report`

Key state fields to know when touching a node: `retry_count`/`max_retries` (loop control), `execution_error` (drives the retry branch), and the fact that `sandbox_node` always prefers `repaired_code` over `generated_code` and clears `repaired_code` on success.

`rag_pipeline.py` builds the FAISS index in-memory on every run (no persistence) from a hardcoded `DOCS` list of ~10 curated documentation chunks (pandas core/datetime/groupby/missing-values, matplotlib headless/backend, seaborn charts, numpy patterns, python debugging, string ops) using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`, CPU) and `RecursiveCharacterTextSplitter`. To extend the agent's self-healing knowledge, add `Document` entries here rather than editing individual nodes.

`prompts.py` holds all LLM system/human prompt templates used by the EDA, code-gen, code-repair, insight, and report nodes — check here first when adjusting LLM behavior rather than editing prompts inline in `nodes/`.

`app.py` specifics: uploaded CSVs go to `uploads/`, generated artifacts to `output/` (served statically at `/output`). Each `/analyze` call spawns a background thread running the graph and pushes events through an in-process `queue.Queue` per session (`sessions` dict, keyed by UUID), consumed by `/stream/{session_id}` as SSE. `NODE_LABELS` in `app.py` maps node names to the icon/label/description shown in `ui/index.html`; update it if node names change in `graph.py`.

Sandbox execution is process-level isolation only (`subprocess.run` with a timeout) — not a hardened sandbox. Treat `nodes/sandbox.py` as executing untrusted LLM-generated code with the privileges of the running user.

## Output

Generated artifacts land in `output/`: `report.md`, `analysis_script.py` (or `analysis_script_attempt{N}.py` per repair attempt), and `*.png` charts. These are gitignored except for `.gitkeep`.
