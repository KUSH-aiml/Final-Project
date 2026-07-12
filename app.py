"""
app.py — FastAPI Web Server for the Autonomous Data Science Co-Pilot
Streams LangGraph agent events to the browser via Server-Sent Events (SSE).

Usage:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations
import os, sys, uuid, queue, threading, json, glob
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ── Output dir for this session lives here ──────────────────────────────────
UPLOAD_DIR  = Path("uploads")
OUTPUT_DIR  = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DS Co-Pilot UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount output/ so browser can GET charts directly
app.mount("/output", StaticFiles(directory="output"), name="output")

# ── In-memory session store ──────────────────────────────────────────────────
sessions: dict[str, "AgentSession"] = {}

class AgentSession:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.finished = False

    def emit(self, event_type: str, **data):
        self.q.put({"type": event_type, **data})

    def done(self):
        self.finished = True
        self.q.put(None)   # sentinel


# ── SSE generator ────────────────────────────────────────────────────────────
def sse_stream(session: AgentSession) -> Generator[str, None, None]:
    while True:
        try:
            event = session.q.get(timeout=120)
        except queue.Empty:
            yield "data: {\"type\":\"heartbeat\"}\n\n"
            continue
        if event is None:
            yield "data: {\"type\":\"done\"}\n\n"
            break
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ── Agent runner (background thread) ────────────────────────────────────────
def run_agent(session_id: str, dataset_path: str, session: AgentSession):
    try:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key.startswith("gsk_..."):
            session.emit("error", message="GROQ_API_KEY is not set. Edit your .env file.")
            return

        session.emit("log", level="info", message="🔧 Initialising Groq LLM…")
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.1,
            max_tokens=3072,
        )

        session.emit("log", level="info", message="🗄️  Building RAG knowledge base…")
        from rag_pipeline import build_rag_retriever
        retriever = build_rag_retriever(k=4)
        session.emit("log", level="success", message="✔ FAISS index ready (10 doc sources)")

        from graph import build_graph
        agent_graph = build_graph(llm, retriever)

        initial_state = {
            "dataset_path": dataset_path,
            "retry_count": 0,
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "execution_error": "",
            "repaired_code": "",
        }

        # ── Node labels ──────────────────────────────────────────────────────
        NODE_LABELS = {
            "ingestion":       ("📥", "Dataset Ingestion",   "Loading & inspecting CSV…"),
            "eda":             ("🔍", "EDA Planning",        "Asking Groq to plan analysis…"),
            "code_gen":        ("⚙️",  "Code Generation",    "Groq writing Python script…"),
            "sandbox":         ("🧪", "Sandbox Execution",   "Running Python in subprocess…"),
            "error_detection": ("🔎", "Error Detection",     "Parsing traceback…"),
            "rag_debug":       ("📚", "RAG Debugging",       "Querying FAISS knowledge base…"),
            "code_repair":     ("🔧", "Code Repair",         "Groq fixing the code…"),
            "insight_gen":     ("💡", "Insight Generation",  "Extracting business insights…"),
            "report_gen":      ("📝", "Report Generation",   "Writing final report…"),
        }

        prev_state: dict = {}

        for step_output in agent_graph.stream(initial_state, stream_mode="updates"):
            node_name = list(step_output.keys())[0]
            node_state: dict = step_output[node_name]

            icon, label, desc = NODE_LABELS.get(node_name, ("▶", node_name, ""))
            session.emit("node_start", node=node_name, label=label, icon=icon, desc=desc)
            session.emit("log", level="info", message=f"{icon} {label}: {desc}")

            # ── Emit rich node-specific events ───────────────────────────────
            if node_name == "ingestion":
                info = node_state.get("df_info", {})
                session.emit("dataset_info",
                    shape=info.get("shape", []),
                    columns=info.get("columns", []),
                    dtypes=info.get("dtypes", {}),
                    missing=info.get("missing", {}),
                    sample=info.get("sample", ""),
                )
                session.emit("log", level="success",
                    message=f"✔ Loaded {info.get('shape', [0,0])[0]} rows × {info.get('shape',[0,0])[1]} columns")

            elif node_name == "eda":
                session.emit("eda_plan", text=node_state.get("eda_summary", ""))
                session.emit("log", level="success", message="✔ EDA plan ready")

            elif node_name == "code_gen":
                code = node_state.get("generated_code", "")
                session.emit("code_generated", code=code, lines=len(code.splitlines()))
                session.emit("log", level="success",
                    message=f"✔ Generated {len(code.splitlines())} lines of Python")

            elif node_name == "sandbox":
                err = node_state.get("execution_error", "")
                result = node_state.get("execution_result", "")
                if err:
                    session.emit("exec_error", error=err)
                    session.emit("log", level="error", message=f"✘ Execution failed")
                else:
                    session.emit("exec_success", output=result)
                    session.emit("log", level="success", message="✔ Execution successful")

            elif node_name == "error_detection":
                session.emit("error_detected",
                    summary=node_state.get("error_summary", ""),
                    retry=node_state.get("retry_count", 0),
                    max_retries=node_state.get("max_retries", 3),
                )
                retry = node_state.get("retry_count", 0)
                max_r  = node_state.get("max_retries", 3)
                session.emit("log", level="warn",
                    message=f"⚠ Error classified (attempt {retry}/{max_r})")

            elif node_name == "rag_debug":
                ctx = node_state.get("rag_context", "")
                session.emit("rag_retrieved", context=ctx)
                chunks = ctx.count("[Source:") if ctx else 0
                session.emit("log", level="purple",
                    message=f"📚 Retrieved {chunks} documentation chunks from FAISS")

            elif node_name == "code_repair":
                repaired = node_state.get("repaired_code", "")
                session.emit("code_repaired", code=repaired, lines=len(repaired.splitlines()))
                session.emit("log", level="success",
                    message=f"✔ Code repaired ({len(repaired.splitlines())} lines)")

            elif node_name == "insight_gen":
                session.emit("insights_ready", text=node_state.get("insights", ""))
                session.emit("log", level="success", message="✔ Insights generated")

            elif node_name == "report_gen":
                session.emit("report_ready", text=node_state.get("report", ""))
                session.emit("log", level="success", message="✔ Report written")

            session.emit("node_done", node=node_name)
            prev_state.update(node_state)

        # ── Collect output files ─────────────────────────────────────────────
        charts = sorted(glob.glob(str(OUTPUT_DIR / "*.png")))
        session.emit("results",
            charts=[os.path.basename(c) for c in charts],
            has_report=os.path.exists(OUTPUT_DIR / "report.md"),
            has_script=os.path.exists(OUTPUT_DIR / "analysis_script.py"),
        )
        session.emit("log", level="success",
            message=f"🎉 Done! {len(charts)} charts generated.")

    except Exception as exc:
        import traceback
        session.emit("log", level="error", message=f"Agent error: {exc}")
        session.emit("fatal_error", error=str(exc), traceback=traceback.format_exc())
    finally:
        session.done()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("ui/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found — run from copilot/ directory</h1>")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    session_id = str(uuid.uuid4())
    save_path  = str(UPLOAD_DIR / f"{session_id}_{file.filename}")
    content    = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    session = AgentSession()
    sessions[session_id] = session

    thread = threading.Thread(
        target=run_agent, args=(session_id, save_path, session), daemon=True
    )
    thread.start()
    return {"session_id": session_id}


@app.get("/stream/{session_id}")
async def stream(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return StreamingResponse(
        sse_stream(session),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/output/report")
async def get_report():
    path = OUTPUT_DIR / "report.md"
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return {"content": path.read_text(encoding="utf-8")}


@app.get("/output/script")
async def get_script():
    path = OUTPUT_DIR / "analysis_script.py"
    if not path.exists():
        raise HTTPException(404, "Script not found")
    return {"content": path.read_text(encoding="utf-8")}
