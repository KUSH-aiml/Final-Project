# ⚡ Autonomous Data Science Co-Pilot

An advanced, production-grade autonomous AI agent that ingestion-checks, plans, generates, executes, and self-heals Python code to analyze raw CSV datasets with minimal human supervision. Powered by **LangChain**, **LangGraph**, and **Groq's Llama 3.1 8B Instant** model, it runs visualization scripts in a secure sandbox, automatically resolves errors via a local **RAG system**, and outputs structured business reports.

---

## 🚀 Key Features

*   **State-of-the-Art Model Integration**: Engineered to leverage the raw speed of Groq's LPU inference running `llama-3.1-8b-instant` for rapid agentic iteration. Swap `GROQ_MODEL` to a larger model (e.g. `llama-3.3-70b-versatile`) if you need stronger code-gen quality over raw speed.
*   **Self-Healing Code Execution**: Automatically detects run-time errors (KeyError, AttributeError, ValueError, etc.), queries a local **FAISS Vector Store** of Python/Pandas documentation, and repairs the script.
*   **FastAPI & SSE Streaming**: Real-time event streaming via Server-Sent Events (SSE) updates the clean web dashboard as the agent steps through the pipeline.
*   **Fully-Featured CLI & Web UI**: Flexibility to run immediate analyses via command-line arguments or using the responsive browser dashboard.
*   **Headless Visualization Pipeline**: Uses a dedicated non-interactive Matplotlib backend (`Agg`) to safely generate crisp Seaborn and Matplotlib charts.

---

## 🛠️ Tech Stack & Environment

*   **Primary Language**: **Python 3.13.9** (fully tested & compatible with Python 3.13+)
*   **LLM Orchestration**: LangChain, LangGraph
*   **AI Models**: Groq Llama 3.1 8B Instant (`llama-3.1-8b-instant`)
*   **Vector Database & Embeddings**: FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Data & Plotting**: Pandas, NumPy, Matplotlib, Seaborn, Scipy
*   **Backend & Server**: FastAPI, Uvicorn (standard), Watchfiles

---

## 📈 System Architecture

```
                                      Dataset (CSV)
                                           │
                                           ▼
                               ┌──────────────────────┐
                               │ 1. Dataset Ingestion │
                               └──────────────────────┘
                                           │
                                           ▼
                               ┌──────────────────────┐
                               │   2. EDA Planning    │
                               └──────────────────────┘
                                           │
                                           ▼
                               ┌──────────────────────┐
                               │  3. Code Generation  │
                               └──────────────────────┘
                                           │
                                           ▼
                               ┌──────────────────────┐◀──────────────────────┐
                               │ 4. Sandbox Execution │                      │
                               └──────────────────────┘                      │
                                           │                                 │
                            ┌──────────────┴──────────────┐                  │
                            │                             │                  │
                        [Success]                      [Error]               │
                            │                             │                  │
                            ▼                             ▼                  │
                ┌──────────────────────┐      ┌──────────────────────┐       │
                │ 8. Insight Extraction│      │  5. Error Detection  │       │
                └──────────────────────┘      └──────────────────────┘       │
                            │                             │                  │
                            ▼                             ▼                  │
                ┌──────────────────────┐      ┌──────────────────────┐       │
                │ 9. Report Generation │      │     6. RAG Debug     │       │
                └──────────────────────┘      └──────────────────────┘       │
                                                          │                  │
                                                          ▼                  │
                                              ┌──────────────────────┐       │
                                              │   7. Code Repair     │───────┘
                                              └──────────────────────┘
```

---

## ⚙️ Setup & Installation

Follow these steps to set up the environment using `uv` (recommended for ultra-fast installs) or standard `pip`:

### 1. Create a Virtual Environment
```bash
# Using uv (fastest)
uv venv

# OR using standard python
python -m venv .venv
```

### 2. Activate the Virtual Environment
*   **Windows (PowerShell)**:
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
*   **Windows (CMD)**:
    ```cmd
    .\.venv\Scripts\activate.bat
    ```
*   **Linux / macOS**:
    ```bash
    source .venv/bin/activate
    ```

### 3. Install Dependencies
```bash
# Using uv
uv pip install -r requirements.txt

# OR using standard pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
MAX_RETRIES=3
```
Get a free Groq API key at **[console.groq.com/keys](https://console.groq.com/keys)**.

---

## 🖥️ Running the Application

### Option A: Real-Time Web Dashboard (Recommended)
Launch the FastAPI web server with hot-reloading:
```bash
uvicorn app:app --reload
```
Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**.
Upload any CSV dataset, click **Analyze**, and watch the live agent execution steps, code generation, self-healing traceback logs, and final chart renders update in real time.

### Option B: Command Line Interface (CLI)
Run a direct, synchronous analysis on a local dataset:
```bash
# Basic run with demo retail dataset
python main.py --dataset demo_dataset.csv

# Run with a custom model override
python main.py --dataset demo_dataset.csv --model llama-3.3-70b-versatile   # slower, higher-quality code-gen

# Adjust maximum self-correction attempts
python main.py --dataset demo_dataset.csv --max-retries 5

# Deliberately inject a bug to test the RAG self-healing loop
python main.py --dataset demo_dataset.csv --inject-error
```

---

## 📂 Output Deliverables

All generated outputs are saved dynamically under the `output/` directory:

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Markdown Report** | `output/report.md` | Full structured business analysis, summary statistics, and executive findings. |
| **Analysis Script** | `output/analysis_script.py` | The complete, executable, and sanitized Python script generated by the agent. |
| **Visual Charts** | `output/*.png` | Headless-rendered high-resolution visual assets (heatmaps, bar plots, distributions). |

---

## 📦 Project Structure

```
copilot/
├── main.py                  # CLI entry point
├── app.py                   # FastAPI SSE web server
├── graph.py                 # LangGraph workflow definition & state machine
├── state.py                 # AgentState TypedDict definitions
├── prompts.py               # Optimized LLM system & human prompt templates
├── rag_pipeline.py          # FAISS offline vector database setup & retrieval
├── requirements.txt         # Core dependencies
├── .env.example             # Template env file
├── demo_dataset.csv         # Retail sales sample dataset
├── output/                  # Directory for generated reports & charts
├── ui/
│   └── index.html           # Interactive stream web dashboard
└── nodes/
    ├── ingestion.py         # Node 1: CSV verification & summary statistics
    ├── eda.py               # Node 2: Iterative EDA design plan
    ├── code_gen.py          # Node 3: Structured pandas & seaborn script generation
    ├── sandbox.py           # Node 4: Secure subprocess execution with timeout
    ├── error_detection.py   # Node 5: Traceback classification & severity scoring
    ├── rag_debug.py         # Node 6: Context retrieval from local documentation
    ├── code_repair.py       # Node 7: Multi-turn self-healing code repair
    ├── insights.py          # Node 8: Executive insight extraction
    └── report.py            # Node 9: Elegant markdown compiler
```

---

## 🧠 LangGraph Node Breakdown

| Node Name | Operational Purpose |
| :--- | :--- |
| **Ingestion** | Reads the dataset, validates columns, maps shapes, checks missing values, and samples head rows. |
| **EDA** | Creates an analytical plan with targeted questions based on data types and distribution styles. |
| **Code Gen** | Converts the plan into python-executable visualization code with safety margins. |
| **Sandbox** | Spawns a virtual execution context using sandboxed subprocess execution. |
| **Error Detection**| Catches and parses any stdout tracebacks, identifying key error classifications. |
| **RAG Debugging** | Automatically runs vector search matches against 10 curated local docs to fetch targeted fixes. |
| **Code Repair** | Rewrites the faulty script segments using standard best practices and retrieved documentation context. |
| **Insights** | Analyzes the statistical outputs to generate core business bullet points. |
| **Report** | Unifies insights, plots, and summaries into a presentation-ready markdown document. |

---

## 🩹 Local RAG Knowledge Base

The FAISS vector index is populated with 10 production-quality documentation chunks covering common analytical pitfalls:
*   Pandas core structures & handling `KeyError` / `AttributeError`.
*   Temporal processing using `.dt` accessor constraints.
*   Data type validation prior to `.groupby` aggregations.
*   Safe null-value imputation (e.g., using `.ffill()` / `.bfill()`).
*   Non-interactive Matplotlib backend optimization (`matplotlib.use('Agg')`).
*   Modern Seaborn visualization patterns.
*   Numeric column parsing (e.g., stripping commas and symbols).
*   Multidimensional array alignment in NumPy.
