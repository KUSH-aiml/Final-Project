"""
RAG Pipeline
Builds a FAISS vector store from Python/Pandas documentation, enabling
the agent to retrieve relevant debugging context when code execution fails.
"""

from __future__ import annotations
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ──────────────────────────────────────────────────────────────────────────────
# Documentation Knowledge Base
# Each entry represents curated documentation / debugging knowledge.
# ──────────────────────────────────────────────────────────────────────────────
DOCS = [
    # ── Pandas Core ───────────────────────────────────────────────────────────
    Document(
        page_content="""
Pandas DataFrame: Key Operations and Common Errors

pd.read_csv(filepath): Load a CSV file into a DataFrame. 
  - FileNotFoundError: File path is wrong or file doesn't exist.
  - Use raw strings r"path" or forward slashes on Windows.

df.head(n): Returns first n rows (default 5).
df.tail(n): Returns last n rows.
df.info(): Print summary info including dtypes and non-null counts.
df.describe(): Generate descriptive statistics.
df.shape: Tuple of (rows, columns).
df.columns: Index of column names.
df.dtypes: Series of dtype for each column.

KeyError: Occurs when accessing a column that doesn't exist.
  Fix: Use df.columns.tolist() to inspect column names first.
  Fix: Check for leading/trailing whitespace: df.columns.str.strip()

AttributeError: Occurs when calling a method that doesn't exist on a dtype.
  Fix: Ensure you're calling the right method for the data type.
  Example: df['date'].dt.year only works after pd.to_datetime(df['date'])
""",
        metadata={"source": "pandas-core"},
    ),
    Document(
        page_content="""
Pandas: DateTime Handling

To use datetime-based operations, you must first convert the column:
  df['date'] = pd.to_datetime(df['date'])

Then you can use:
  df['date'].dt.year    → extract year
  df['date'].dt.month   → extract month
  df['date'].dt.day     → extract day
  df['date'].dt.dayofweek → 0=Monday, 6=Sunday

Grouping by time periods:
  df.groupby(df['date'].dt.month)['sales'].sum()
  df.set_index('date').resample('M')['sales'].sum()  # Monthly

Common Error: AttributeError: 'Series' object has no attribute 'dt'
  Fix: The column must be datetime dtype first. Call pd.to_datetime() first.
""",
        metadata={"source": "pandas-datetime"},
    ),
    Document(
        page_content="""
Pandas: Groupby and Aggregation

df.groupby('column')['value'].sum()   → sum by group
df.groupby('column')['value'].mean()  → mean by group
df.groupby(['col1', 'col2']).agg({'val': 'sum', 'other': 'mean'})

Common errors:
  - DataError: No numeric types to aggregate
    Fix: Ensure the column being aggregated is numeric dtype.
    Fix: Convert with pd.to_numeric(df['col'], errors='coerce')

  - KeyError when groupby column doesn't exist:
    Fix: Validate df.columns before groupby.

pivot_table:
  pd.pivot_table(df, values='sales', index='region', columns='category', aggfunc='sum')
  Use fill_value=0 to replace NaN.
""",
        metadata={"source": "pandas-groupby"},
    ),
    Document(
        page_content="""
Pandas: Handling Missing Values

Check missing values:
  df.isnull().sum()         → missing count per column
  df.isnull().sum().sum()   → total missing

Fill missing:
  df['col'].fillna(0)             → fill with constant
  df['col'].fillna(df['col'].mean())  → fill with mean
  df.fillna(method='ffill')       → forward fill (deprecated in newer pandas)
  df.ffill()                      → use this instead (pandas >= 2.0)
  df.bfill()                      → backward fill

Drop rows with missing:
  df.dropna()
  df.dropna(subset=['col'])

FutureWarning about fillna method parameter: 
  Use df.ffill() or df.bfill() instead of fillna(method='ffill').
""",
        metadata={"source": "pandas-missing"},
    ),
    # ── Matplotlib / Seaborn ─────────────────────────────────────────────────
    Document(
        page_content="""
Matplotlib: Saving Figures Without Display

In headless environments (no display/GUI), NEVER use plt.show().
Always use plt.savefig() instead.

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt

Creating and saving a figure:
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x, y)
  ax.set_title('Title')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  plt.tight_layout()
  plt.savefig('output/chart.png', dpi=150, bbox_inches='tight')
  plt.close(fig)     ← ALWAYS close to free memory

Common Error: cannot connect to X server
  Fix: Add matplotlib.use('Agg') at the very top of your script, before any other imports.

Common Error: figure has no axes
  Fix: Ensure you create a figure/axes before plotting.
""",
        metadata={"source": "matplotlib-headless"},
    ),
    Document(
        page_content="""
Seaborn: Creating Statistical Visualizations

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

Histogram:
  fig, ax = plt.subplots()
  sns.histplot(data=df, x='column', ax=ax)

Bar plot:
  sns.barplot(data=df, x='category', y='value', ax=ax)

Box plot:
  sns.boxplot(data=df, x='category', y='value', ax=ax)

Correlation heatmap:
  fig, ax = plt.subplots(figsize=(10, 8))
  sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)

Line plot:
  sns.lineplot(data=df, x='date', y='value', hue='category', ax=ax)

Common Error: TypeError: boxplot() got unexpected keyword
  Fix: Upgrade seaborn or use compatible parameter names.

Always end with:
  plt.tight_layout()
  plt.savefig('output/name.png', dpi=150, bbox_inches='tight')
  plt.close()
""",
        metadata={"source": "seaborn-charts"},
    ),
    # ── Python Debugging ─────────────────────────────────────────────────────
    Document(
        page_content="""
Python: Common Debugging Patterns

AttributeError: 'NoneType' object has no attribute 'X'
  Cause: A function returned None but you tried to call a method on the result.
  Fix: Check that the variable is not None before calling methods.

TypeError: unsupported operand type(s)
  Cause: Mixing incompatible types (e.g., str + int).
  Fix: Explicitly convert types: int(x), float(x), str(x)

ValueError: could not convert string to float
  Cause: A string like '1,234' or '' cannot be directly cast.
  Fix: df['col'] = pd.to_numeric(df['col'].str.replace(',', ''), errors='coerce')

NameError: name 'X' is not defined
  Cause: Variable used before assignment, or import missing.
  Fix: Check all imports at the top. Verify variable scope.

IndexError: list index out of range
  Cause: Accessing index >= len(list)
  Fix: Check length before access: if len(lst) > i: ...

PermissionError writing files:
  Fix: Ensure directory exists: os.makedirs('output', exist_ok=True)
  Fix: Don't use reserved Windows paths.

Ensuring output directory:
  import os
  os.makedirs('output', exist_ok=True)
""",
        metadata={"source": "python-debugging"},
    ),
    Document(
        page_content="""
Python: NumPy Common Patterns

import numpy as np

Array operations:
  arr = np.array([1, 2, 3])
  np.mean(arr), np.std(arr), np.median(arr)
  np.percentile(arr, [25, 75])   → quartiles

Boolean masks:
  mask = arr > threshold
  filtered = arr[mask]

Common Errors:
  ValueError: operands could not be broadcast together
    Cause: Arrays with incompatible shapes.
    Fix: Check arr.shape and align dimensions.

  RuntimeWarning: invalid value encountered in double_scalars
    Cause: NaN or inf values.
    Fix: Use np.nanmean(), np.nanstd() for NaN-safe operations.
    Fix: Filter: arr = arr[~np.isnan(arr)]

Correlation:
  np.corrcoef(x, y)[0, 1]   → Pearson correlation between two arrays
""",
        metadata={"source": "numpy-patterns"},
    ),
    Document(
        page_content="""
Pandas: String Column Operations

df['col'].str.lower()          → to lowercase
df['col'].str.strip()          → remove whitespace
df['col'].str.contains('X')    → boolean mask
df['col'].str.replace('a','b') → replace substring
df['col'].str.split(',')       → split into list

Checking string column unique categories:
  df['category'].unique()
  df['category'].value_counts()

Common Error: AttributeError: Can only use .str accessor with string values
  Fix: Convert column: df['col'] = df['col'].astype(str)
  OR ensure column is object dtype.

Type conversion:
  df['col'] = df['col'].astype('category')
  df['col'] = pd.to_numeric(df['col'], errors='coerce')
  df['col'] = pd.to_datetime(df['col'])
""",
        metadata={"source": "pandas-strings"},
    ),
    Document(
        page_content="""
Matplotlib Backend Issues and Fixes

Error: UserWarning: Matplotlib is currently using agg 
  This is actually fine — agg is the non-interactive backend.

Error: _tkinter.TclError: no display name and no $DISPLAY environment variable
  Fix: Add this at the TOP of the script before any matplotlib import:
    import matplotlib
    matplotlib.use('Agg')

Error: RuntimeError: main thread is not in main loop
  Fix: Move plt.show() to the end, or use matplotlib.use('Agg') and remove plt.show().

Subplot creation:
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
  axes[0, 0].set_title('Plot 1')
  plt.tight_layout()
  plt.savefig('output/subplots.png', dpi=150)
  plt.close()

Setting dark style:
  plt.style.use('seaborn-v0_8-darkgrid')  # use this in newer matplotlib
  # Old: plt.style.use('seaborn-darkgrid')  ← deprecated in matplotlib 3.6+
""",
        metadata={"source": "matplotlib-backend"},
    ),
]


def build_rag_retriever(k: int = 4):
    """
    Build and return a FAISS-based retriever from the documentation knowledge base.

    Args:
        k: Number of documents to retrieve per query.

    Returns:
        LangChain VectorStoreRetriever
    """
    from rich.console import Console
    console = Console()
    console.print("\n[bold cyan]🗄️  Building RAG Knowledge Base...[/bold cyan]")

    # Local sentence-transformers model — no API key needed, runs fully offline
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    # Split documents into chunks
    chunks = splitter.split_documents(DOCS)
    console.print(f"   Indexed [yellow]{len(chunks)}[/yellow] documentation chunks from {len(DOCS)} sources")

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    console.print("   [green]✔ FAISS index ready[/green]")
    return retriever
