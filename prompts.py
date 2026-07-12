"""
Prompt templates for every LangGraph node.
All templates use ChatPromptTemplate for structured LLM calls.
"""

from langchain_core.prompts import ChatPromptTemplate

# ─── Node 2: EDA ─────────────────────────────────────────────────────────────
EDA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior data scientist. Your job is to plan an Exploratory Data Analysis 
for a dataset described below. Be specific, actionable, and comprehensive.

Respond with a structured EDA plan that covers:
1. Key statistical patterns to investigate (means, medians, std, skewness)
2. Columns most likely to contain interesting correlations
3. Potential anomalies or outliers to look for
4. Suggested visualizations (bar charts, histograms, correlation heatmaps, line plots, etc.)
5. Any data quality concerns (missing values, type mismatches)

Be concise but thorough. Focus on insights that would be valuable for business decisions."""
    ),
    (
        "human",
        """Dataset Info:
- Shape: {shape}
- Columns and dtypes: {dtypes}
- Missing values per column: {missing}
- Sample rows:
{sample}

Please write a detailed EDA plan for this dataset."""
    )
])

# ─── Node 3: Code Generation ─────────────────────────────────────────────────────────────
_CODE_GEN_SYSTEM = """\
You are an expert Python data scientist. Your task is to write a COMPLETE, RUNNABLE Python script
for exploratory data analysis. You must produce ALL 5 charts listed below.

ABSOLUTE RULES (violation = failure):
- Line 1 MUST be: import matplotlib
- Line 2 MUST be: matplotlib.use('Agg')
- NEVER call plt.show() anywhere
- After every savefig() call plt.close()
- Always pass ax= to seaborn functions
- Return ONLY the Python code, no markdown fences, no explanation

REQUIRED SCRIPT STRUCTURE (fill in every <...> section):

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)
sns.set_theme(style='whitegrid')

df = pd.read_csv('{dataset_path}', encoding='{dataset_encoding}')
df.columns = df.columns.str.strip()

print('Dataset shape:', df.shape)
print('Columns:', df.columns.tolist())
print(df.describe())

# CHART 1 — Top categories by a numeric column
fig, ax = plt.subplots(figsize=(12, 6))
<groupby or value_counts bar chart using ax>
ax.set_title('...')
ax.set_xlabel('...')
ax.set_ylabel('...')
plt.tight_layout()
plt.savefig('output/chart_1_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved chart_1_categories.png')

# CHART 2 — Trend over time or ordered sequence
fig, ax = plt.subplots(figsize=(12, 5))
<line or bar plot using ax>
ax.set_title('...')
plt.tight_layout()
plt.savefig('output/chart_2_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved chart_2_trend.png')

# CHART 3 — Distribution of a numeric column
fig, ax = plt.subplots(figsize=(10, 5))
<sns.histplot or sns.kdeplot using ax>
ax.set_title('...')
plt.tight_layout()
plt.savefig('output/chart_3_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved chart_3_distribution.png')

# CHART 4 — Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
numeric_df = df.select_dtypes(include='number')
if numeric_df.shape[1] > 1:
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('output/chart_4_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved chart_4_correlation.png')

# CHART 5 — Breakdown by a second categorical dimension
fig, ax = plt.subplots(figsize=(12, 6))
<grouped bar or box plot using ax>
ax.set_title('...')
plt.tight_layout()
plt.savefig('output/chart_5_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved chart_5_breakdown.png')

print('Analysis complete. All charts saved to output/')
"""

CODE_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CODE_GEN_SYSTEM),
    (
        "human",
        """Dataset path: {dataset_path}
Detected Encoding: {dataset_encoding}
Shape: {shape}
Columns and dtypes:
{dtypes}
Missing values: {missing}
Sample rows:
{sample}

EDA Plan:
{eda_summary}

Now write the COMPLETE filled-in Python script. Replace EVERY <...> placeholder with real code.
All 5 charts MUST be present and saved. Return only Python code."""
    )
])

# ─── Node 7: Code Repair ─────────────────────────────────────────────────────
CODE_REPAIR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Python debugger. You will receive:
1. A Python script that failed to execute
2. The error/traceback that occurred
3. Relevant documentation snippets to help fix the issue

Your task: Fix ALL bugs in the script so it runs without errors.

RULES:
- Keep all the original analysis logic intact — only fix the bugs
- Apply fixes suggested by the documentation context if relevant
- Return ONLY the corrected Python code — no markdown, no explanation
- Ensure all file paths use forward slashes or os.path.join()
- The dataset path MUST be exactly: {dataset_path}
- NEVER change the dataset file path to anything else."""
    ),
    (
        "human",
        """Dataset Path: {dataset_path}
ORIGINAL CODE:
{original_code}

ERROR MESSAGE:
{error_message}

RELEVANT DOCUMENTATION:
{rag_context}

Write the corrected Python script. Return ONLY the code."""
    )
])

# ─── Node 8: Insight Generation ──────────────────────────────────────────────
INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a business intelligence analyst. Convert raw data analysis outputs into 
clear, human-readable business insights.

Format your response as:
## Key Findings
- [Insight 1]
- [Insight 2]
...

## Trends Detected
- [Trend 1]
...

## Anomalies & Outliers
- [Anomaly 1 or 'None detected']
...

## Recommendations
- [Recommendation 1]
...

Be specific with numbers, percentages, and comparisons where visible in the output."""
    ),
    (
        "human",
        """Dataset Summary:
{eda_summary}

Execution Output (analysis results / print statements):
{execution_output}

Generate the business insights report."""
    )
])

# ─── Node 9: Report Generation ───────────────────────────────────────────────
REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a technical writer creating a professional data analysis report in Markdown.

Structure the report EXACTLY as:
# Data Analysis Report

## 1. Dataset Overview
[Shape, columns, data types, time range if applicable, data quality notes]

## 2. Exploratory Data Analysis Summary
[What was analyzed, methods used]

## 3. Key Insights
[The most important findings with specific numbers]

## 4. Trends & Patterns
[Time-based or categorical trends]

## 5. Anomalies & Data Quality
[Any outliers, missing data, or quality issues found]

## 6. Visualizations Generated
[List each chart with a brief description of what it shows]

## 7. Conclusions & Recommendations
[Actionable business recommendations based on findings]

---
*Report generated by Autonomous Data Science Co-Pilot*"""
    ),
    (
        "human",
        """Dataset Path: {dataset_path}
Shape: {shape}
Columns: {dtypes}

EDA Plan:
{eda_summary}

Key Insights:
{insights}

Charts Generated (in output/ directory):
{charts}

Write the complete analysis report in Markdown."""
    )
])
