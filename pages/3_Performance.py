"""Performance — Aggregated metrics visualization across models and test sets."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from eval.results import load_all_results

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"

st.set_page_config(page_title="Performance", page_icon="📊", layout="wide")
st.title("📊 Performance Dashboard")

# ── Load all results ────────────────────────────────────────────────────────

all_results = load_all_results(RESULTS_DIR)

if all_results.empty:
    st.info(
        "No evaluation results saved yet. "
        "Go to **Run Evaluation** to run an evaluation and save the results."
    )
    st.stop()

# Parse evaluated_at as datetime for sorting
all_results["evaluated_at_dt"] = pd.to_datetime(all_results["evaluated_at"], errors="coerce")
all_results = all_results.sort_values("evaluated_at_dt")

# ── Filters ─────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")

available_testsets = sorted(all_results["testset_name"].unique())
selected_testsets = st.sidebar.multiselect(
    "Test sets", available_testsets, default=available_testsets
)

available_models = sorted(all_results["model_name"].unique())
selected_models = st.sidebar.multiselect(
    "Models", available_models, default=available_models
)

# Date range filter
if all_results["evaluated_at_dt"].notna().any():
    min_date = all_results["evaluated_at_dt"].min().date()
    max_date = all_results["evaluated_at_dt"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None

# Apply filters
filtered = all_results[
    all_results["testset_name"].isin(selected_testsets)
    & all_results["model_name"].isin(selected_models)
]
if date_range and len(date_range) == 2 and filtered["evaluated_at_dt"].notna().any():
    start, end = date_range
    filtered = filtered[
        (filtered["evaluated_at_dt"].dt.date >= start)
        & (filtered["evaluated_at_dt"].dt.date <= end)
    ]

if filtered.empty:
    st.warning("No results match the current filters.")
    st.stop()

# ── Metric Charts ───────────────────────────────────────────────────────────

st.header("Metrics Over Time")
st.caption(
    "Each chart shows a specific metric. X-axis is model name ordered by evaluation "
    "datetime. Each test set is a separate line."
)

# Create a display label for x-axis: model name (keeps order by datetime)
filtered = filtered.copy()
filtered["model_label"] = filtered["model_name"]

metrics_config = [
    ("rouge1_f", "ROUGE-1 F1"),
    ("rouge2_f", "ROUGE-2 F1"),
    ("rougeL_f", "ROUGE-L F1"),
    ("bleu", "BLEU"),
]

for metric_col, metric_title in metrics_config:
    fig = px.line(
        filtered,
        x="model_label",
        y=metric_col,
        color="testset_name",
        markers=True,
        title=metric_title,
        labels={
            "model_label": "Model",
            metric_col: "Score",
            "testset_name": "Test Set",
        },
        category_orders={"model_label": filtered["model_label"].tolist()},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── Summary Table ───────────────────────────────────────────────────────────

st.header("All Evaluation Runs")

display_cols = [
    "model_name", "testset_name", "evaluated_at",
    "rouge1_f", "rouge2_f", "rougeL_f", "bleu",
    "embedding_name",
]
display_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(
    filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=300,
)
