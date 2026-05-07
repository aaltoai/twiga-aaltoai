"""Twiga Evaluation Dashboard — Single unified interface."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from eval.testset import (
    list_saved_testsets,
    load_testset,
    get_reference_columns,
    validate_testset,
    save_testset,
)
from eval.results import save_results, delete_result, load_all_results
from eval.answer_generator import run_evaluation

st.set_page_config(
    page_title="Twiga Evaluation Dashboard",
    page_icon="🦒",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
TESTSETS_DIR = DATA_DIR / "testsets"
RESULTS_DIR = DATA_DIR / "results"

# Ensure data directories exist
TESTSETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

st.title("🦒 Twiga Evaluation Dashboard")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: MANAGE TESTSETS
# ════════════════════════════════════════════════════════════════════════════

st.header("📋 1. Manage Testsets")

col_upload, col_name = st.columns([3, 2])
with col_upload:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="testset_upload")
with col_name:
    testset_name = st.text_input(
        "Testset name",
        placeholder="e.g. geography_v2",
        help="Only alphanumeric, hyphens, underscores.",
    )

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    is_valid, errors = validate_testset(df)

    if is_valid:
        ref_cols = get_reference_columns(df)
        st.success(
            f"✅ Valid — {len(df)} rows, "
            f"references: {', '.join(ref_cols)}"
        )
    else:
        st.error("❌ Validation failed:")
        for err in errors:
            st.warning(err)

    with st.expander("Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    if is_valid:
        if testset_name.strip():
            if st.button("💾 Save Testset", type="primary", key="save_testset_btn"):
                path = save_testset(df, testset_name.strip(), DATA_DIR)
                st.success(f"Saved to `{path.relative_to(DATA_DIR.parent)}`")
                st.rerun()
        else:
            st.info("Enter a testset name to save.")

st.subheader("Saved Testsets")
testsets = list_saved_testsets(DATA_DIR)

if not testsets:
    st.info("No testsets saved yet.")
else:
    for ts in testsets:
        with st.expander(
            f"**{ts['name']}** — {ts['rows']} rows, refs: {', '.join(ts['reference_columns'])}"
        ):
            col1, col2 = st.columns([4, 1])
            with col1:
                try:
                    preview = pd.read_csv(ts["path"], nrows=10)
                    st.dataframe(preview, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read: {e}")
            with col2:
                st.markdown(f"**Columns:** {', '.join(ts['columns'])}")
                if st.button("🗑️ Delete", key=f"del_{ts['name']}"):
                    ts["path"].unlink()
                    st.success(f"Deleted '{ts['name']}'")
                    st.rerun()

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: RUN EVALUATION
# ════════════════════════════════════════════════════════════════════════════

st.header("▶️ 2. Run Evaluation")

testsets = list_saved_testsets(DATA_DIR)
testset_names = [ts["name"] for ts in testsets]

if not testset_names:
    st.warning("No testsets available. Upload one in section 1 above.")
else:
    st.subheader("Select Testset")
    selected_name = st.selectbox("Saved testset", testset_names, key="eval_testset")

    try:
        testset_df = load_testset(selected_name, DATA_DIR)
        ref_cols = get_reference_columns(testset_df)
        st.info(f"Loaded **{selected_name}**: {len(testset_df)} rows, references: {', '.join(ref_cols)}")
    except Exception as e:
        st.error(f"Failed to load testset: {e}")
        st.stop()

    with st.expander("Preview testset"):
        st.dataframe(testset_df.head(20), use_container_width=True)

    st.subheader("Run Evaluation")

    save_results_toggle = st.checkbox("Save results after evaluation", value=True)

    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
        st.session_state.eval_run_dir = None

    if st.button("🚀 Start Evaluation", type="primary", key="start_eval"):
        st.session_state.eval_results = None
        st.session_state.eval_run_dir = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current: int, total: int, text: str | None = None):
            if total > 0:
                progress_bar.progress(current / total)
            if text:
                status_text.text(text)

        try:
            results_df = asyncio.run(
                run_evaluation(testset_df, ref_cols, progress_callback)
            )
            st.session_state.eval_results = results_df

            # Auto-save if toggle is on
            if save_results_toggle:
                evaluated_at = datetime.now(timezone.utc).isoformat()
                metric_cols = ["rouge1_f", "rouge2_f", "rougeL_f", "bleu"]
                summary = pd.DataFrame([{
                    "testset_name": selected_name,
                    **{m: results_df[m].mean() for m in metric_cols},
                }])
                metadata = {
                    "evaluated_at": evaluated_at,
                    "testset_name": selected_name,
                    "num_questions": len(results_df),
                }
                run_dir = save_results(summary, metadata, RESULTS_DIR)
                st.session_state.eval_run_dir = run_dir
                st.success(f"Results saved to `{run_dir.name}/`")

            progress_bar.progress(1.0)
            status_text.text("Evaluation complete.")

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display results if available
    if st.session_state.eval_results is not None:
        results_df = st.session_state.eval_results
        st.subheader("Results")

        # Summary metrics
        metric_cols = ["rouge1_f", "rouge2_f", "rougeL_f", "bleu"]
        metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

        cols = st.columns(4)
        for col, metric, label in zip(cols, metric_cols, metric_labels):
            with col:
                avg = results_df[metric].mean()
                st.metric(label, f"{avg:.4f}")

        # Detailed results table
        st.subheader("Per-question Results")
        display_cols = ["user_query", "generated_response"] + [
            c for c in ref_cols if c in results_df.columns
        ] + metric_cols
        display_cols = [c for c in display_cols if c in results_df.columns]
        st.dataframe(results_df[display_cols], use_container_width=True, height=400)

        # Save opt-out: delete if user unchecks after save
        if st.session_state.eval_run_dir is not None and not save_results_toggle:
            run_dir = Path(st.session_state.eval_run_dir)
            if run_dir.exists():
                delete_result(run_dir)
                st.session_state.eval_run_dir = None
                st.info("Saved results removed (save toggle unchecked).")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: PERFORMANCE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

st.header("📊 3. Performance Overview")

all_results = load_all_results(RESULTS_DIR)

if all_results.empty:
    st.info(
        "No evaluation results saved yet. "
        "Run an evaluation in section 2 with 'Save results' enabled."
    )
else:
    # Parse evaluated_at as datetime for sorting
    all_results["evaluated_at_dt"] = pd.to_datetime(all_results["evaluated_at"], errors="coerce")
    all_results = all_results.sort_values("evaluated_at_dt")

    # Filters in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        available_testsets = sorted(all_results["testset_name"].unique())
        selected_testsets = st.multiselect(
            "Test sets", available_testsets, default=available_testsets, key="perf_testsets"
        )

    with col2:
        available_models = sorted(all_results["model_name"].unique())
        selected_models = st.multiselect(
            "Models", available_models, default=available_models, key="perf_models"
        )

    with col3:
        if all_results["evaluated_at_dt"].notna().any():
            min_date = all_results["evaluated_at_dt"].min().date()
            max_date = all_results["evaluated_at_dt"].max().date()
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="perf_date_range",
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
    else:
        # Metric Charts
        st.subheader("Metrics Over Time")

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

        # Summary Table
        st.subheader("All Evaluation Runs")

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
