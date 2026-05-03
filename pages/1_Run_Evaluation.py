"""Run Evaluation — Select a testset, run the LLM, and view metrics."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from eval.testset import list_saved_testsets, load_testset, get_reference_columns, validate_testset
from eval.results import save_results, delete_result

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"

st.set_page_config(page_title="Run Evaluation", page_icon="▶️", layout="wide")
st.title("▶️ Run Evaluation")

# ── Testset Selection ───────────────────────────────────────────────────────

st.header("1. Select Testset")

testsets = list_saved_testsets(DATA_DIR)
testset_names = [ts["name"] for ts in testsets]

if not testset_names:
    st.warning("No testsets available. Go to **Manage Testsets** to upload one.")
    st.stop()

selected_name = st.selectbox("Saved testset", testset_names)

try:
    testset_df = load_testset(selected_name, DATA_DIR)
    ref_cols = get_reference_columns(testset_df)
    st.info(f"Loaded **{selected_name}**: {len(testset_df)} rows, references: {', '.join(ref_cols)}")
except Exception as e:
    st.error(f"Failed to load testset: {e}")
    st.stop()

with st.expander("Preview testset"):
    st.dataframe(testset_df.head(20), use_container_width=True)

# ── Model Configuration (read-only from Twiga's settings) ──────────────────

st.header("2. Model Configuration")

try:
    from app.config import llm_settings, embedding_settings

    llm_model = llm_settings.llm_name
    llm_provider = llm_settings.provider.value
    embedding_model = embedding_settings.embedder_name
    embedding_provider = embedding_settings.provider.value
except Exception:
    llm_model = "unknown (Twiga not importable)"
    llm_provider = "unknown"
    embedding_model = "unknown"
    embedding_provider = "unknown"

col1, col2 = st.columns(2)
with col1:
    st.text_input("LLM Model", value=llm_model, disabled=True)
    st.text_input("LLM Provider", value=llm_provider, disabled=True)
with col2:
    st.text_input("Embedding Model", value=embedding_model, disabled=True)
    st.text_input("Embedding Provider", value=embedding_provider, disabled=True)

st.caption("Model configuration is read from Twiga's `.env` file. Change it there to use a different model.")

# ── Run Controls ────────────────────────────────────────────────────────────

st.header("3. Run")

save_results_toggle = st.checkbox("Save results after evaluation", value=True)

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
    st.session_state.eval_run_dir = None

if st.button("🚀 Start Evaluation", type="primary"):
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
        from eval.answer_generator import run_evaluation

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
                "llm_model_name": llm_model,
                "embedding_model_name": embedding_model,
                "llm_provider": llm_provider,
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

# ── Results Display ─────────────────────────────────────────────────────────

if st.session_state.eval_results is not None:
    results_df = st.session_state.eval_results
    st.header("4. Results")

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
