"""Manage Testsets — Upload, validate, browse, and delete test sets."""

import streamlit as st
import pandas as pd
from pathlib import Path

from eval.testset import validate_testset, get_reference_columns, list_saved_testsets, save_testset

DATA_DIR = Path(__file__).parent.parent / "data"

st.set_page_config(page_title="Manage Testsets", page_icon="📋", layout="wide")
st.title("📋 Manage Testsets")

# ── Upload Section ──────────────────────────────────────────────────────────

st.header("Upload New Testset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="testset_upload")
testset_name = st.text_input(
    "Testset name",
    placeholder="e.g. geography_v2",
    help="Used as the filename. Only alphanumeric, hyphens, and underscores.",
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
            f"✅ Testset is valid — {len(df)} rows, "
            f"reference columns: {', '.join(ref_cols)}"
        )
    else:
        st.error("❌ Testset validation failed:")
        for err in errors:
            st.warning(err)

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if is_valid:
        if not testset_name.strip():
            st.info("Enter a testset name above to save.")
        elif st.button("💾 Save Testset", type="primary"):
            path = save_testset(df, testset_name.strip(), DATA_DIR)
            st.success(f"Saved to `{path.relative_to(DATA_DIR.parent)}`")
            st.rerun()

# ── Browse Section ──────────────────────────────────────────────────────────

st.header("Saved Testsets")

testsets = list_saved_testsets(DATA_DIR)

if not testsets:
    st.info("No testsets saved yet. Upload one above.")
else:
    for ts in testsets:
        with st.expander(f"**{ts['name']}** — {ts['rows']} rows, refs: {', '.join(ts['reference_columns'])}"):
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
