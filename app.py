import streamlit as st
from pathlib import Path

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
st.markdown(
    "Evaluate Twiga AI's response quality using ROUGE and BLEU metrics. "
    "Use the sidebar to navigate between pages."
)

st.markdown("### Quick Start")
st.markdown(
    "1. **Manage Testsets** — Upload and validate test sets\n"
    "2. **Run Evaluation** — Select a test set, run the LLM, and view metrics\n"
    "3. **Performance** — Compare metrics across models and test sets over time"
)
