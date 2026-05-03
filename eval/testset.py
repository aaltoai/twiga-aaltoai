"""Testset validation, storage, and loading utilities."""

import re
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

REQUIRED_QUERY_COL = "user_query"
REQUIRED_REF_COL = "reference"
EXTRA_REF_PATTERN = re.compile(r"^reference_\d+$")


def validate_testset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that a DataFrame meets the testset column requirements.

    Returns (is_valid, list_of_error_messages).
    """
    errors: list[str] = []
    cols = list(df.columns)

    # --- Check user_query column ---
    query_cols = [c for c in cols if c == REQUIRED_QUERY_COL]
    if len(query_cols) == 0:
        close = get_close_matches(REQUIRED_QUERY_COL, cols, n=3, cutoff=0.5)
        if close:
            hint = ", ".join(f"'{c}'" for c in close)
            errors.append(
                f"Missing required column '{REQUIRED_QUERY_COL}'. "
                f"Found similar column(s): {hint} — rename to '{REQUIRED_QUERY_COL}'?"
            )
        else:
            errors.append(
                f"Missing required column '{REQUIRED_QUERY_COL}'. "
                f"Available columns: {', '.join(cols)}"
            )
    elif len(query_cols) > 1:
        errors.append(
            f"Found multiple '{REQUIRED_QUERY_COL}' columns. "
            f"Only exactly 1 is allowed to avoid ambiguity."
        )

    # --- Check reference columns ---
    ref_cols = [c for c in cols if c == REQUIRED_REF_COL]
    extra_ref_cols = [c for c in cols if EXTRA_REF_PATTERN.match(c)]

    if len(ref_cols) == 0 and len(extra_ref_cols) == 0:
        close = get_close_matches(REQUIRED_REF_COL, cols, n=3, cutoff=0.5)
        if close:
            hint = ", ".join(f"'{c}'" for c in close)
            errors.append(
                f"Missing required column '{REQUIRED_REF_COL}'. "
                f"Found similar column(s): {hint} — rename to '{REQUIRED_REF_COL}'?"
            )
        else:
            errors.append(
                f"Missing required column '{REQUIRED_REF_COL}'. "
                f"At least one reference column is required."
            )
    elif len(ref_cols) == 0:
        errors.append(
            f"Found extra reference columns ({', '.join(extra_ref_cols)}) but missing "
            f"the base '{REQUIRED_REF_COL}' column. Please add a '{REQUIRED_REF_COL}' column."
        )

    # --- Flag malformed reference_* columns ---
    for c in cols:
        if c.startswith("reference_") and not EXTRA_REF_PATTERN.match(c) and c != REQUIRED_REF_COL:
            errors.append(
                f"Column '{c}' looks like an extra reference but doesn't match "
                f"the expected pattern 'reference_N' (e.g. 'reference_1', 'reference_2'). "
                f"Rename it or it will be ignored."
            )

    return (len(errors) == 0, errors)


def get_reference_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of valid reference columns in the DataFrame."""
    cols = []
    if REQUIRED_REF_COL in df.columns:
        cols.append(REQUIRED_REF_COL)
    for c in sorted(df.columns):
        if EXTRA_REF_PATTERN.match(c):
            cols.append(c)
    return cols


def list_saved_testsets(data_dir: Path) -> list[dict]:
    """List all saved testset CSVs with metadata."""
    testsets = []
    testsets_dir = data_dir / "testsets"
    if not testsets_dir.exists():
        return testsets
    for csv_path in sorted(testsets_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, nrows=0)  # header only
            row_count = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
            ref_cols = get_reference_columns(df)
            testsets.append({
                "name": csv_path.stem,
                "path": csv_path,
                "rows": max(row_count, 0),
                "columns": list(df.columns),
                "reference_columns": ref_cols,
            })
        except Exception:
            continue
    return testsets


def save_testset(df: pd.DataFrame, name: str, data_dir: Path) -> Path:
    """Save a testset DataFrame as a CSV."""
    testsets_dir = data_dir / "testsets"
    testsets_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize name: keep alphanumeric, hyphens, underscores
    safe_name = re.sub(r"[^\w\-]", "_", name).strip("_")
    if not safe_name:
        safe_name = "unnamed"
    path = testsets_dir / f"{safe_name}.csv"
    df.to_csv(path, index=False)
    return path


def load_testset(name: str, data_dir: Path) -> pd.DataFrame:
    """Load a saved testset by name."""
    path = data_dir / "testsets" / f"{name}.csv"
    return pd.read_csv(path)
