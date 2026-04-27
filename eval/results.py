"""Result save/load/delete utilities.

Results are stored as CSV + metadata.txt files in data/results/{run_id}/.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _make_run_id(model_name: str) -> str:
    """Create a unique run directory name: YYYY-MM-DD_HH-MM-SS_{model_short}."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    # Shorten model name: take last segment after /
    short = model_name.rsplit("/", 1)[-1][:40]
    # Remove characters unsafe for directory names
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in short)
    return f"{ts}_{safe}"


def save_results(
    summary_df: pd.DataFrame,
    metadata: dict,
    results_dir: Path,
) -> Path:
    """Save evaluation results and metadata.

    Args:
        summary_df: DataFrame with columns [testset_name, rouge1_f, rouge2_f, rougeL_f, bleu].
        metadata: Dict with keys like llm_model_name, embedding_model_name,
                  llm_provider, evaluated_at, testset_name, num_questions.
        results_dir: Base directory (data/results/).

    Returns:
        Path to the created run directory.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id(metadata.get("llm_model_name", "unknown"))
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save results CSV
    summary_df.to_csv(run_dir / "results.csv", index=False)

    # Save metadata as plain text
    with open(run_dir / "metadata.txt", "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    return run_dir


def _parse_metadata(meta_path: Path) -> dict:
    """Parse a metadata.txt file into a dict."""
    meta = {}
    if not meta_path.exists():
        return meta
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ": " in line:
                key, value = line.split(": ", 1)
                meta[key] = value
    return meta


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Scan all result subdirectories and return a merged DataFrame.

    Returns DataFrame with columns:
        testset_name, rouge1_f, rouge2_f, rougeL_f, bleu,
        model_name, embedding_name, evaluated_at, run_dir
    """
    rows = []
    if not results_dir.exists():
        return pd.DataFrame()

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "results.csv"
        meta_path = run_dir / "metadata.txt"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            meta = _parse_metadata(meta_path)
        except Exception:
            continue

        for _, row in df.iterrows():
            rows.append({
                "testset_name": row.get("testset_name", meta.get("testset_name", "unknown")),
                "rouge1_f": row.get("rouge1_f", 0),
                "rouge2_f": row.get("rouge2_f", 0),
                "rougeL_f": row.get("rougeL_f", 0),
                "bleu": row.get("bleu", 0),
                "model_name": meta.get("llm_model_name", "unknown"),
                "embedding_name": meta.get("embedding_model_name", "unknown"),
                "evaluated_at": meta.get("evaluated_at", ""),
                "run_dir": str(run_dir),
            })

    return pd.DataFrame(rows)


def delete_result(run_dir: Path) -> None:
    """Remove a result directory and all its contents."""
    if run_dir.exists() and run_dir.is_dir():
        shutil.rmtree(run_dir)
