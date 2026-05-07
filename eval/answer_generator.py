"""Answer generation — delegates to answer_generation/twiga_runner.py."""

import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# Make answer_generation/ importable as a namespace package
sys.path.insert(0, str(Path(__file__).parent.parent))
from answer_generation.twiga_runner import run_twiga  # noqa: E402

from eval.metrics import compute_all_metrics


async def run_evaluation(
    testset_df: pd.DataFrame,
    reference_cols: list[str],
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> pd.DataFrame:
    """Run evaluation on a testset DataFrame.

    Args:
        testset_df: DataFrame with 'user_query' column and reference columns.
        reference_cols: List of reference column names to score against.
        progress_callback: Optional callback(current_row, total_rows, status_text).

    Returns:
        DataFrame with original columns + 'generated_response', 'rouge1_f',
        'rouge2_f', 'rougeL_f', 'bleu'.
    """
    results = testset_df.copy()
    total = len(results)

    generated_responses = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []

    for i in range(total):
        question = str(results.iloc[i]["user_query"])

        if progress_callback:
            progress_callback(i, total, f"Generating answer for question {i + 1}/{total}...")

        try:
            result = await run_twiga(question)
            answer = result["twiga_answer"]
        except Exception as e:
            answer = f"[ERROR] {e}"

        generated = answer or ""
        generated_responses.append(generated)

        # Collect references for this row
        refs = []
        for col in reference_cols:
            val = results.iloc[i].get(col)
            if pd.notna(val) and str(val).strip():
                refs.append(str(val).strip())

        if generated and refs:
            metrics = compute_all_metrics(generated, refs)
        else:
            metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0, "bleu": 0.0}

        rouge1_scores.append(metrics["rouge1_f"])
        rouge2_scores.append(metrics["rouge2_f"])
        rougeL_scores.append(metrics["rougeL_f"])
        bleu_scores.append(metrics["bleu"])

    results["generated_response"] = generated_responses
    results["rouge1_f"] = rouge1_scores
    results["rouge2_f"] = rouge2_scores
    results["rougeL_f"] = rougeL_scores
    results["bleu"] = bleu_scores

    if progress_callback:
        progress_callback(total, total, "Evaluation complete.")

    return results
