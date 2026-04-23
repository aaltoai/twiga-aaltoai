"""
Evaluate AI-generated answers against references using ROUGE and BLEU scores.

Reads from testset_with_answers_qwen3.5_397B_A17B.csv and computes:
- ROUGE-1 (unigram overlap)
- ROUGE-2 (bigram overlap)
- ROUGE-L (longest common subsequence)
- BLEU score (with smoothing)
"""

import csv
import sys
from pathlib import Path

try:
    from rouge_score import rouge_scorer
except ImportError:
    sys.exit("Missing dependency: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Ensure punkt tokenizer is available
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
except ImportError:
    sys.exit("Missing dependency: pip install nltk")


def compute_scores(reference: str, hypothesis: str, scorer, smoothing):
    """Compute ROUGE and BLEU scores for a single pair."""
    # ROUGE
    rouge_results = scorer.score(reference, hypothesis)

    # BLEU (tokenize by whitespace)
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    bleu = sentence_bleu(
        [ref_tokens], hyp_tokens, smoothing_function=smoothing.method1
    )

    return {
        "rouge1_precision": rouge_results["rouge1"].precision,
        "rouge1_recall": rouge_results["rouge1"].recall,
        "rouge1_f1": rouge_results["rouge1"].fmeasure,
        "rouge2_precision": rouge_results["rouge2"].precision,
        "rouge2_recall": rouge_results["rouge2"].recall,
        "rouge2_f1": rouge_results["rouge2"].fmeasure,
        "rougeL_precision": rouge_results["rougeL"].precision,
        "rougeL_recall": rouge_results["rougeL"].recall,
        "rougeL_f1": rouge_results["rougeL"].fmeasure,
        "bleu": bleu,
    }


def main():
    input_path = Path(__file__).parent / "testset_with_answers_qwen3.5_397B_A17B.csv"
    output_path = Path(__file__).parent / "evaluation_rouge_bleu_results.csv"

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction()

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        sys.exit("No data rows found in the CSV.")

    results = []
    agg = {k: 0.0 for k in [
        "rouge1_precision", "rouge1_recall", "rouge1_f1",
        "rouge2_precision", "rouge2_recall", "rouge2_f1",
        "rougeL_precision", "rougeL_recall", "rougeL_f1",
        "bleu",
    ]}

    for i, row in enumerate(rows):
        reference = row.get("reference", "").strip()
        actual_output = row.get("actual_output", "").strip()

        if not reference or not actual_output:
            print(f"Row {i+1}: skipping (empty reference or actual_output)")
            continue

        scores = compute_scores(reference, actual_output, scorer, smoothing)
        scores["row"] = i + 1
        scores["user_input"] = row.get("user_input", "")[:80]
        results.append(scores)

        for k in agg:
            agg[k] += scores[k]

    n = len(results)
    if n == 0:
        sys.exit("No valid rows to evaluate.")

    # Print per-row results
    print(f"{'Row':>4}  {'ROUGE-1 F1':>11}  {'ROUGE-2 F1':>11}  {'ROUGE-L F1':>11}  {'BLEU':>6}  Question")
    print("-" * 100)
    for r in results:
        print(
            f"{r['row']:4d}  {r['rouge1_f1']:11.4f}  {r['rouge2_f1']:11.4f}  "
            f"{r['rougeL_f1']:11.4f}  {r['bleu']:6.4f}  {r['user_input']}"
        )

    # Print averages
    print("-" * 100)
    print(f"\nAverage scores over {n} samples:")
    print(f"  ROUGE-1  P: {agg['rouge1_precision']/n:.4f}  R: {agg['rouge1_recall']/n:.4f}  F1: {agg['rouge1_f1']/n:.4f}")
    print(f"  ROUGE-2  P: {agg['rouge2_precision']/n:.4f}  R: {agg['rouge2_recall']/n:.4f}  F1: {agg['rouge2_f1']/n:.4f}")
    print(f"  ROUGE-L  P: {agg['rougeL_precision']/n:.4f}  R: {agg['rougeL_recall']/n:.4f}  F1: {agg['rougeL_f1']/n:.4f}")
    print(f"  BLEU:    {agg['bleu']/n:.4f}")

    # Write detailed results to CSV
    fieldnames = [
        "row", "user_input",
        "rouge1_precision", "rouge1_recall", "rouge1_f1",
        "rouge2_precision", "rouge2_recall", "rouge2_f1",
        "rougeL_precision", "rougeL_recall", "rougeL_f1",
        "bleu",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        # Write average row
        avg_row = {k: agg.get(k, 0) / n for k in fieldnames if k in agg}
        avg_row["row"] = ""
        avg_row["user_input"] = "AVERAGE"
        writer.writerow(avg_row)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
