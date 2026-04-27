"""Multi-reference ROUGE and BLEU metric computation."""

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_smoothing = SmoothingFunction().method1


def compute_rouge(generated: str, references: list[str]) -> dict:
    """Compute ROUGE-1/2/L F1 with multi-reference support.

    For each reference, scores individually and takes the **max** per metric
    (standard multi-reference ROUGE).
    """
    best = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    for ref in references:
        if not ref:
            continue
        scores = _rouge.score(ref, generated)
        best["rouge1_f"] = max(best["rouge1_f"], scores["rouge1"].fmeasure)
        best["rouge2_f"] = max(best["rouge2_f"], scores["rouge2"].fmeasure)
        best["rougeL_f"] = max(best["rougeL_f"], scores["rougeL"].fmeasure)
    return {k: round(v, 4) for k, v in best.items()}


def compute_bleu(generated: str, references: list[str]) -> float:
    """Compute sentence-level BLEU with multi-reference support.

    sentence_bleu natively accepts a list of reference token sequences.
    """
    ref_token_lists = []
    for ref in references:
        if not ref:
            continue
        ref_token_lists.append(nltk.word_tokenize(ref.lower()))

    if not ref_token_lists:
        return 0.0

    gen_tokens = nltk.word_tokenize(generated.lower())
    score = sentence_bleu(ref_token_lists, gen_tokens, smoothing_function=_smoothing)
    return round(score, 4)


def compute_all_metrics(generated: str, references: list[str]) -> dict:
    """Compute all metrics (ROUGE-1/2/L F1, BLEU) against multiple references.

    Returns dict with keys: rouge1_f, rouge2_f, rougeL_f, bleu.
    """
    rouge = compute_rouge(generated, references)
    rouge["bleu"] = compute_bleu(generated, references)
    return rouge
