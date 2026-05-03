"""Tests for eval.metrics — ROUGE and BLEU metric computation."""

import pytest

from eval.metrics import compute_all_metrics, compute_bleu, compute_rouge


# ---------------------------------------------------------------------------
# compute_rouge
# ---------------------------------------------------------------------------

class TestComputeRouge:
    def test_single_reference(self):
        gen = "The cat sat on the mat"
        refs = ["The cat sat on the mat"]
        result = compute_rouge(gen, refs)
        assert result["rouge1_f"] == pytest.approx(1.0, abs=1e-4)
        assert result["rouge2_f"] == pytest.approx(1.0, abs=1e-4)
        assert result["rougeL_f"] == pytest.approx(1.0, abs=1e-4)

    def test_multiple_references_takes_max(self):
        gen = "The cat sat on the mat"
        refs = [
            "Something completely different about dogs and chairs",
            "The cat sat on the mat",  # exact match
        ]
        result = compute_rouge(gen, refs)
        # The exact-match reference should dominate
        assert result["rouge1_f"] == pytest.approx(1.0, abs=1e-4)

    def test_partial_overlap(self):
        gen = "The cat sat on the mat"
        refs = ["The dog sat on the rug"]
        result = compute_rouge(gen, refs)
        # There is some overlap ("The", "sat", "on", "the") so scores > 0
        assert result["rouge1_f"] > 0.0
        assert result["rouge1_f"] < 1.0

    def test_empty_references(self):
        gen = "The cat sat on the mat"
        refs = ["", ""]
        result = compute_rouge(gen, refs)
        assert result["rouge1_f"] == 0.0
        assert result["rouge2_f"] == 0.0
        assert result["rougeL_f"] == 0.0

    def test_no_references(self):
        gen = "The cat sat on the mat"
        refs = []
        result = compute_rouge(gen, refs)
        assert result == {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

    def test_values_are_rounded(self):
        gen = "The cat sat on the mat and then jumped"
        refs = ["The dog sat on the rug then ran away"]
        result = compute_rouge(gen, refs)
        for v in result.values():
            # At most 4 decimal places
            assert v == round(v, 4)


# ---------------------------------------------------------------------------
# compute_bleu
# ---------------------------------------------------------------------------

class TestComputeBleu:
    def test_exact_match(self):
        gen = "The cat sat on the mat"
        refs = ["The cat sat on the mat"]
        score = compute_bleu(gen, refs)
        # Exact match should give high BLEU (may not be exactly 1.0 due to smoothing)
        assert score > 0.5

    def test_multiple_references(self):
        gen = "The cat sat on the mat"
        refs = [
            "Something completely different",
            "The cat sat on the mat",
        ]
        score = compute_bleu(gen, refs)
        assert score > 0.5

    def test_no_overlap(self):
        gen = "alpha beta gamma"
        refs = ["one two three four five"]
        score = compute_bleu(gen, refs)
        assert score < 0.1

    def test_empty_references(self):
        gen = "The cat sat on the mat"
        refs = ["", ""]
        score = compute_bleu(gen, refs)
        assert score == 0.0

    def test_no_references(self):
        gen = "The cat sat on the mat"
        refs = []
        score = compute_bleu(gen, refs)
        assert score == 0.0

    def test_value_is_rounded(self):
        gen = "The cat sat and then jumped"
        refs = ["A dog sat and then ran away"]
        score = compute_bleu(gen, refs)
        assert score == round(score, 4)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        gen = "The cat sat on the mat"
        refs = ["The cat sat on the mat"]
        result = compute_all_metrics(gen, refs)
        assert set(result.keys()) == {"rouge1_f", "rouge2_f", "rougeL_f", "bleu"}

    def test_values_are_floats(self):
        gen = "The cat sat on the mat"
        refs = ["The cat sat on the mat"]
        result = compute_all_metrics(gen, refs)
        for v in result.values():
            assert isinstance(v, float)

    def test_perfect_match_scores_high(self):
        gen = "Tanzania is a country in East Africa"
        refs = ["Tanzania is a country in East Africa"]
        result = compute_all_metrics(gen, refs)
        assert result["rouge1_f"] > 0.9
        assert result["bleu"] > 0.5
