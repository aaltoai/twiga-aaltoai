"""Tests for eval.results — result save/load/delete utilities."""

import pandas as pd
import pytest

from eval.results import (
    _make_run_id,
    _parse_metadata,
    delete_result,
    load_all_results,
    save_results,
)


# ---------------------------------------------------------------------------
# _make_run_id
# ---------------------------------------------------------------------------

class TestMakeRunId:
    def test_format(self):
        run_id = _make_run_id("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        # Format: YYYY-MM-DD_HH-MM-SS_<model_short>
        # split("_", 2) gives [date, time, model]
        parts = run_id.split("_", 2)
        assert len(parts) == 3
        # First part is date YYYY-MM-DD
        assert len(parts[0]) == 10  # YYYY-MM-DD
        # Second part is time HH-MM-SS
        assert len(parts[1]) == 8  # HH-MM-SS

    def test_unsafe_chars_removed(self):
        run_id = _make_run_id("model/with spaces & special!")
        assert " " not in run_id
        assert "&" not in run_id
        assert "!" not in run_id

    def test_long_model_name_truncated(self):
        run_id = _make_run_id("provider/" + "a" * 100)
        # Model part should be at most 40 chars
        # Total run_id = timestamp + "_" + model_short
        model_part = run_id.split("_", 3)[-1]
        assert len(model_part) <= 40


# ---------------------------------------------------------------------------
# _parse_metadata
# ---------------------------------------------------------------------------

class TestParseMetadata:
    def test_parses_key_value(self, tmp_path):
        meta_file = tmp_path / "metadata.txt"
        meta_file.write_text(
            "llm_model_name: test-model\n"
            "evaluated_at: 2026-01-01T00:00:00\n"
            "num_questions: 10\n",
            encoding="utf-8",
        )
        meta = _parse_metadata(meta_file)
        assert meta["llm_model_name"] == "test-model"
        assert meta["evaluated_at"] == "2026-01-01T00:00:00"
        assert meta["num_questions"] == "10"

    def test_missing_file(self, tmp_path):
        meta = _parse_metadata(tmp_path / "nonexistent.txt")
        assert meta == {}

    def test_value_with_colon(self, tmp_path):
        meta_file = tmp_path / "metadata.txt"
        meta_file.write_text("url: http://example.com:8080\n", encoding="utf-8")
        meta = _parse_metadata(meta_file)
        assert meta["url"] == "http://example.com:8080"


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_creates_run_dir(self, tmp_path):
        df = pd.DataFrame([{
            "testset_name": "test",
            "rouge1_f": 0.5,
            "rouge2_f": 0.3,
            "rougeL_f": 0.4,
            "bleu": 0.2,
        }])
        metadata = {
            "llm_model_name": "test-model",
            "evaluated_at": "2026-01-01T00:00:00",
        }
        run_dir = save_results(df, metadata, tmp_path)
        assert run_dir.is_dir()
        assert (run_dir / "results.csv").exists()
        assert (run_dir / "metadata.txt").exists()

    def test_results_csv_content(self, tmp_path):
        df = pd.DataFrame([{
            "testset_name": "test",
            "rouge1_f": 0.55,
            "bleu": 0.22,
        }])
        run_dir = save_results(df, {"llm_model_name": "m"}, tmp_path)
        loaded = pd.read_csv(run_dir / "results.csv")
        assert loaded.iloc[0]["rouge1_f"] == pytest.approx(0.55)

    def test_metadata_txt_content(self, tmp_path):
        df = pd.DataFrame([{"testset_name": "test"}])
        metadata = {"llm_model_name": "my-model", "num_questions": "5"}
        run_dir = save_results(df, metadata, tmp_path)
        text = (run_dir / "metadata.txt").read_text(encoding="utf-8")
        assert "llm_model_name: my-model" in text
        assert "num_questions: 5" in text


# ---------------------------------------------------------------------------
# load_all_results
# ---------------------------------------------------------------------------

class TestLoadAllResults:
    def test_empty_dir(self, tmp_path):
        result = load_all_results(tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_nonexistent_dir(self, tmp_path):
        result = load_all_results(tmp_path / "nope")
        assert result.empty

    def test_loads_single_run(self, tmp_path):
        df = pd.DataFrame([{
            "testset_name": "ts1",
            "rouge1_f": 0.5,
            "rouge2_f": 0.3,
            "rougeL_f": 0.4,
            "bleu": 0.2,
        }])
        metadata = {
            "llm_model_name": "test-model",
            "embedding_model_name": "test-embed",
            "evaluated_at": "2026-01-01T00:00:00",
        }
        save_results(df, metadata, tmp_path)
        all_results = load_all_results(tmp_path)
        assert len(all_results) == 1
        assert all_results.iloc[0]["model_name"] == "test-model"
        assert all_results.iloc[0]["rouge1_f"] == pytest.approx(0.5)

    def test_loads_multiple_runs(self, tmp_path):
        for i in range(3):
            df = pd.DataFrame([{
                "testset_name": f"ts{i}",
                "rouge1_f": 0.1 * i,
                "rouge2_f": 0.0,
                "rougeL_f": 0.0,
                "bleu": 0.0,
            }])
            metadata = {"llm_model_name": f"model-{i}"}
            save_results(df, metadata, tmp_path)
        all_results = load_all_results(tmp_path)
        assert len(all_results) == 3


# ---------------------------------------------------------------------------
# delete_result
# ---------------------------------------------------------------------------

class TestDeleteResult:
    def test_deletes_run_dir(self, tmp_path):
        df = pd.DataFrame([{"testset_name": "test"}])
        run_dir = save_results(df, {"llm_model_name": "m"}, tmp_path)
        assert run_dir.exists()
        delete_result(run_dir)
        assert not run_dir.exists()

    def test_delete_nonexistent_is_noop(self, tmp_path):
        # Should not raise
        delete_result(tmp_path / "nonexistent")
