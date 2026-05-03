"""Tests for eval.testset — validation, storage, and loading utilities."""

import pandas as pd
import pytest

from eval.testset import (
    EXTRA_REF_PATTERN,
    REQUIRED_QUERY_COL,
    REQUIRED_REF_COL,
    get_reference_columns,
    list_saved_testsets,
    load_testset,
    save_testset,
    validate_testset,
)


# ---------------------------------------------------------------------------
# validate_testset
# ---------------------------------------------------------------------------

class TestValidateTestset:
    def test_valid_minimal(self):
        df = pd.DataFrame({"user_query": ["q1"], "reference": ["r1"]})
        is_valid, errors = validate_testset(df)
        assert is_valid is True
        assert errors == []

    def test_valid_with_extra_refs(self):
        df = pd.DataFrame({
            "user_query": ["q1"],
            "reference": ["r1"],
            "reference_1": ["r2"],
            "reference_2": ["r3"],
        })
        is_valid, errors = validate_testset(df)
        assert is_valid is True

    def test_missing_user_query(self):
        df = pd.DataFrame({"reference": ["r1"], "other_col": ["x"]})
        is_valid, errors = validate_testset(df)
        assert is_valid is False
        assert any("user_query" in e for e in errors)

    def test_missing_user_query_close_match(self):
        df = pd.DataFrame({"user_qurey": ["q1"], "reference": ["r1"]})  # typo
        is_valid, errors = validate_testset(df)
        assert is_valid is False
        assert any("user_qurey" in e for e in errors)

    def test_missing_reference(self):
        df = pd.DataFrame({"user_query": ["q1"], "other": ["x"]})
        is_valid, errors = validate_testset(df)
        assert is_valid is False
        assert any("reference" in e for e in errors)

    def test_missing_reference_close_match(self):
        df = pd.DataFrame({"user_query": ["q1"], "referenc": ["r1"]})  # typo
        is_valid, errors = validate_testset(df)
        assert is_valid is False
        assert any("referenc" in e for e in errors)

    def test_extra_refs_without_base_reference(self):
        df = pd.DataFrame({
            "user_query": ["q1"],
            "reference_1": ["r1"],
            "reference_2": ["r2"],
        })
        is_valid, errors = validate_testset(df)
        assert is_valid is False
        assert any("base" in e.lower() or "reference" in e for e in errors)

    def test_malformed_reference_column(self):
        df = pd.DataFrame({
            "user_query": ["q1"],
            "reference": ["r1"],
            "reference_foo": ["bad"],
        })
        is_valid, errors = validate_testset(df)
        # Malformed column triggers a warning but df is still valid
        # because user_query and reference exist
        assert any("reference_foo" in e for e in errors)

    def test_extra_columns_preserved(self):
        """Extra columns should not cause validation failure."""
        df = pd.DataFrame({
            "user_query": ["q1"],
            "reference": ["r1"],
            "persona_name": ["Alice"],
            "query_style": ["PERFECT_GRAMMAR"],
        })
        is_valid, errors = validate_testset(df)
        assert is_valid is True


# ---------------------------------------------------------------------------
# get_reference_columns
# ---------------------------------------------------------------------------

class TestGetReferenceColumns:
    def test_only_base_reference(self):
        df = pd.DataFrame({"user_query": ["q"], "reference": ["r"]})
        assert get_reference_columns(df) == ["reference"]

    def test_with_extra_refs(self):
        df = pd.DataFrame({
            "user_query": ["q"],
            "reference": ["r"],
            "reference_2": ["r2"],
            "reference_1": ["r1"],
        })
        cols = get_reference_columns(df)
        assert cols == ["reference", "reference_1", "reference_2"]

    def test_no_reference_columns(self):
        df = pd.DataFrame({"user_query": ["q"], "other": ["x"]})
        assert get_reference_columns(df) == []

    def test_only_extra_refs_no_base(self):
        df = pd.DataFrame({"user_query": ["q"], "reference_1": ["r1"]})
        cols = get_reference_columns(df)
        # Should return only the extra refs (base is missing)
        assert cols == ["reference_1"]


# ---------------------------------------------------------------------------
# save_testset / load_testset round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadTestset:
    def test_round_trip(self, tmp_path):
        df = pd.DataFrame({"user_query": ["q1", "q2"], "reference": ["r1", "r2"]})
        save_testset(df, "my_testset", tmp_path)
        loaded = load_testset("my_testset", tmp_path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_sanitizes_name(self, tmp_path):
        df = pd.DataFrame({"user_query": ["q1"], "reference": ["r1"]})
        path = save_testset(df, "bad name!@#$", tmp_path)
        assert "!" not in path.stem
        assert "@" not in path.stem
        assert path.exists()

    def test_empty_name_becomes_unnamed(self, tmp_path):
        df = pd.DataFrame({"user_query": ["q1"], "reference": ["r1"]})
        path = save_testset(df, "!!!", tmp_path)
        assert path.stem == "unnamed"

    def test_creates_testsets_subdir(self, tmp_path):
        df = pd.DataFrame({"user_query": ["q1"], "reference": ["r1"]})
        save_testset(df, "test", tmp_path)
        assert (tmp_path / "testsets").is_dir()


# ---------------------------------------------------------------------------
# list_saved_testsets
# ---------------------------------------------------------------------------

class TestListSavedTestsets:
    def test_empty_dir(self, tmp_path):
        assert list_saved_testsets(tmp_path) == []

    def test_lists_saved_csv(self, tmp_path):
        df = pd.DataFrame({"user_query": ["q1", "q2"], "reference": ["r1", "r2"]})
        save_testset(df, "alpha", tmp_path)
        save_testset(df, "beta", tmp_path)
        testsets = list_saved_testsets(tmp_path)
        names = [ts["name"] for ts in testsets]
        assert "alpha" in names
        assert "beta" in names

    def test_metadata_correct(self, tmp_path):
        df = pd.DataFrame({
            "user_query": ["q1", "q2"],
            "reference": ["r1", "r2"],
            "reference_1": ["r1a", "r2a"],
        })
        save_testset(df, "meta_test", tmp_path)
        testsets = list_saved_testsets(tmp_path)
        ts = testsets[0]
        assert ts["rows"] == 2
        assert "reference" in ts["reference_columns"]
        assert "reference_1" in ts["reference_columns"]
