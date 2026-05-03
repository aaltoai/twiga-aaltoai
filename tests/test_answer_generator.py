"""Tests for eval.answer_generator — LLM answer generation and evaluation.

All twiga `app.*` imports are stubbed in conftest.py so these tests run
without a database, env vars, or a live LLM.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# The conftest stubs are already installed. Import the module under test.
from eval.answer_generator import (
    generate_answer,
    run_evaluation,
    setup_mock_user,
)


# ---------------------------------------------------------------------------
# setup_mock_user
# ---------------------------------------------------------------------------

class TestSetupMockUser:
    def test_returns_user_object(self):
        user = setup_mock_user()
        assert user.id == 9999
        assert user.wa_id == "eval_user"
        assert user.name == "Eval Teacher"

    def test_user_has_taught_classes(self):
        user = setup_mock_user()
        assert hasattr(user, "taught_classes")
        assert len(user.taught_classes) == 1

    def test_class_info_dict(self):
        user = setup_mock_user()
        assert "geography" in user.class_info


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_returns_assistant_content(self):
        MessageRole = sys.modules["app.database.enums"].MessageRole
        mock_msg = MagicMock()
        mock_msg.role = MessageRole.assistant
        mock_msg.content = "Generated answer text"

        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=[mock_msg])

        result = await generate_answer("What is geography?")
        assert result == "Generated answer text"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_result(self):
        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=None)

        result = await generate_answer("What is geography?")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_empty_list(self):
        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=[])

        result = await generate_answer("What is geography?")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_last_assistant_message(self):
        MessageRole = sys.modules["app.database.enums"].MessageRole

        msg1 = MagicMock()
        msg1.role = MessageRole.user
        msg1.content = "echo"

        msg2 = MagicMock()
        msg2.role = MessageRole.assistant
        msg2.content = "First response"

        msg3 = MagicMock()
        msg3.role = MessageRole.assistant
        msg3.content = "Final response"

        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=[msg1, msg2, msg3])

        result = await generate_answer("test")
        assert result == "Final response"


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    @pytest.mark.asyncio
    async def test_basic_evaluation(self):
        MessageRole = sys.modules["app.database.enums"].MessageRole
        mock_msg = MagicMock()
        mock_msg.role = MessageRole.assistant
        mock_msg.content = "Tanzania is in East Africa"

        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=[mock_msg])

        df = pd.DataFrame({
            "user_query": ["Where is Tanzania?"],
            "reference": ["Tanzania is located in East Africa"],
        })
        result = await run_evaluation(df, ["reference"])

        assert "generated_response" in result.columns
        assert "rouge1_f" in result.columns
        assert "rouge2_f" in result.columns
        assert "rougeL_f" in result.columns
        assert "bleu" in result.columns
        assert len(result) == 1
        assert result.iloc[0]["rouge1_f"] > 0.0

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        MessageRole = sys.modules["app.database.enums"].MessageRole
        mock_msg = MagicMock()
        mock_msg.role = MessageRole.assistant
        mock_msg.content = "answer"

        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=[mock_msg])

        df = pd.DataFrame({
            "user_query": ["q1", "q2"],
            "reference": ["r1", "r2"],
        })
        callback = MagicMock()
        await run_evaluation(df, ["reference"], progress_callback=callback)

        # Should be called for each row + final completion
        assert callback.call_count >= 2

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(side_effect=RuntimeError("LLM down"))

        df = pd.DataFrame({
            "user_query": ["question"],
            "reference": ["answer"],
        })
        result = await run_evaluation(df, ["reference"])

        # Should not crash; generated_response should contain error
        assert "[ERROR]" in result.iloc[0]["generated_response"]
        # Metrics should be 0 since generated text contains error
        assert result.iloc[0]["rouge1_f"] == pytest.approx(0.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_multiple_reference_columns(self):
        MessageRole = sys.modules["app.database.enums"].MessageRole
        mock_msg = MagicMock()
        mock_msg.role = MessageRole.assistant
        mock_msg.content = "Coffee is important in Tanzania"

        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=[mock_msg])

        df = pd.DataFrame({
            "user_query": ["Tell me about coffee in Tanzania"],
            "reference": ["Coffee is a major export of Tanzania"],
            "reference_1": ["Tanzania produces arabica coffee"],
        })
        result = await run_evaluation(df, ["reference", "reference_1"])
        assert result.iloc[0]["rouge1_f"] > 0.0

    @pytest.mark.asyncio
    async def test_preserves_original_columns(self):
        MessageRole = sys.modules["app.database.enums"].MessageRole
        mock_msg = MagicMock()
        mock_msg.role = MessageRole.assistant
        mock_msg.content = "answer"

        llm_client = sys.modules["app.services.llm_service"].llm_client
        llm_client.generate_response = AsyncMock(return_value=[mock_msg])

        df = pd.DataFrame({
            "user_query": ["q1"],
            "reference": ["r1"],
            "persona_name": ["Alice"],
        })
        result = await run_evaluation(df, ["reference"])
        assert "persona_name" in result.columns
        assert result.iloc[0]["persona_name"] == "Alice"
