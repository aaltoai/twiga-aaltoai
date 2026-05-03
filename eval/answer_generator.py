"""Answer generation using Twiga's llm_client.

Adapted from generate_answer.py — uses mock user objects to call
llm_client.generate_response() without real DB/WhatsApp side effects.
"""

import asyncio
from typing import Callable, Optional

import pandas as pd
from unittest.mock import AsyncMock, patch

from app.database.models import User, Message, TeacherClass, Class, Subject
from app.database.enums import UserState, MessageRole, GradeLevel, SubjectName
from app.services.llm_service import llm_client

from eval.metrics import compute_all_metrics
from eval.testset import get_reference_columns


def setup_mock_user() -> User:
    """Create mock User/Class/Subject/TeacherClass objects for evaluation."""
    mock_subject = Subject()
    mock_subject.name = SubjectName.geography

    mock_class = Class()
    mock_class.id = 1
    mock_class.grade_level = GradeLevel.os2
    mock_class.subject_ = mock_subject

    mock_teacher_class = TeacherClass()
    mock_teacher_class.class_ = mock_class

    mock_user = User(
        id=9999,
        wa_id="eval_user",
        name="Eval Teacher",
        state=UserState.active,
        class_info={"geography": ["os2"]},
    )
    mock_user.taught_classes = [mock_teacher_class]
    return mock_user


_mock_user = setup_mock_user()


async def generate_answer(question: str) -> Optional[str]:
    """Call Twiga's llm_client to generate an answer for a single question."""
    msg = Message(
        user_id=_mock_user.id,
        role=MessageRole.user,
        content=question,
    )
    with patch.object(
        llm_client, "_tool_call_notification", new=AsyncMock(return_value=None)
    ):
        result = await llm_client.generate_response(_mock_user, msg)

    if not result:
        return None

    for m in reversed(result):
        if m.role == MessageRole.assistant and m.content:
            return m.content
    return None


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
            answer = await generate_answer(question)
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
