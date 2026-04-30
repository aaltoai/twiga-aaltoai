"""Pure library: call the Twiga agent on a question, return structured result.

No file I/O. Callers (CSV runners, notebooks, tests, future orchestrators)
build their own batching/persistence on top of this.
"""

from typing import TypedDict
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv
load_dotenv()

from app.database.models import User, Message, TeacherClass, Class, Subject
from app.database.enums import UserState, MessageRole, GradeLevel, SubjectName
from app.services.llm_service import llm_client


class TwigaResult(TypedDict):
    twiga_answer: str | None
    twiga_retrieved_context: str | None
    twiga_tool_call: str | None
    tool_call_output_returned: str | None


def _build_mock_user() -> User:
    """Mock User wired to Geography Form 2 — no DB writes, avoids FK conflicts."""
    subject = Subject()
    subject.name = SubjectName.geography

    cls = Class()
    cls.id = 1
    cls.grade_level = GradeLevel.os2
    cls.subject_ = subject

    teacher_class = TeacherClass()
    teacher_class.class_ = cls

    user = User(
        id=9999,
        wa_id="eval_user",
        name="Eval Teacher",
        state=UserState.active,
        class_info={"geography": ["os2"]},
    )
    user.taught_classes = [teacher_class]
    return user


_mock_user = _build_mock_user()


async def run_twiga(question: str) -> TwigaResult:
    """Run the Twiga agent on a single question and return the answer + tool metadata.

    The returned dict matches the columns written into testset_with_twiga_answers.csv,
    so callers can spread it directly into a row.
    """
    msg = Message(
        user_id=_mock_user.id,
        role=MessageRole.user,
        content=question,
    )
    # mock _tool_call_notification to skip DB writes + WhatsApp sends
    with patch.object(llm_client, "_tool_call_notification", new=AsyncMock(return_value=None)):
        result = await llm_client.generate_response(_mock_user, msg)

    out: TwigaResult = {
        "twiga_answer": None,
        "twiga_retrieved_context": None,
        "twiga_tool_call": None,
        "tool_call_output_returned": None,
    }

    if not result:
        return out

    tool_names: list[str] = []
    tool_outputs: list[str] = []
    tool_call_count = 0

    for m in result:
        if m.role == MessageRole.assistant and m.tool_calls:
            for tc in m.tool_calls:
                tool_call_count += 1
                name = tc.get("function", {}).get("name")
                if name:
                    tool_names.append(name)
        elif m.role == MessageRole.tool and m.content:
            tool_outputs.append(m.content)

    # Final assistant message with content is the answer
    for m in reversed(result):
        if m.role == MessageRole.assistant and m.content:
            out["twiga_answer"] = m.content
            break

    if tool_call_count > 0:
        out["twiga_tool_call"] = ", ".join(tool_names) if tool_names else None
        # "yes" only if every tool call produced a non-empty tool message
        out["tool_call_output_returned"] = "yes" if len(tool_outputs) >= tool_call_count else "no"
    if tool_outputs:
        out["twiga_retrieved_context"] = "\n---\n".join(tool_outputs)

    return out
