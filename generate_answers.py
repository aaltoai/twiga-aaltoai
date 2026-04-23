import asyncio
import pandas as pd
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv
load_dotenv('/home/athul/my_workspace/twiga_project/twiga/.env')


from app.database.models import User, Message, TeacherClass, Class, Subject
from app.database.enums import UserState, MessageRole, GradeLevel, SubjectName
from app.services.llm_service import llm_client


mock_subject = Subject()
mock_subject.name = SubjectName.geography

mock_class = Class()
mock_class.id = 1          # Geography Form 2 in the DB
mock_class.grade_level = GradeLevel.os2
mock_class.subject_ = mock_subject

mock_teacher_class = TeacherClass()
mock_teacher_class.class_ = mock_class

mock_user = User(
    id=9999,               # fake id — no history in DB, avoids FK conflicts
    wa_id="eval_user",
    name="Eval Teacher",
    state=UserState.active,
    class_info={"geography": ["os2"]}
)
mock_user.taught_classes = [mock_teacher_class]


async def generate_answer(question: str) -> str | None:
    msg = Message(
        user_id=mock_user.id,
        role=MessageRole.user,
        content=question
    )
    # mock _tool_call_notification to skip DB writes + WhatsApp sends for tool notifications
    with patch.object(llm_client, '_tool_call_notification', new=AsyncMock(return_value=None)):
        result = await llm_client.generate_response(mock_user, msg)

    if not result:
        return None

    for m in reversed(result):
        if m.role == MessageRole.assistant and m.content:
            return m.content
    return None



async def main():
    CSV_PATH_input = '/home/athul/my_workspace/twiga_project/twiga_evaluation/testset_progress.csv'
    CSV_PATH_output = '/home/athul/my_workspace/twiga_project/twiga_evaluation/testset_with_answers_qwen3.5_397B_A17B.csv'

    import os
    read_path = CSV_PATH_output if os.path.exists(CSV_PATH_output) else CSV_PATH_input
    df = pd.read_csv(read_path)

    if 'actual_output' not in df.columns:
        df['actual_output'] = None

    for i in range(len(df)):
        if pd.notna(df.loc[i, 'actual_output']):
            continue  # resume checkpoint

        question = df.loc[i, 'user_input']
        print(f"[{i}/{len(df)}] {question[:70]}...")

        answer = await generate_answer(question)
        df.loc[i, 'actual_output'] = answer
        df.to_csv(CSV_PATH_output, index=False)  # save after every row

        print(f"  -> {str(answer)[:100]}\n")


if __name__ == "__main__":
    asyncio.run(main())
