import json
import pandas as pd
from openai import AsyncOpenAI

from .base import LLMAsAJudgeMetric


SYSTEM_PROMPT = (
    "You are evaluating the factual correctness of a generated answer against "
    "a reference answer. Judge whether the key information is correct and "
    "present. Be lenient on wording, structure, and style — match meaning, "
    "not surface form."
)

USER_PROMPT_TEMPLATE = """Question:
{user_input}

Reference answer (ground truth):
{reference_answer}

Generated answer (to evaluate):
{generated_answer}

Score how well the generated answer covers the key information in the reference answer, on a 1-5 scale:

5 - Fully correct: all key information present, factually consistent. Extra correct information is fine.
4 - Mostly correct: most key information present; minor gaps or small imprecisions.
3 - Partially correct: some key information present, with notable gaps or minor factual issues.
2 - Largely incorrect: significant gaps or factual errors; misses the main point.
1 - Fundamentally wrong: contradicts the reference, addresses a different question, or is empty.

Rules:
- Paraphrasing and reorganization do NOT lower the score.
- Extra correct information does NOT lower the score.
- Style, formatting, and length differences do NOT lower the score.
- Score on factual coverage, not stylistic similarity.

Output JSON only:
{"score": <integer 1-5>, "justification": "<one sentence>"}"""


class CustomCorrectness(LLMAsAJudgeMetric):
    """Lenient holistic correctness against the reference answer.

    Alternative to ragas's claim-decomposition AnswerCorrectness: judges
    whether the generated answer covers the key information in the reference,
    tolerating paraphrase, extra correct info, and stylistic differences.
    Returns a 1-5 rubric score normalized to [0, 1].
    """

    name = "custom_correctness"
    required_columns = [
        "user_input",
        "reference_answer",
        "twiga_answer_cleaned_by_regex",
    ]

    def __init__(self, llm, client: AsyncOpenAI, model: str = "gpt-4o-mini", embeddings=None):
        super().__init__(llm, embeddings)
        self._client = client
        self._model = model

    async def ascore(self, row: pd.Series) -> float:
        ref = str(row["reference_answer"]).strip()
        if not ref or ref.lower() == "nan":
            raise ValueError("reference_answer is empty")

        user_msg = (
            USER_PROMPT_TEMPLATE
            .replace("{user_input}", str(row["user_input"]))
            .replace("{reference_answer}", ref)
            .replace("{generated_answer}", str(row["twiga_answer_cleaned_by_regex"]))
        )

        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        parsed = json.loads(completion.choices[0].message.content)
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score out of range: {score}")

        if score < 4:
            justification = str(parsed.get("justification", ""))[:240]
            print(f"      [justification] {justification}")

        return (score - 1) / 4.0
