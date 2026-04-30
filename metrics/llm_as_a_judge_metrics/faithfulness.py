import pandas as pd
from ragas.metrics.collections import Faithfulness as _RagasFaithfulness

from .base import LLMAsAJudgeMetric

# twiga_retrieved_context is stored as one string with chunks joined by this separator
# (see generate_answers.py).
CONTEXT_SEPARATOR = "\n---\n"


class Faithfulness(LLMAsAJudgeMetric):
    """Is the response grounded in the retrieved context?

    Extracts atomic claims from the response and checks each against the
    retrieved context — flags hallucinations beyond what was retrieved.
    """

    name = "faithfulness"
    required_columns = [
        "user_input",
        "twiga_answer_cleaned_by_regex",
        "twiga_retrieved_context",
    ]

    def __init__(self, llm, embeddings=None):
        super().__init__(llm, embeddings)
        self._impl = _RagasFaithfulness(llm=llm)

    async def ascore(self, row: pd.Series) -> float:
        contexts = [
            c for c in str(row["twiga_retrieved_context"]).split(CONTEXT_SEPARATOR)
            if c.strip()
        ]
        result = await self._impl.ascore(
            user_input=str(row["user_input"]),
            response=str(row["twiga_answer_cleaned_by_regex"]),
            retrieved_contexts=contexts,
        )
        return float(result.value)
