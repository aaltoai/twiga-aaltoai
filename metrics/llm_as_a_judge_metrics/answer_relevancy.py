import pandas as pd
from ragas.metrics.collections import AnswerRelevancy as _RagasAnswerRelevancy

from .base import LLMAsAJudgeMetric


class AnswerRelevancy(LLMAsAJudgeMetric):
    """Does the response actually address the question?

    Uses the LLM to generate candidate questions from the response, embeds
    them alongside the original question, and reports cosine similarity.
    Catches off-topic / partially-on-topic answers that lexical similarity
    metrics like BERTScore may rate highly because of vocabulary overlap.
    """

    name = "answer_relevancy"
    required_columns = ["user_input", "twiga_answer_cleaned_by_regex"]
    requires_embeddings = True

    def __init__(self, llm, embeddings=None):
        super().__init__(llm, embeddings)
        self._impl = _RagasAnswerRelevancy(llm=llm, embeddings=embeddings)

    async def ascore(self, row: pd.Series) -> float:
        result = await self._impl.ascore(
            user_input=str(row["user_input"]),
            response=str(row["twiga_answer_cleaned_by_regex"]),
        )
        return float(result.value)
