import pandas as pd
from ragas.metrics.collections import FactualCorrectness as _RagasFactualCorrectness

from .base import LLMAsAJudgeMetric


class FactualCorrectness(LLMAsAJudgeMetric):
    """Claim-level metric between generated answer and reference answer.

    Decomposes both into atomic claims, then computes precision / recall /
    F1 of overlap. Mode controls what's penalized:
      - "f1":        extra claims AND missing claims (default; strict on
                     elaborated answers since references are short).
      - "recall":    only missing claims (lenient on elaboration; closer in
                     spirit to custom_correctness).
      - "precision": only extra claims.

    Column name is set per mode (e.g. factual_correctness_f1) so multiple
    modes can be scored side-by-side in the same run.
    """

    required_columns = [
        "reference_answer",
        "twiga_answer_cleaned_by_regex",
    ]

    def __init__(self, llm, mode: str = "f1", embeddings=None):
        super().__init__(llm, embeddings)
        if mode not in {"f1", "recall", "precision"}:
            raise ValueError(f"mode must be one of f1/recall/precision, got {mode!r}")
        self.name = f"factual_correctness_{mode}"
        self._impl = _RagasFactualCorrectness(llm=llm, mode=mode)

    async def ascore(self, row: pd.Series) -> float:
        ref = str(row["reference_answer"]).strip()
        if not ref or ref.lower() == "nan":
            raise ValueError("reference_answer is empty")

        result = await self._impl.ascore(
            response=str(row["twiga_answer_cleaned_by_regex"]),
            reference=ref,
        )
        return float(result.value)
