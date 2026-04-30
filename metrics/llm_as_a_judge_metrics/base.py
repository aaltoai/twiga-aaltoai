from abc import ABC, abstractmethod
import pandas as pd


class LLMAsAJudgeMetric(ABC):
    """Base class for LLM-as-judge evaluation metrics.

    Each subclass declares:
      - `name`: the column name written into the results CSV
      - `required_columns`: dataset columns the metric reads
      - `requires_embeddings`: whether an embeddings model is needed at construction

    Adding a new metric = one new subclass file. The runner picks it up
    via the metrics list in evaluation.py.
    """

    name: str
    required_columns: list[str]
    requires_embeddings: bool = False

    def __init__(self, llm, embeddings=None):
        if self.requires_embeddings and embeddings is None:
            raise ValueError(
                f"{self.name}: requires_embeddings=True but no embeddings were provided."
            )
        self.llm = llm
        self.embeddings = embeddings

    @abstractmethod
    async def ascore(self, row: pd.Series) -> float:
        """Score a single dataset row. Should raise on failure; the runner catches."""
        ...

    def validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.name}: missing required columns {missing}. "
                f"Available columns: {list(df.columns)}"
            )
