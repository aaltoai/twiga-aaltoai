"""Runner for LLM-as-judge evaluation metrics.

Loads a CSV with twiga answers + retrieved context, scores each row with the
configured metrics, and writes results back to a separate CSV.

Resume support: per-metric. If a metric's column already has a value for a row,
that row is skipped for that metric.
"""

import asyncio
import os
from pathlib import Path
import pandas as pd
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from metrics.llm_as_a_judge_metrics import Faithfulness, AnswerRelevancy, LLMAsAJudgeMetric

DATA_DIR = Path(__file__).resolve().parent / "data" / "current"
INPUT_CSV = str(DATA_DIR / "testset_with_twiga_answers.csv")
OUTPUT_CSV = str(DATA_DIR / "rag_evaluation_results.csv")

JUDGE_MODEL = "gpt-4o-mini"


def build_metrics() -> list[LLMAsAJudgeMetric]:
    client = AsyncOpenAI()
    llm = llm_factory(JUDGE_MODEL, client=client, max_tokens=8192)
    embeddings = OpenAIEmbeddings(client=client)
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]


async def evaluate(df: pd.DataFrame, metrics: list[LLMAsAJudgeMetric]) -> pd.DataFrame:
    for m in metrics:
        m.validate_columns(df)
        if m.name not in df.columns:
            df[m.name] = pd.NA

    total = len(df)
    for i in range(total):
        for m in metrics:
            if pd.notna(df.loc[i, m.name]):
                continue
            try:
                score = await m.ascore(df.iloc[i])
                df.loc[i, m.name] = score
                print(f"  [{i + 1}/{total}] {m.name} = {score:.4f}")
            except Exception as e:
                print(f"  [{i + 1}/{total}] {m.name} failed: {e}")

        df.to_csv(OUTPUT_CSV, index=False)

    return df


async def main():
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Resuming from {OUTPUT_CSV} ({len(df)} rows)")
    else:
        df = pd.read_csv(INPUT_CSV)
        print(f"Starting fresh from {INPUT_CSV} ({len(df)} rows)")

    metrics = build_metrics()
    print(f"Judge: {JUDGE_MODEL}; metrics: {[m.name for m in metrics]}\n")

    df = await evaluate(df, metrics)

    print("\n── Evaluation Summary ──────────────────────────")
    for m in metrics:
        valid = int(df[m.name].notna().sum())
        mean = pd.to_numeric(df[m.name], errors="coerce").mean()
        print(f"  {m.name:20s}: {mean:.4f}  ({valid}/{len(df)} rows scored)")
    print("────────────────────────────────────────────────")
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
