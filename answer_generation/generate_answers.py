"""CSV runner: read questions from a testset CSV, call the Twiga agent on each,
write results back to a results CSV. Resume-safe.

Pure agent-call logic lives in twiga_runner.py — call run_twiga(question)
directly from notebooks, tests, or pipelines if you don't need the CSV plumbing.
"""

import asyncio
import os
from pathlib import Path
import pandas as pd

from twiga_runner import run_twiga

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "current"
INPUT_CSV = str(DATA_DIR / "testset_rewritten_questions.csv")
OUTPUT_CSV = str(DATA_DIR / "testset_with_twiga_answers.csv")

OUTPUT_COLS = [
    "user_input",
    "reference_context",
    "reference_answer",
    "twiga_answer",
    "twiga_retrieved_context",
    "twiga_tool_call",
    "tool_call_output_returned",
]


async def main():
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Resuming from {OUTPUT_CSV}")
    else:
        df_in = pd.read_csv(INPUT_CSV)
        df = pd.DataFrame({
            "user_input": df_in["new_user_input"],
            "reference_context": df_in["reference_contexts"],
            "reference_answer": df_in["new_reference"],
            "twiga_answer": pd.NA,
            "twiga_retrieved_context": pd.NA,
            "twiga_tool_call": pd.NA,
            "tool_call_output_returned": pd.NA,
        })
        print(f"Starting fresh; output will be written to {OUTPUT_CSV}")

    df = df[OUTPUT_COLS]

    for i in range(len(df)):
        if pd.notna(df.loc[i, "twiga_answer"]):
            continue

        question = str(df.loc[i, "user_input"])
        print(f"[{i + 1}/{len(df)}] {question[:70]}...")

        result = await run_twiga(question)
        df.loc[i, "twiga_answer"] = result["twiga_answer"]
        df.loc[i, "twiga_retrieved_context"] = result["twiga_retrieved_context"]
        df.loc[i, "twiga_tool_call"] = result["twiga_tool_call"]
        df.loc[i, "tool_call_output_returned"] = result["tool_call_output_returned"]
        df.to_csv(OUTPUT_CSV, index=False)

        print(f"  answer -> {str(result['twiga_answer'])[:80]}")
        print(f"  tool   -> {result['twiga_tool_call']} (output_returned={result['tool_call_output_returned']})\n")


if __name__ == "__main__":
    asyncio.run(main())
