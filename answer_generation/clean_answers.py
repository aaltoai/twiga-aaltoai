"""CSV runner: apply regex_clean and llm_clean to every twiga_answer in the
results CSV. Resume-safe per cleaner.

Pure cleaning logic lives in cleaners.py — call regex_clean / llm_clean
directly from notebooks, tests, or pipelines if you don't need CSV plumbing.
"""

import time
from pathlib import Path
import pandas as pd

from cleaners import regex_clean, llm_clean

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "current"
CSV_PATH = str(DATA_DIR / "testset_with_twiga_answers.csv")


def main():
    df = pd.read_csv(CSV_PATH)

    # Migrate legacy column name if present
    if "twiga_answer_cleaned" in df.columns and "twiga_answer_cleaned_by_llm" not in df.columns:
        df = df.rename(columns={"twiga_answer_cleaned": "twiga_answer_cleaned_by_llm"})

    if "twiga_answer_cleaned_by_llm" not in df.columns:
        df["twiga_answer_cleaned_by_llm"] = pd.NA
    if "twiga_answer_cleaned_by_regex" not in df.columns:
        df["twiga_answer_cleaned_by_regex"] = pd.NA

    # Enforce column order: LLM-cleaned before regex-cleaned, both at the end
    base_cols = [c for c in df.columns if c not in ("twiga_answer_cleaned_by_llm", "twiga_answer_cleaned_by_regex")]
    df = df[base_cols + ["twiga_answer_cleaned_by_llm", "twiga_answer_cleaned_by_regex"]]

    print(f"Cleaning answers in {CSV_PATH}...")
    for i in range(len(df)):
        if pd.isna(df.loc[i, "twiga_answer"]):
            continue

        raw = str(df.loc[i, "twiga_answer"])
        question = str(df.loc[i, "user_input"])

        # Regex cleaner is cheap — always run if missing
        if pd.isna(df.loc[i, "twiga_answer_cleaned_by_regex"]):
            df.loc[i, "twiga_answer_cleaned_by_regex"] = regex_clean(raw)

        # LLM cleaner — skip if already done
        if pd.isna(df.loc[i, "twiga_answer_cleaned_by_llm"]):
            print(f"[{i + 1}/{len(df)}] {question[:70]}...")
            try:
                df.loc[i, "twiga_answer_cleaned_by_llm"] = llm_clean(question, raw)
            except Exception as e:
                print(f"  Failed at row {i}: {e}")
            time.sleep(1)

        df.to_csv(CSV_PATH, index=False)

    print(f"Done! Updated {CSV_PATH}")


if __name__ == "__main__":
    main()
