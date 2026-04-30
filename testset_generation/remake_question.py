import pandas as pd
from openai import OpenAI
import time
import os
from pathlib import Path

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "current"
INPUT_CSV = str(DATA_DIR / "testset_progress.csv")
OUTPUT_CSV = str(DATA_DIR / "testset_rewritten_questions.csv")
MODEL = "deepseek-v4-pro"


def rewrite_question(original_question, context):
    prompt = f"""
    You are an expert prompt engineer and Tanzanian educational assistant.
    I have a question that was generated for a RAG evaluation, but it contains inappropriate personas (like 'Australian farmer' or 'agribusiness manager').

    Please rewrite this question so it sounds like it is being asked by a Tanzanian secondary school Geography teacher preparing a lesson for their Form 2 students.
    Remove any foreign personas or complex business scenarios. Keep the core geographical or agricultural subject matter exactly the same so that the provided context still answers it.

    Original Question: {original_question}
    Reference Context: {context}

    Output ONLY the rewritten question text. Do not include any conversational filler, quotes, or explanations.
    """

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=16384,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        print(f"    [rewrite_question] finish_reason={choice.finish_reason!r}")
        print(f"    [rewrite_question] message={choice.message!r}")
        print(f"    [rewrite_question] usage={response.usage!r}")
        raise ValueError("Empty response from question rewrite")
    return content.strip()


def generate_answer(question, context):
    prompt = f"""
    You are a Tanzanian secondary school Geography teacher answering a question for your Form 2 students.
    Answer the question accurately and concisely using ONLY the information in the reference context.
    Do not introduce facts that are not in the context. Keep the answer clear and at a Form 2 reading level.

    Question: {question}
    Reference Context: {context}

    Output ONLY the answer text. No preamble, no quotes, no explanation of your reasoning.
    """

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=16384,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        print(f"    [generate_answer] finish_reason={choice.finish_reason!r}")
        print(f"    [generate_answer] message={choice.message!r}")
        print(f"    [generate_answer] usage={response.usage!r}")
        raise ValueError("Empty response from answer generation")
    return content.strip()


def main():
    # Resume from existing output if present, otherwise start fresh from input
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Resuming from existing {OUTPUT_CSV}")
    else:
        df = pd.read_csv(INPUT_CSV)
        df["new_user_input"] = pd.NA
        df["new_reference"] = pd.NA
        print(f"Starting fresh from {INPUT_CSV}")

    print("Rewriting questions and regenerating answers...")
    for i, (index, row) in enumerate(df.iterrows(), start=1):
        # Skip rows already processed (resume support)
        if pd.notna(row.get("new_user_input")) and pd.notna(row.get("new_reference")):
            continue

        print(f"Processing {i}/{len(df)}...")
        try:
            new_q = rewrite_question(row["user_input"], row["reference_contexts"])
            new_a = generate_answer(new_q, row["reference_contexts"])
            df.at[index, "new_user_input"] = new_q
            df.at[index, "new_reference"] = new_a
        except Exception as e:
            print(f"  Failed at row {index}: {e}")
            # Leave the row's new_* columns as NA so a later run picks it up

        # Incremental save after every row so a crash doesn't lose progress
        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(1)

    print(f"Done! Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
