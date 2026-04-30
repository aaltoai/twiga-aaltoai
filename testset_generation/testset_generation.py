import json
import os
import random
from pathlib import Path

import pandas as pd
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "current"

# --- Load and sample chunks ---
with open(DATA_DIR / "chunks_export.jsonl") as f:
    all_chunks = [json.loads(line) for line in f]

random.seed(42)
sampled = random.sample(all_chunks, 100)

# --- Set up ragas generator ---
llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6"))

embeddings = OpenAIEmbeddings(client=OpenAI())

generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

# --- Generate testset with incremental saving ---
SAVE_PATH = str(DATA_DIR / "testset_progress_v1.csv")
results = []

if os.path.exists(SAVE_PATH):
    existing = pd.read_csv(SAVE_PATH)
    results.append(existing)
    completed_chunks = len(existing) // 2  # testset_size=2 per chunk
    print(f"Resuming from chunk {completed_chunks}, {len(existing)} samples already saved")
else:
    completed_chunks = 0
    print("Starting fresh")

for i, chunk in enumerate(sampled):
    if i < completed_chunks:
        continue

    try:
        dataset = generator.generate_with_chunks(
            chunks=[chunk["content"]],
            testset_size=2,
        )
        df = dataset.to_pandas()
        results.append(df)

        pd.concat(results, ignore_index=True).to_csv(SAVE_PATH, index=False)
        print(f"✓ Chunk {i+1}/{len(sampled)} saved ({len(df)} samples)")

    except Exception as e:
        print(f"✗ Chunk {i+1}/{len(sampled)} failed: {e}")
        continue

final = pd.concat(results, ignore_index=True)
print(f"Done: {len(final)} total samples from {len(sampled)} chunks")
