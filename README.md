# twiga-evaluation

Evaluation pipeline for the [Twiga](../twiga) Tanzanian classroom AI assistant. Generates a testset, runs Twiga to produce answers, and scores them with LLM-as-judge metrics (Faithfulness, AnswerRelevancy).

## Project structure

```
twiga_evaluation/
├── evaluation.py                              # entry point: scores answers with the configured metrics
├── metrics/
│   └── llm_as_a_judge_metrics/                # pluggable LLM-as-judge metric classes
├── testset_generation/                        # build the Q&A testset
├── answer_generation/                         # call Twiga on each question, post-process answers
├── data/
│   ├── current/                               # active pipeline inputs/outputs
│   └── old/                                   # legacy CSVs from earlier runs
└── notebooks/                                 # exploratory notebooks, not part of the active pipeline
```

## `testset_generation/`

Builds the question/answer testset Twiga is evaluated against.

- **`testset_generation.py`** — Generates Q&A pairs from textbook chunks using the Ragas `TestsetGenerator` with **Claude Sonnet** as the synthesis LLM and OpenAI embeddings.
- **`remake_question.py`** — Rewrites generated questions that ended up with inappropriate personas (e.g. *Australian farmer*, *agribusiness manager*) which Twiga's safety layer blocks. Uses **DeepSeek V4 Pro** to regenerate both the question and the reference answer. See `analysis.md` for the full motivation.

## `answer_generation/`

Runs Twiga on each testset question and post-processes the responses for evaluation.

- **`twiga_runner.py`** — Pure library. `run_twiga(question)` calls the Twiga agent and returns a dict with the answer, retrieved context, tool call name, and tool-call success flag. No file I/O.
- **`cleaners.py`** — Pure library. `regex_clean(raw)` strips decoration faithfully (markdown, emojis, greetings, "Teaching Tip" / CTA sections); `llm_clean(question, raw)` uses DeepSeek to do the same with model judgment.
- **`generate_answers.py`** — CSV runner. Reads `testset_rewritten_questions.csv`, calls `run_twiga` for each row, writes `testset_with_twiga_answers.csv`. Resume-safe.
- **`clean_answers.py`** — CSV runner. Adds `twiga_answer_cleaned_by_regex` and `twiga_answer_cleaned_by_llm` columns. Resume-safe per cleaner.

## `metrics/llm_as_a_judge_metrics/`

Pluggable metric classes built on top of Ragas. Adding a new metric = one new file.

- **`base.py`** — Abstract `LLMAsAJudgeMetric`. Subclasses declare `name`, `required_columns`, `requires_embeddings`, and an async `ascore(row) -> float`.
- **`faithfulness.py`** — Is the response grounded in the retrieved context? Catches hallucinations beyond what was retrieved.
- **`answer_relevancy.py`** — Does the response actually address the question? Catches off-topic answers that lexical metrics may miss.

## `evaluation.py`

Top-level runner. Loads `data/current/testset_with_twiga_answers.csv`, applies each metric to every row, writes scores to `data/current/rag_evaluation_results.csv`, prints a summary. Resume-safe per metric. Judge LLM: `gpt-4o-mini` (deliberately *not* DeepSeek to avoid self-preference bias against the DeepSeek-written reference answers — see `analysis.md`).

## How to run

```bash
# 1. Generate the testset (once, slow)
uv run testset_generation/testset_generation.py
uv run testset_generation/remake_question.py

# 2. Run Twiga to produce answers
uv run answer_generation/generate_answers.py
uv run answer_generation/clean_answers.py

# 3. Evaluate
uv run evaluation.py
```

Each stage reads/writes CSVs in `data/current/` and is independently resume-safe — kill any stage and re-run, it picks up where it left off.

## Division of labor with other evaluation work

This pipeline owns **generation-side** evaluation. Other components are evaluated by collaborators:

| Component | Owner | Metrics |
|---|---|---|
| Retrieval (ranking) | colleague | Recall@k, MRR, NDCG |
| Generation (semantic match) | colleague | BERTScore, Word Mover's Distance |
| **Generation (judgment)** | **this repo** | **AnswerRelevancy, Faithfulness** |

## Further reading

- `analysis.md` — methodology decisions, evaluation findings, and lessons learned (e.g. why the testset needed remaking, why the LLM cleaner is unreliable, why `gpt-4o-mini` for judging).
