## Evaluation for the TWIGA Project

A Streamlit dashboard for evaluating Twiga AI's response quality using ROUGE and BLEU metrics with multi-reference support.

### Features

- **Run Evaluation** — Select a test set, run Twiga's LLM, and view per-question ROUGE-1/2/L and BLEU scores
- **Manage Testsets** — Upload, validate, and browse CSV test sets with structured column requirements
- **Performance Dashboard** — Track metrics over time across different models and test sets with interactive Plotly charts
- **Multi-reference scoring** — ROUGE uses max-across-references; BLEU natively supports multiple references
- **Persistent results** — Evaluation results are saved as CSV + metadata files in the repository

### Testset Format

Your CSV **must** have:

| Column | Required | Description |
|--------|----------|-------------|
| `user_query` | Yes (exactly 1) | The question to send to the LLM |
| `reference` | Yes (at least 1) | A reference answer for scoring |
| `reference_1`, `reference_2`, ... | Optional | Additional reference answers (multi-reference ROUGE/BLEU) |

Any other columns (e.g. `query_style`, `persona_name`) are preserved but not used for scoring.

### Setup

1. **Install Twiga as an editable package** (from this directory):

   ```bash
   pip install -e ../twiga
   ```

2. **Install dashboard dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**: Copy `.env.example` to `../twiga/.env` (or wherever Twiga expects it) and fill in the required values. The LLM model and embedding model are determined by this `.env` file.

4. **Run the dashboard**:

   ```bash
   streamlit run app.py
   ```

### Project Structure

```
twiga-aaltoai/
├── app.py                        # Streamlit entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── pages/
│   ├── 1_Run_Evaluation.py       # Run eval and view results
│   ├── 2_Manage_Testsets.py      # Upload/validate test sets
│   └── 3_Performance.py          # Aggregated metrics charts
├── eval/
│   ├── answer_generator.py       # Twiga llm_client wrapper
│   ├── metrics.py                # ROUGE/BLEU computation
│   ├── testset.py                # Testset validation & I/O
│   └── results.py                # Result save/load utilities
└── data/
    ├── testsets/                  # Saved test set CSVs
    │   └── default.csv           # Bundled default (72 rows)
    └── results/                  # One subdirectory per eval run
        └── {timestamp}_{model}/
            ├── results.csv       # Aggregate scores
            └── metadata.txt      # Model/embedding/datetime info
```

### How Results Are Stored

Each evaluation run creates a subdirectory in `data/results/`:
- **`results.csv`** — rows are test set names, columns are metric scores (rouge1_f, rouge2_f, rougeL_f, bleu)
- **`metadata.txt`** — plain text with model name, embedding name, provider, evaluation datetime, test set name, and number of questions

The Performance page reads all result directories to build time-series charts.

### Saving and Deleting Results

- By default, results are **auto-saved** after each evaluation run
- Uncheck the "Save results" toggle on the Run Evaluation page to **opt out** — if results were already saved, they will be deleted