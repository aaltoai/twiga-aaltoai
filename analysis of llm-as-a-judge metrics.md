# Analysis of LLM-as-a-Judge Metrics

**Testset:** 71 rows. One row (#67) is a pipeline artifact (`[tool call: search_knowledge]` with no follow-up answer captured) and is excluded from the analysis below — **n = 70**.

**Judge model:** `gpt-4o-mini`, `temperature=0`.

## What each metric measures

| Metric | Compares | Catches |
|---|---|---|
| `faithfulness` | answer vs **retrieved context** | Hallucinations beyond what was retrieved |
| `answer_relevancy` | answer vs **question** | Off-topic / partially-on-topic answers |
| `custom_correctness` | answer vs **reference answer** | Factually wrong answers (lenient holistic LLM rubric) |
| `factual_correctness_f1` | answer vs **reference answer** | Claim-level F1 (ragas baseline; strict on elaboration) |
| `factual_correctness_recall` | answer vs **reference answer** | Claim-level recall (ragas baseline; lenient on elaboration) |

All metrics return scores in [0, 1]. The first three are the **LLM-as-judge contribution**; the two `factual_correctness` modes are included as **ragas baseline comparisons** — see the [comparison section](#comparison-with-ragas-factual_correctness) below for why we recommend `custom_correctness` over them as the primary correctness signal.

## Score summary (n = 70)

| Metric | Mean | Median | ≥ 0.75 | ≥ 0.50 |
|---|---:|---:|---:|---:|
| `faithfulness` | 0.809 | 0.889 | 52/70 (74.3%) | 63/70 (90.0%) |
| `answer_relevancy` | 0.800 | 0.830 | 50/70 (71.4%) | 68/70 (97.1%) |
| `custom_correctness` | 0.918 | 1.000 | 68/70 (97.1%) | 68/70 (97.1%) |
| `factual_correctness_f1` | 0.464 | 0.460 | 7/70 (10.0%) | 33/70 (47.1%) |
| `factual_correctness_recall` | 0.898 | 1.000 | 63/70 (90.0%) | 66/70 (94.3%) |

**Headline:** the typical Twiga answer is correct (median `custom_correctness` = 1.0, median `factual_correctness_recall` = 1.0) and on-topic (median relevancy = 0.83). The dramatically low `factual_correctness_f1` (mean 0.46) is **not** a sign Twiga is wrong — it's the F1 metric punishing Twiga's elaborated, structured answers against very short reference answers. See the comparison section.

## Low-score examples by metric

### Faithfulness

**Row 65** — `faithfulness = 0.00`, `correctness = 1.00`
- **Q:** *How do fluctuations in fossil fuel prices affect the costs of agricultural production?*
- **Reference:** *Fluctuations of prices of fossil fuel affect production.*
- **Twiga:** *"...the direct relationship between fossil fuel prices and agricultural costs is not explicitly covered in the Form 2 Agriculture chapter. However, I can provide you with curriculum [content]..."* (then provided general knowledge)
- **Why low:** retrieved context did not contain info on this topic, so Twiga generated answer beyond the retrieved chunks. Faithfulness correctly flagged it. Note correctness is 1.00 — the answer was still right; faithfulness is doing its job (flagging ungrounded generation), it just doesn't mean the answer is wrong.

**Row 54** — `faithfulness = 0.18`, `correctness = 0.75`
- **Q:** *What types of infrastructure support tourism at major attraction centers in Tanzania?*
- **Reference:** *Roads, railways, airports and hotels.*
- **Twiga:** *"...Roads connecting tourist sites (e.g., routes to Serengeti, Lake Manyara), Airports and airstrips..."* (correct answer with examples)
- **Why low:** the retrieved context was about **Namibia's** tourism infrastructure (visible in retrieved chunks), but Twiga answered about **Tanzania**. Faithfulness penalizes the mismatch between answer and retrieved context. The underlying issue is retrieval, not generation.

### Answer Relevancy

**Row 24** — `answer_relevancy = 0.00`, `faithfulness = 0.81`, `correctness = 0.00`
- **Q:** *How did the Tennessee Valley Authority promote economic development...*
- **Twiga:** *"The Tennessee Valley Authority (TVA) is a United States historical topic... not part of the Tanzanian Geography Form 2 curriculum..."* (refusal)
- **Why low:** Twiga correctly refused an out-of-curriculum question. Answer relevancy compares answer-to-question semantically, so a refusal does not "address" the question and scores 0. **Known limitation: relevancy penalizes correct refusals.** These rows should be tagged and analyzed separately.

**Row 46** — `answer_relevancy = 0.00`, `faithfulness = 0.94`, `correctness = 1.00`
- **Q:** *...I am using the Tennessee Valley Authority as a case study. I need to prepare notes on the Tennessee River Basin's extent...*
- **Twiga:** *"The Tennessee River Basin covers a seven-state area in the south-eastern United States... Tennessee, Kentucky, Virginia..."* (full correct answer)
- **Why low:** also TVA-related, but here Twiga **answered** instead of refusing. The answer is correct (correctness = 1.0) but relevancy is 0 — likely a judge artifact from the off-curriculum framing. **This row also reveals an inconsistency on Twiga's side: it refused row 24 but answered row 46 on essentially the same off-curriculum topic.**

### Custom Correctness

**Row 24** — `custom_correctness = 0.00`
- Same row as above. Twiga refused; the reference answer is a detailed TVA explanation; the rubric judged "no key information from reference is present" → score = 1 (= 0.0 normalized). **Same refusal issue: correctness penalizes refusals when the reference is non-empty.**

**Row 57** — `custom_correctness = 0.25`, `faithfulness = 0.21`, `relevancy = 0.65`
- **Q:** *...which countries are the major generators of nuclear power, including whether Israel is among them, and what are the key advantages and disadvantages of nuclear energy?*
- **Reference:** *The major generators of nuclear power are the USA, France, Russia, China, India, Canada, Israel and the UK... [advantages and disadvantages listed]*
- **Twiga:** *"...the textbook mentions: 'Nuclear energy is listed [as one of the energy sources]'..."* (hedged, did not list countries or advantages/disadvantages)
- **Why low:** the question maps to real textbook content, but Twiga gave a thin, hedged response and missed the listed countries (including the explicit "Is Israel among them?" sub-question) and the advantages/disadvantages. Genuine answer-quality failure — not a metric artifact.

## Why all three metrics are needed

### Correlation across metrics (n = 70)

| Pair | Pearson r | Reading |
|---|---:|---|
| `faithfulness` ↔ `answer_relevancy` | -0.03 | Independent — measure different things |
| `faithfulness` ↔ `custom_correctness` | 0.17 | Weakly related — low faithfulness ≠ wrong answer |
| `answer_relevancy` ↔ `custom_correctness` | 0.38 | Moderate — being on-topic correlates with being right |

The three core metrics aren't redundant. Faithfulness in particular catches a failure mode the other two miss (ungrounded generation), while correctness catches a failure mode faithfulness misses (right answer but extends beyond retrieved context).

### Cross-metric disagreements that justify the tripod

The strongest evidence that all three are needed is rows where the metrics actively disagree with each other.

**Pattern 1 — Low faithfulness, high correctness: Twiga's answer is right but elaborates beyond context.**

**Row 22** — `faithfulness = 0.27`, `relevancy = 0.89`, `correctness = 1.00`
- **Q:** *What is fertilisation in crop production?*
- **Reference:** *Fertilisation is done to add nutrients needed by plants in the soil.*
- **Twiga:** *"Fertilisation is done to add nutrients needed by plants in the soil. This is a key practice in crop production... — Replenishes nutrients that crops remove from the soil during growth..."*
- Twiga **literally repeats the reference verbatim** and adds correct supporting context. Faithfulness penalizes the elaboration; correctness (correctly) scores it 1.0. **If we only had faithfulness, this row would look like a failure.**

**Row 65** (covered above) — same pattern: `faithfulness = 0.00`, `correctness = 1.00`. Twiga correctly noted the textbook gap, then gave a correct broader answer. Faithfulness alone would call this a hallucination.

**Pattern 2 — Low relevancy + low correctness, high faithfulness: refusal of out-of-scope question.**

**Row 24** (covered above) — `faithfulness = 0.81`, `relevancy = 0.00`, `correctness = 0.00`. Twiga correctly refused. Without the three-way comparison, we couldn't distinguish "valid refusal" from "wrong answer" — both look the same on relevancy and correctness alone. The high faithfulness gives the signal that the answer is internally coherent, which combined with the refusal pattern lets us tag this as a refusal case for separate analysis.

**Pattern 3 — High faithfulness + high correctness, low relevancy: answered off-curriculum.**

**Row 46** (covered above) — `faithfulness = 0.94`, `relevancy = 0.00`, `correctness = 1.00`. Twiga answered an off-curriculum question correctly. The split between high correctness and zero relevancy flags an inconsistency — Twiga should probably have refused like in row 24. **Only the combination reveals this is a behavioral inconsistency rather than a generation failure.**

### Bottom line

- `faithfulness` alone overstates failures (penalizes correct elaboration)
- `answer_relevancy` alone overstates failures (penalizes correct refusals)
- `custom_correctness` alone has low resolution (effectively binary in practice — only 4 of 5 rubric levels were used)
- Together they triangulate: each row's three scores tell you not just *whether* it failed but *what kind* of failure or non-failure it is

## Comparison with ragas `factual_correctness`

We also ran `ragas.metrics.collections.FactualCorrectness` in two modes — F1 and recall — to compare the strict claim-decomposition approach against our lenient holistic `custom_correctness`.

### Score comparison

| Metric | Mean | Median | ≥ 0.75 | ≥ 0.50 | Resolution |
|---|---:|---:|---:|---:|:---:|
| `custom_correctness` | 0.918 | 1.000 | 97.1% | 97.1% | 4 unique values |
| `factual_correctness_recall` | 0.898 | 1.000 | 90.0% | 94.3% | 9 unique values |
| `factual_correctness_f1` | **0.464** | **0.460** | **10.0%** | **47.1%** | 30+ unique values |

### F1 collapses on this dataset — and that's not a bug, it's measuring something else

The 0.43-point gap between F1 (mean 0.464) and recall (mean 0.898) is fully explained by the asymmetry between Twiga's elaborated answers and the short reference answers:

- Reference answers: typically 1–2 sentences
- Twiga answers: ~1350 chars on average, structured with explanations and examples
- Twiga **covers** the reference (recall = 1.0 on most rows)
- Twiga **adds** lots of extra correct claims (precision tanks → drags F1 down)
- 25/70 rows have `recall − f1 > 0.5`
- 0 rows have `f1 > recall` — F1 is bounded above by recall in this dataset

**F1 is measuring answer parsimony (how concise the answer is), not correctness.** For an educational chatbot where pedagogical elaboration is a feature, F1 is the wrong objective. This explains why earlier `factual_correctness` runs gave bad scores — the metric was working correctly but answering a different question than the user thought it was.

#### Example of the elaboration penalty

**Row 51** — `factual_correctness_f1 = 0.14`, `factual_correctness_recall = 1.00`, `custom_correctness = 1.00`
- **Q:** *What is the contribution of livestock industries to Australia's agricultural sector?*
- **Reference:** *About 45% of the gross value of annual production.*
- **Twiga:** *"Livestock Industries in Australia's Agricultural Sector — Based on the TIE Geography textbook (Form 2, Chapter Two: Agriculture)... [structured multi-paragraph explanation]..."*
- F1 collapsed because of the extra elaborative content. Recall and `custom_correctness` correctly recognized that Twiga did cover the reference fact.

### Recall and `custom_correctness` agree in spirit, differ in resolution

Both metrics ask "did Twiga cover the reference's key facts?" Their Pearson correlation is **0.42** (moderate). The difference is resolution:

- `factual_correctness_recall` produces 9 unique scores via claim-level overlap
- `custom_correctness` collapses to 4 buckets via holistic LLM judgment

For headline reporting this is fine; for ranking similar-quality answers, recall is more discriminating.

### Pearson correlations including the comparison metrics (n = 70)

| Pair | Pearson r |
|---|---:|
| `factual_correctness_f1` ↔ `factual_correctness_recall` | 0.56 |
| `factual_correctness_f1` ↔ `custom_correctness` | 0.32 |
| `factual_correctness_recall` ↔ `custom_correctness` | 0.42 |
| `factual_correctness_f1` ↔ `faithfulness` | 0.12 |
| `factual_correctness_recall` ↔ `faithfulness` | 0.17 |
| `factual_correctness_f1` ↔ `answer_relevancy` | 0.08 |
| `factual_correctness_recall` ↔ `answer_relevancy` | 0.31 |

`factual_correctness_f1` is nearly independent of all the other metrics — further evidence that it's measuring a different construct (parsimony, not correctness).

### When to use which correctness metric

| Metric | Use when |
|---|---|
| `custom_correctness` | You want a single, interpretable correctness signal with judge justifications. Lenient by design. **Recommended primary.** |
| `factual_correctness_recall` | You want a continuous claim-level recall signal with finer resolution than `custom_correctness`. Useful as a secondary check. |
| `factual_correctness_f1` | **Don't use as primary correctness here.** It penalizes verbosity, which is a feature of educational answers. Useful only if you specifically want a parsimony signal. |

## Known limitations (for follow-up)

1. **Refusals (rows 24, 57)** are penalized by relevancy and correctness. Tag these rows and report metrics with/without them.
2. **`custom_correctness` is low-resolution** — judge skipped the score = 3 (mid) bucket entirely. Calibrate with examples in the rubric prompt if finer resolution is needed.
3. **Row 67 was a pipeline bug** in answer generation, not a Twiga quality issue. Add output validation before evaluation.
4. **n = 70 has wide confidence intervals** (~±0.03 SE on the means). Don't over-interpret small differences.
