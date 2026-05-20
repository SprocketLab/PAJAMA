# PAJAMA Judge Programs

This folder contains **programmatic judge functions** used in the PAJAMA pipeline. Each judge is a Python function that scores a single `(query, response)` pair. The Snorkel pipeline later combines 80 such programs into a weak-supervision label model.

The main entry point is:

```
judge_programs_scripts/generate_judging_programs.py
```

For each dataset, the script asks Claude to write **80 programs** = **10 heuristics × 8 variants**.

---

## Quick start

### 1. Install dependencies

```bash
pip install anthropic datasets huggingface_hub
```

### 2. Set API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# PAJAMA data lives on Hugging Face:
hf auth login
# or: export HF_TOKEN="hf_..."
```

### 3. Generate programs for one dataset

```bash
cd judge_programs/judge_programs_scripts

python generate_judging_programs.py --dataset judgelm
python generate_judging_programs.py --dataset pandalm
python generate_judging_programs.py --dataset multipref
python generate_judging_programs.py --dataset prometheus
python generate_judging_programs.py --dataset preference_700K
```

### 4. Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | One of: `judgelm`, `pandalm`, `multipref`, `prometheus`, `preference_700K` |
| `--repo-id` | `sprocket-lab/PAJAMA` | Hugging Face dataset repo |
| `--num-few-shot` | `10` | Number of validation examples shown in the prompt |
| `--seed` | `42` | Random seed for few-shot sampling |

Example:

```bash
python generate_judging_programs.py \
    --dataset multipref \
    --repo-id sprocket-lab/PAJAMA \
    --num-few-shot 10 \
    --seed 42
```

---

## Supported datasets

Few-shot examples are sampled from the **validation split** of [sprocket-lab/PAJAMA](https://huggingface.co/datasets/sprocket-lab/PAJAMA).

| `--dataset` | HF config | Val size | Few-shot shows scores? | Output directory |
|-------------|-----------|----------|------------------------|------------------|
| `judgelm` | `judgelm` | 500 | yes (`score1`, `score2`) | `judge_programs_judgelm/` |
| `pandalm` | `pandalm` | 500 | no (verdict only) | `judge_programs_pandalm/` |
| `multipref` | `multipref` | 170 | yes | `judge_programs_multipref/` |
| `prometheus` | `prometheus` | 500 | yes | `judge_programs_prometheus/` |
| `preference_700K` | `preference_700K` | 500 | yes | `judge_programs_preference_700K/` |

Each PAJAMA row uses a unified schema:

- `query`, `response1`, `response2`, `verdict` (`1` = response1 preferred, `2` = response2 preferred)
- `score1`, `score2` (when available)

---

## Output layout

After generation, files are written under `judge_programs_scripts/`:

```
judge_programs_scripts/
├── generate_judging_programs.py
├── judge_programs_judgelm/
│   ├── judge_1.py
│   ├── ...
│   └── judge_80.py
├── program_manifest_judgelm.json
├── judge_programs_pandalm/
│   └── ...
└── program_manifest_pandalm.json
```

### Program files

Each successful program defines:

```python
def judging_function(query, response):
    ...
    return score  # numeric; higher = better quality
```

### Manifest

`program_manifest_<dataset>.json` tracks every slot:

- `program_id` (1–80)
- `heuristic_id`, `heuristic_name`, `variant` (1–8)
- `status` (`success` or error message)
- `approach_summary` (keyword tags extracted from generated code)
- `dataset`, `hf_config`, `hf_repo`

### Resume

The script is **crash-safe**. Re-running the same command skips programs already marked `success` in the manifest and only fills missing slots.

---

## Generation pipeline (what the script does)

For each of the 80 program slots:

1. Build a Claude prompt (one heuristic + few-shot examples + optional “avoid these approaches”).
2. Call **Claude Opus 4.6** (`temperature=0.713`, `max_tokens=5000`).
3. Extract Python from the ` ```python ` block.
4. **Validate** the code (syntax, runs, non-constant output on fixed test cases).
5. **Deduplicate** against all existing programs (code similarity > 0.60 → reject and retry).
6. Save `judge_<id>.py` and update the manifest.

Default limits:

| Parameter | Value |
|-----------|-------|
| Model | `claude-opus-4-6` |
| Programs per heuristic | 8 |
| Max retries per slot | 5 |
| Max global API attempts | 200 |
| Similarity threshold | 0.60 |

---

## Prompt design (what we tell Claude)

Each API call sends **one user message** built by `build_prompt()`. Below is the logical structure.

### Role & task

> You are an expert Python developer and AI evaluation researcher.
>
> Write a Python function that evaluates the quality of an LLM-generated response to a given query. Return a numeric score where **higher = better quality**.

### Evaluation dimension (one of 10 heuristics)

The prompt injects **exactly one** heuristic per program:

```
EVALUATION STRATEGY — focus STRICTLY on this dimension:
<Heuristic Name>: <Heuristic Description>
```

See [Heuristics](#heuristics-10--8-variants--80-programs) below for the full list.

### Diversity constraint (variants 2–8 within the same heuristic)

If earlier variants for the same heuristic already succeeded, the prompt adds:

```
The following approaches have ALREADY been implemented for this heuristic.
You MUST use a substantially DIFFERENT algorithm:
  - Variant 1: <approach summary>
  - Variant 2: <approach summary>
  ...
```

Approach summaries are auto-extracted from prior code (e.g. “Flesch readability, syllable counting”, “Jaccard similarity, word overlap”).

### Few-shot examples (from PAJAMA validation)

The prompt includes **real pairwise examples** sampled from the target dataset’s HF validation split. Format:

**With scores** (judgelm, multipref, prometheus, preference_700K):

```
--- Example i ---
Query: <query truncated to 300 chars>

Response A (score=<score1>):
<response1 truncated to 400 chars>

Response B (score=<score2>):
<response2 truncated to 400 chars>

Ground-truth verdict: Response A is better (gap = |score1 - score2|)
```

**Without scores** (pandalm):

```
--- Example i ---
Query: ...

Response A:
...

Response B:
...

Ground-truth verdict: Response A is better
```

Purpose: show Claude what real queries/responses look like and what “good vs bad” means on this dataset—**without** asking Claude to copy a fixed scoring rubric or score range.

### Hard requirements (code contract)

Claude must follow these constraints:

| Requirement | Detail |
|-------------|--------|
| Output format | **Only** executable Python inside ` ```python ... ``` `; no prose |
| Function signature | `def judging_function(query, response):` |
| Return type | Single numeric score (`int` or `float`); higher = better |
| Allowed libraries | `re`, `math`, `collections`, `string`, `statistics` only — **no** heavy ML/NLP libs |
| Variant diversity | Variant `k/8` must use a **meaningfully different** algorithm than siblings |
| Robustness | `try/except`; must not crash on empty/short/long inputs |
| Score range | Reasonable range (e.g. 0–10 or 0–100); not enforced to a single scale |
| Discriminative | Must clearly separate high- vs low-quality responses |

The prompt ends with a stub:

```python
def judging_function(query, response):
    # Your implementation here
```

### What we do **not** tell Claude

- No fixed output score range tied to downstream normalization
- No per-program validation accuracy or coverage
- No Snorkel threshold or top-k selection details

Those are handled later in the Snorkel pipeline, not at generation time.

---

## Heuristics (10 × 8 variants = 80 programs)

| ID | Name | What the judge should measure |
|----|------|-------------------------------|
| 1 | Relevance to the Query | Semantic relevance, topic alignment, direct answer to intent; penalize tangents |
| 2 | Language Quality and Readability | Grammar, spelling, punctuation, readability (sentence length, TTR, Flesch-like) |
| 3 | Completeness and Coverage | Thoroughness, sub-questions, edge cases; penalize shallow answers |
| 4 | Factual Accuracy Indicators | Verifiable facts, citations, hedging; penalize hallucination red flags |
| 5 | Logical Coherence and Argument Structure | Flow, premises→conclusions, no contradictions or non-sequiturs |
| 6 | Clarity and Conciseness | Clear, efficient prose; penalize filler and repetition |
| 7 | Reasoning Transparency and Step-wise Formulation | Visible step-by-step reasoning; penalize opaque jumps to conclusions |
| 8 | Epistemic Calibration and Uncertainty Communication | Appropriate confidence/hedging; penalize false certainty |
| 9 | Structural Organization and Formatting | Lists, headers, paragraphs; penalize wall-of-text |
| 10 | Evidence Density and Specificity | Concrete examples, numbers, named entities; penalize vague hand-waving |

Program IDs map to heuristics in order:

- `judge_1`–`judge_8` → heuristic 1, variants 1–8  
- `judge_9`–`judge_16` → heuristic 2, variants 1–8  
- …  
- `judge_73`–`judge_80` → heuristic 10, variants 1–8  

---

## Post-generation checks (script-side, not in the prompt)

Before accepting a program, the script runs local checks **without** calling Claude again:

1. **Syntax & execution** — `compile` + `exec`, must define `judging_function`.
2. **Non-constant output** — run on 6 fixed `(query, response)` pairs; reject if all scores identical or range < 0.05.
3. **Code similarity** — `difflib.SequenceMatcher` on comment-stripped code; reject if similarity to any existing program > 0.60.

Failed slots retry up to 5 times; if still failing, a placeholder `judge_<id>.py` is written (`return len(response)`) and marked failed in the manifest.

