# PAJAMA Snorkel Label Model Pipeline

This folder runs the **weak-supervision aggregation** step of PAJAMA. It takes a pool of programmatic judges (`judge_*.py`, each scoring a single `(query, response)` pair) and combines them with [Snorkel](https://snorkel.ai/) into a single preference predictor for pairwise comparisons.

Upstream: judge programs are generated under [`~/pajama/judge_programs`](../judge_programs/README.md) (typically 80 programs per dataset).

The main entry point is:

```
snorkel_pipeline.py
```

---

## Quick start

### 1. Install dependencies

```bash
pip install numpy tqdm scikit-learn snorkel datasets huggingface_hub
```

### 2. Hugging Face access (for PAJAMA splits)

```bash
hf auth login
# or: export HF_TOKEN="hf_..."
```

Optional: override the dataset repo with `export PAJAMA_HF_REPO="your-org/PAJAMA"`.

### 3. Ensure judge programs exist

By default the pipeline loads judges from:

```
~/pajama/judge_programs/judge_programs_<dataset>/
```

Generate them first if needed (see [judge_programs README](../judge_programs/README.md)):

```bash
cd ~/pajama/judge_programs/judge_programs_scripts
python generate_judging_programs.py --dataset judgelm
```

### 4. Run the pipeline for one dataset

```bash
cd ~/pajama/snorkel_label_model_pipeline

python snorkel_pipeline.py --dataset judgelm
python snorkel_pipeline.py --dataset pandalm
python snorkel_pipeline.py --dataset multipref
python snorkel_pipeline.py --dataset prometheus
python snorkel_pipeline.py --dataset preference_700K
```

### 5. CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | One of: `judgelm`, `pandalm`, `multipref`, `prometheus`, `preference_700K` |
| `--repo-id` | `sprocket-lab/PAJAMA` | Hugging Face dataset repo |
| `--val` | HF `validation` split | Optional local JSONL override (unified PAJAMA schema) |
| `--test` | HF `test` split | Optional local JSONL override |
| `--judges` | `~/pajama/judge_programs/judge_programs_<dataset>/` | Directory of `judge_*.py` files |
| `--output` | `pipeline_outputs_<dataset>/` | Output directory (relative paths resolve under this folder) |
| `--tag` | same as `--dataset` | Prefix for summary / model filenames |
| `--seed` | `42` | NumPy random seed |
| `--min-acc` | `0.50` | Minimum per-program validation accuracy to keep a judge |

Examples:

```bash
# Custom judge pool and output directory
python snorkel_pipeline.py --dataset judgelm \
    --judges /path/to/my_judges \
    --output my_judgelm_run \
    --tag judgelm

# Local val/test only (no HF download)
python snorkel_pipeline.py --dataset pandalm \
    --val /path/to/val.jsonl \
    --test /path/to/test.jsonl

# Stricter filtering (fewer programs survive)
python snorkel_pipeline.py --dataset multipref --min-acc 0.55
```

---

## Supported datasets

Val and test splits are loaded from [sprocket-lab/PAJAMA](https://huggingface.co/datasets/sprocket-lab/PAJAMA) unless `--val` / `--test` are provided.

Each row uses the unified PAJAMA schema:

- `query`, `response1`, `response2`, `verdict` (`1` = response1 preferred, `2` = response2 preferred)

| `--dataset` | HF config | Typical val / test size | Default judges dir | Default output |
|-------------|-----------|-------------------------|--------------------|----------------|
| `judgelm` | `judgelm` | 500 / 5000 | `judge_programs/judge_programs_judgelm/` | `pipeline_outputs_judgelm/` |
| `pandalm` | `pandalm` | 500 / 894 | `judge_programs/judge_programs_pandalm/` | `pipeline_outputs_pandalm/` |
| `multipref` | `multipref` | 170 / 1700 | `judge_programs/judge_programs_multipref/` | `pipeline_outputs_multipref/` |
| `prometheus` | `prometheus` | 500 / 5000 | `judge_programs/judge_programs_prometheus/` | `pipeline_outputs_prometheus/` |
| `preference_700K` | `preference_700K` | 500 / 5000 | `judge_programs/judge_programs_preference_700K/` | `pipeline_outputs_preference_700K/` |

---

## What the pipeline does (high level)

Individual judge programs use **incompatible score scales** and are only weakly aligned with human preferences. This script:

1. Scores both responses with every loaded judge.
2. Normalizes scores per judge on **validation** only.
3. Converts score gaps into discrete votes with an **abstention band**.
4. Drops judges that do not beat a validation accuracy floor.
5. Selects the **top-k** subset that maximizes Snorkel agreement on validation.
6. Trains a Snorkel `LabelModel` on the resulting label matrix and evaluates on test.

Gold labels (`verdict`) are used only for tuning, filtering, top-k selection, and evaluation—not as inputs to each judge call.

---

## Snorkel pipeline (stage by stage)

### Stage 1 — Raw score collection

- Load validation and test data (HF or JSONL).
- Dynamically import every `judge_*.py` in `--judges`; each must define `judging_function(query, response) -> float` (higher = better quality).
- For each sample `i` and judge `j`, compute `s1[i,j]` and `s2[i,j]` on `response1` and `response2`. Failures become `NaN`.
- Cache to `val_s1.npy`, `val_s2.npy`, `test_s1.npy`, `test_s2.npy` (plus `.json` copies). Re-runs skip scoring if shapes match.

### Stage 2 — Per-program normalization

- Pool validation scores for judge `j` (both responses), take **P1 / P99** as robust min/max bounds.
- Min-max normalize to `[0, 1]`, then `diff = norm(s1) - norm(s2)` ∈ `[-1, 1]`.
- Apply the **same** bounds to test (no test-set fitting).

### Stage 3 — Per-program threshold tuning

- Grid search `t ∈ {0.00, 0.01, …, 0.14}` on the full validation set per judge:

| Condition | Label | Meaning |
|-----------|-------|---------|
| `diff > t` | `0` | response1 preferred |
| `diff < -t` | `1` | response2 preferred |
| otherwise | `-1` | abstain |

- Pick `t` with highest validation accuracy vs gold; tie-break toward **larger** `t` (more abstention).

Gold mapping: `verdict == 1` → label `0`, `verdict == 2` → label `1`.

### Stage 4 — Program filtering

- Keep judges with validation accuracy ≥ `--min-acc` (default `0.50`).
- Log dropped program names.

### Stage 5 — Label matrix and top-k selection

- Build `M_val` / `M_test`: rows = samples, columns = surviving judges, entries ∈ `{-1, 0, 1}`.
- **Stage 5b:** Rank surviving judges by validation accuracy. For `k = 1 … |surviving|`, take the top `k`, build `M`, and predict on validation:
  - `k < 3`: `MajorityLabelVoter`
  - `k ≥ 3`: quick `LabelModel` fit (200 epochs)
- Choose `k` with highest validation **agreement** (accuracy on non-abstain predictions).
- Rebuild `M` with only the selected top-k columns; report test label **conflict** rate via `LFAnalysis`.

### Stage 6 — Train LabelModel and evaluate

- Fit Snorkel `LabelModel` on `M_val` with `Y_dev = y_val` (500 epochs, `cardinality=2`).
- Predict on val/test. Rows where **all** selected judges abstain stay at `-1` and are **excluded** from agreement / F1 (coverage reflects this).
- Report **LabelModel** and **MajorityVote** baselines on test; also in-sample LabelModel metrics on val.
- Save hard predictions, soft probabilities (`Y_hat_test_soft`), and `{tag}_trained_label_model.pkl`.

### Stage 6b — Throughput

- Re-run top-k judge calls on the full test set and time LabelModel inference.
- Record wall-clock breakdown in `{tag}_pipeline_summary.json`.

### Default Snorkel / tuning hyperparameters

| Parameter | Value |
|-----------|-------|
| Threshold candidates | `0.00 … 0.14` step `0.01` |
| LabelModel epochs (final) | `500` |
| LabelModel epochs (top-k sweep) | `200` |
| `l2` | `0.01` |
| `lr` | `0.01` |
| LabelModel seed | `123` |
| Pipeline NumPy seed | `42` (overridable via `--seed`) |

---

## Output layout

After a run, artifacts live under `pipeline_outputs_<dataset>/` (or your `--output` path):

```
snorkel_label_model_pipeline/
├── snorkel_pipeline.py
├── README.md
└── pipeline_outputs_judgelm/          # example
    ├── judgelm_pipeline_summary.json    # primary report (read this first)
    ├── evaluation_metrics.json
    ├── abstained_test_number.json
    ├── judgelm_trained_label_model.pkl
    ├── val_s1.npy / val_s2.npy
    ├── test_s1.npy / test_s2.npy
    ├── val_diffs.npy / test_diffs.npy
    ├── best_thresholds.npy / best_accuracies.npy
    ├── M_val.npy / M_test.npy
    ├── Y_hat_val.npy / Y_hat_test_soft.npy
    └── *.json                             # JSON mirrors of the .npy arrays
```

### `{tag}_pipeline_summary.json`

Comparable to LLM-judge run summaries. Key fields:

| Field | Description |
|-------|-------------|
| `best_k` | Number of judges selected after the validation sweep |
| `selected_program_names` / `selected_program_ids` | Final judge subset |
| `selected_program_thresholds` | Per-judge abstention threshold `t` |
| `selected_program_val_accuracies` | Single-judge val accuracy before aggregation |
| `label_matrix_diagnostics` | Test conflict rate, coverage, all-abstain count |
| `LabelModel_test` / `MajorityVote_test` | Agreement, precision, recall, F1, coverage vs gold |
| `LabelModel_val` | Same metrics on validation (in-sample) |
| `throughput` | Top-k program time, label-model time, samples/sec |

### Other files

- **`evaluation_metrics.json`** — Stage 6 metrics only (`LabelModel_test`, `MajorityVote_test`, `LabelModel_val`).
- **`abstained_test_number.json`** — Indices of test samples where every selected judge abstained.
- **`Y_hat_test_soft.npy`** — Class-1 probability from `predict_proba` on covered test rows (`NaN` elsewhere).

---

## Score caching

If `val_s1.npy` and `val_s2.npy` already exist and match `(n_samples, n_judges)`, Stage 1 reloads them instead of re-executing judges. The same applies to test caches.

**Invalidate the cache** when you:

- Change the judge directory or add/remove `judge_*.py` files
- Switch datasets or splits

Either delete the cached `.npy` files or use a fresh `--output` directory. A shape mismatch triggers an automatic re-score with a console warning.

---

## Reading results

1. Open **`{tag}_pipeline_summary.json`** for the selected program set and test metrics.
2. Use **`evaluation_metrics.json`** for a minimal metrics-only view.
3. Load **`{tag}_trained_label_model.pkl`** with Snorkel to `predict` / `predict_proba` on new label matrices `M` built with the same thresholds and judge subset.

Agreement is **accuracy** of non-abstain predictions against gold `verdict` on covered samples. Coverage is the fraction of samples with at least one non-abstain vote (LabelModel) or a non-`-1` prediction after aggregation.

---

## Relationship to judge generation

| Step | Location | What happens |
|------|----------|--------------|
| Generate judges | `judge_programs/` | Claude writes 80 `judging_function` programs per dataset |
| Aggregate judges | **this folder** | Normalize, threshold, filter, top-k, Snorkel `LabelModel` |

Judge generation does **not** fix score scales, thresholds, or Snorkel weights—that is intentional. Those steps live here so all programs are calibrated on the same validation split before aggregation.

---

## What this script does **not** do

- Does not generate or modify judge programs (see `judge_programs/`)
- Does not train neural LLM judges
- Does not use training-split PAJAMA data (only validation + test by default)
- Does not force every test sample to receive a prediction; all-abstain rows remain uncovered

---

## End-to-end flow (PAJAMA)

```
judge_programs/  →  snorkel_label_model_pipeline/  →  downstream (e.g. program aggregation)
   80 weak LFs          LabelModel + metrics              uses pipeline_summary.json
```

For a single dataset, the typical command sequence is:

```bash
# 1. Generate judges (once per dataset)
cd ~/pajama/judge_programs/judge_programs_scripts
python generate_judging_programs.py --dataset judgelm

# 2. Aggregate with Snorkel
cd ~/pajama/snorkel_label_model_pipeline
python snorkel_pipeline.py --dataset judgelm
```
