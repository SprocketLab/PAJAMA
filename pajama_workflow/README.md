# Programmatic Judge Workflow

We use the pool of synthesized programmatic judges from `synthesized_programmatic_judges/` and model them with [Snorkel](https://snorkel.ai/) into an aggregated preference verdict.

## Quick start

```bash
python run.py --dataset judgelm
python run.py --dataset pandalm
python run.py --dataset multipref
python run.py --dataset prometheus
python run.py --dataset preference_700K
```

## How it works?

1. **Program Execution** — invoke every `judging_function(query, response)` on val + test; cache to `.npy`.
2. **Output Normalization** — per-program P1/P99 min-max on val scores; compute `diff = norm(s1) - norm(s2)`.
3. **Threshold Tuning** — grid search `t ∈ {0.00 … 0.14}` per program on val; then pick best accuracy.
4. **Program Filtering** — drop programs below `50%`.
5. **Top-k Program Selection** — rank survivors by val accuracy; sweep `k` and pick the subset with highest Snorkel agreement on val.
6. **Verdict Aggregation** — fit Snorkel `LabelModel` on `M_val`; evaluate on test. All-abstain rows stay uncovered.

## Outputs


| File                              | Description                                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `*_pipeline_summary.json`         | Primary report: selected programs, thresholds, test accuracy/F1/coverage, throughput                         |
| `evaluation_metrics.json`         | Metrics-only view: LabelModel and MajorityVote results on val and test                                       |
| `abstained_test_number.json`      | Indices of test samples where every selected judge abstained (no prediction made)                            |
| `*_trained_label_model.pkl`       | Trained Snorkel `LabelModel`; reload with Snorkel to run `predict` / `predict_proba` on new label matrices   |
| `val_s1/s2.npy`, `test_s1/s2.npy` | Raw scores from each judge on response1 and response2; cached to skip re-scoring on re-runs                  |
| `val_diffs.npy`, `test_diffs.npy` | Normalized score differences per judge: `norm(s1) - norm(s2)` ∈ `[-1, 1]`                                    |
| `M_val.npy`, `M_test.npy`         | Final label matrices for the selected top-k judges: entries ∈ `{-1, 0, 1}` (abstain / response1 / response2) |
| `Y_hat_val.npy`                   | Hard predictions on val (−1 for all-abstain rows)                                                            |
| `Y_hat_test_soft.npy`             | Class-1 probability from `predict_proba` on covered test rows (`NaN` for all-abstain rows)                   |


Score `.npy` caches are reused on re-runs if shapes match. Delete them (or use a new `--output`) after changing the judge pool.