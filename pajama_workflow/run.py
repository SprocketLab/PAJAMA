"""
PAJAMA — Unified Snorkel Pipeline
==================================
Score collection → normalization → threshold tuning → program filtering
→ label matrix M → Snorkel LabelModel → evaluation.

Supports five datasets via --dataset:
    judgelm, pandalm, multipref, prometheus, preference_700K

Val/test splits are loaded from the PAJAMA Hugging Face dataset by default:
    https://huggingface.co/datasets/sprocket-lab/PAJAMA

Usage:
    hf auth login                           # if the HF repo is private

    python run.py --dataset judgelm
    python run.py --dataset pandalm
    python run.py --dataset multipref
    python run.py --dataset prometheus
    python run.py --dataset preference_700K

Optional: pass --val / --test to override with local JSONL (same unified schema).
"""

import os
import re
import json
import time
import argparse
import importlib.util
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'datasets'. Install with:\n"
        "  pip install datasets huggingface_hub"
    ) from exc

from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from snorkel.labeling import LFAnalysis

HF_REPO_ID = os.environ.get("PAJAMA_HF_REPO", "sprocket-lab/PAJAMA")

# ── Dataset configurations ─────────────────────────────────────────────────
# Paths are relative to SCRIPT_DIR (pajama_workflow/); _resolve() in main() makes them absolute.
DATASET_CONFIGS = {
    "judgelm": {
        "hf_config": "judgelm",
        "judges": "../synthesized_programmatic_judges/judgelm",
        "output": "judgelm_outputs",
        "tag": "judgelm",
    },
    "pandalm": {
        "hf_config": "pandalm",
        "judges": "../synthesized_programmatic_judges/pandalm",
        "output": "pandalm_outputs",
        "tag": "pandalm",
    },
    "multipref": {
        "hf_config": "multipref",
        "judges": "../synthesized_programmatic_judges/multipref",
        "output": "multipref_outputs",
        "tag": "multipref",
    },
    "prometheus": {
        "hf_config": "prometheus",
        "judges": "../synthesized_programmatic_judges/prometheus",
        "output": "prometheus_outputs",
        "tag": "prometheus",
    },
    "preference_700K": {
        "hf_config": "preference_700K",
        "judges": "../synthesized_programmatic_judges/preference_700K",
        "output": "preference_700K_outputs",
        "tag": "preference_700K",
    },
}


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="PAJAMA unified Snorkel pipeline (5 datasets, HF data by default)")
    p.add_argument("--dataset", required=True,
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Dataset to run the pipeline on")
    p.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face dataset repo (default: {HF_REPO_ID})",
    )
    p.add_argument(
        "--val", default=None,
        help="Optional local validation JSONL override (unified PAJAMA schema)",
    )
    p.add_argument(
        "--test", default=None,
        help="Optional local test JSONL override (unified PAJAMA schema)",
    )
    p.add_argument("--judges",  default=None,
                   help="Directory containing judge_*.py programs (default: per-dataset)")
    p.add_argument("--output",  default=None,
                   help="Output directory (default: per-dataset)")
    p.add_argument("--tag",     default=None,
                   help="Short tag for filenames (default: per-dataset)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--min-acc", type=float, default=0.50,
                   help="Min val accuracy to keep a program (default: 0.50)")
    args = p.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    if args.judges is None:
        args.judges = cfg["judges"]
    if args.output is None:
        args.output = cfg["output"]
    if args.tag is None:
        args.tag = cfg["tag"]

    return args


# ── Configuration (set in main from CLI args) ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
MIN_ACCURACY_THRESHOLD = 0.50
THRESHOLD_CANDIDATES = np.arange(0.0, 0.16, 0.01)

SNORKEL_CARDINALITY = 2
SNORKEL_EPOCHS = 500
SNORKEL_SEED = 123
SNORKEL_L2 = 0.01
SNORKEL_LR = 0.01

OUTPUT_DIR = None   # set in main()
TAG = None          # short identifier used in filenames
# ───────────────────────────────────────────────────────────────────────────


def _verdict_to_label(verdict) -> int:
    """Map PAJAMA verdict to Snorkel label: 0=response1 wins, 1=response2 wins."""
    if verdict in (1, "1"):
        return 0
    if verdict in (2, "2"):
        return 1
    raise ValueError(f"Unsupported verdict: {verdict!r}")


def _extract_fields(row):
    """Extract (query, response1, response2) from a PAJAMA row."""
    return (
        str(row.get("query", "")),
        str(row.get("response1", "")),
        str(row.get("response2", "")),
    )


def _rows_to_labels(rows) -> np.ndarray:
    return np.array([_verdict_to_label(row["verdict"]) for row in rows])


def load_hf_split(dataset_key: str, split: str, repo_id: str = HF_REPO_ID):
    """Load one split from the PAJAMA Hugging Face dataset."""
    cfg = DATASET_CONFIGS[dataset_key]
    print(f"  Loading HF: {repo_id}  config={cfg['hf_config']}  split={split}")
    ds = load_dataset(repo_id, cfg["hf_config"], split=split)
    rows = [dict(row) for row in ds]
    labels = _rows_to_labels(rows)
    return rows, labels


def load_jsonl_split(path: str):
    """Load a local JSONL file in the unified PAJAMA schema."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    labels = _rows_to_labels(rows)
    return rows, labels


def load_split(dataset_key: str, split: str, repo_id: str, local_path: str | None):
    """Load val/test from HF by default, or from a local JSONL override."""
    if local_path:
        print(f"  Loading local JSONL override: {local_path}")
        return load_jsonl_split(local_path)
    return load_hf_split(dataset_key, split, repo_id=repo_id)


def load_judges(judge_dir):
    judges, names = [], []
    files = sorted(f for f in os.listdir(judge_dir) if f.endswith(".py"))
    for f in files:
        path = os.path.join(judge_dir, f)
        spec = importlib.util.spec_from_file_location(f.replace(".py", ""), path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            judges.append(mod.judging_function)
            names.append(f.replace(".py", ""))
        except Exception as e:
            print(f"  ⚠ Failed to load {f}: {e}")
    return judges, names


# ── Score collection ──────────────────────────────────────────────────────

def collect_raw_scores(data, judges, desc="Scoring"):
    """Collect raw (s1, s2) per sample per judge — no normalization yet."""
    n, m = len(data), len(judges)
    s1_mat = np.full((n, m), np.nan)
    s2_mat = np.full((n, m), np.nan)
    for i, row in tqdm(enumerate(data), total=n, desc=desc):
        query, ans1, ans2 = _extract_fields(row)
        for j, fn in enumerate(judges):
            try:
                s1_mat[i, j] = float(fn(query, ans1))
                s2_mat[i, j] = float(fn(query, ans2))
            except Exception:
                pass
    return s1_mat, s2_mat


# ── Normalization ─────────────────────────────────────────────────────────

def compute_norm_params(s1, s2):
    """Per-program robust min-max bounds (P1/P99)."""
    m = s1.shape[1]
    params = []
    for j in range(m):
        pooled = np.concatenate([
            s1[:, j][~np.isnan(s1[:, j])],
            s2[:, j][~np.isnan(s2[:, j])],
        ])
        if len(pooled) > 1 and np.ptp(pooled) > 0:
            lo = np.percentile(pooled, 1)
            hi = np.percentile(pooled, 99)
            if hi <= lo:
                lo, hi = pooled.min(), pooled.max()
        else:
            lo, hi = 0.0, 1.0
        params.append((float(lo), float(hi)))
    return params


def normalize_and_diff(s1, s2, params):
    """Normalize each program's scores to [0,1] via min-max, then diff.

    Returns diffs in [-1, 1].  NaN where either score was NaN.
    """
    n, m = s1.shape
    diffs = np.full((n, m), np.nan)
    for j in range(m):
        lo, hi = params[j]
        rng = hi - lo
        if rng > 0:
            n1 = np.clip((s1[:, j] - lo) / rng, 0.0, 1.0)
            n2 = np.clip((s2[:, j] - lo) / rng, 0.0, 1.0)
        else:
            n1 = np.full(n, 0.5)
            n2 = np.full(n, 0.5)
        diffs[:, j] = n1 - n2
        mask = np.isnan(s1[:, j]) | np.isnan(s2[:, j])
        diffs[mask, j] = np.nan
    return diffs


# ── Threshold & labelling ────────────────────────────────────────────────

def apply_threshold(diffs, threshold):
    """2-class discrete labels with abstention zone."""
    labels = np.full(diffs.shape, -1, dtype=int)
    labels[diffs > threshold] = 0
    labels[diffs < -threshold] = 1
    labels[np.isnan(diffs)] = -1
    return labels


def tune_thresholds(val_diffs, y_val, candidates):
    """Per-program threshold tuning on full validation set."""
    m = val_diffs.shape[1]
    best_t = np.zeros(m)
    best_acc = np.zeros(m)
    for j in range(m):
        top_acc, top_t = -1.0, 0.0
        for t in candidates:
            votes = apply_threshold(val_diffs[:, j:j+1], t).flatten()
            valid = votes != -1
            if valid.sum() == 0:
                continue
            a = accuracy_score(y_val[valid], votes[valid])
            if a > top_acc or (a == top_acc and t > top_t):
                top_acc, top_t = a, t
        best_t[j] = top_t
        best_acc[j] = top_acc
    return best_t, best_acc


def build_label_matrix(diffs, indices, thresholds):
    n = diffs.shape[0]
    M = np.full((n, len(indices)), -1, dtype=int)
    for col, j in enumerate(indices):
        M[:, col] = apply_threshold(diffs[:, j:j+1], thresholds[j]).flatten()
    return M


# ── Cached score loading ─────────────────────────────────────────────────

def _cache_path(name):
    return os.path.join(OUTPUT_DIR, f"{name}.npy")


def _save_array(name, arr):
    """Save array as both .npy (for fast reloading) and .json (for portability)."""
    npy_path = _cache_path(name)
    np.save(npy_path, arr)

    json_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    obj = arr.tolist()
    with open(json_path, "w") as f:
        json.dump(obj, f, default=lambda x: None if np.isnan(x) else x)


def load_or_score(split_name, data, judges):
    p1, p2 = _cache_path(f"{split_name}_s1"), _cache_path(f"{split_name}_s2")
    if os.path.exists(p1) and os.path.exists(p2):
        s1, s2 = np.load(p1), np.load(p2)
        if s1.shape == (len(data), len(judges)) and s2.shape == (len(data), len(judges)):
            print(f"  Loading cached {split_name} scores ...")
            return s1, s2
        print(f"  Cache shape mismatch for {split_name} "
              f"(cached: {s1.shape[1]} programs, current: {len(judges)}). Re-scoring ...")
    else:
        print(f"  Scoring {split_name} set ...")
    s1, s2 = collect_raw_scores(data, judges, desc=f"{split_name} scoring")
    _save_array(f"{split_name}_s1", s1)
    _save_array(f"{split_name}_s2", s2)
    return s1, s2


# ══════════════════════════════════════════════════════════════════════════
def main():
    global OUTPUT_DIR, TAG, SEED, MIN_ACCURACY_THRESHOLD

    args = parse_args()

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(SCRIPT_DIR, p)

    VAL_SOURCE  = args.val  if args.val  else f"{args.repo_id} (validation)"
    TEST_SOURCE = args.test if args.test else f"{args.repo_id} (test)"
    JUDGE_DIR   = _resolve(args.judges)
    OUTPUT_DIR  = _resolve(args.output)
    TAG         = args.tag
    SEED        = args.seed
    MIN_ACCURACY_THRESHOLD = args.min_acc

    pipeline_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 60)
    print(f"PAJAMA Pipeline — {TAG}")
    print("=" * 60)
    print(f"  Dataset : {args.dataset}")
    print(f"  HF repo : {args.repo_id}")
    print(f"  Val     : {VAL_SOURCE}")
    print(f"  Test    : {TEST_SOURCE}")
    print(f"  Judges  : {JUDGE_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("STAGE 1: Loading data and collecting raw scores (s1, s2)")
    print("=" * 60)

    val_data,   y_val   = load_split(
        args.dataset, "validation", args.repo_id, args.val)
    test_data,  y_test  = load_split(
        args.dataset, "test", args.repo_id, args.test)
    judges, judge_names = load_judges(JUDGE_DIR)

    print(f"  Val: {len(val_data)}  | Test: {len(test_data)}")
    print(f"  Judges: {len(judges)} programs")

    val_s1,   val_s2   = load_or_score("val",   val_data,   judges)
    test_s1,  test_s2  = load_or_score("test",  test_data,  judges)

    print("\n" + "=" * 60)
    print("STAGE 2: Per-program min-max normalization")
    print("=" * 60)

    norm_params = compute_norm_params(val_s1, val_s2)
    for j, (lo, hi) in enumerate(norm_params):
        if j < 5 or j == len(norm_params) - 1:
            print(f"  {judge_names[j]:>20s}:  raw [{lo:+10.4f}, {hi:+10.4f}]")
        elif j == 5:
            print(f"  {'...':>20s}")

    val_diffs   = normalize_and_diff(val_s1,   val_s2,   norm_params)
    test_diffs  = normalize_and_diff(test_s1,  test_s2,  norm_params)

    _save_array("val_diffs",   val_diffs)
    _save_array("test_diffs",  test_diffs)

    print(f"\n  Normalized diff range (val): "
          f"[{np.nanmin(val_diffs):.4f}, {np.nanmax(val_diffs):.4f}]")

    print("\n" + "=" * 60)
    print("STAGE 3: Per-program threshold tuning (full val set)")
    print("=" * 60)

    best_thresholds, best_accuracies = tune_thresholds(
        val_diffs, y_val, THRESHOLD_CANDIDATES
    )

    print(f"\n  {'Program':<20s} {'Best T':>7} {'ValAcc':>8} {'Status':>8}")
    print("  " + "-" * 48)
    for j in range(len(judges)):
        tag = "Y" if best_accuracies[j] >= MIN_ACCURACY_THRESHOLD else "N"
        print(f"  {judge_names[j]:<20s} {best_thresholds[j]:>7.3f}"
              f" {best_accuracies[j]:>8.4f} {tag:>8}")

    _save_array("best_thresholds", best_thresholds)
    _save_array("best_accuracies", best_accuracies)

    print("\n" + "=" * 60)
    print("STAGE 4: Program filtering")
    print("=" * 60)

    surviving = np.where(best_accuracies >= MIN_ACCURACY_THRESHOLD)[0]
    cut       = np.where(best_accuracies <  MIN_ACCURACY_THRESHOLD)[0]

    print(f"  Kept: {len(surviving)}/{len(judges)}  |  Cut: {len(cut)}")
    if len(cut):
        print(f"  Cut: {[judge_names[i] for i in cut]}")

    print("\n" + "=" * 60)
    print("STAGE 5: Building label matrix (2-class with abstention)")
    print("=" * 60)

    M_val   = build_label_matrix(val_diffs,   surviving, best_thresholds)
    M_test  = build_label_matrix(test_diffs,  surviving, best_thresholds)

    unique, counts = np.unique(M_val, return_counts=True)
    total = M_val.size
    print(f"  M_val shape (all surviving): {M_val.shape}")
    for u, c in zip(unique, counts):
        name = {-1: "Abstain", 0: "Ans1", 1: "Ans2"}.get(u, str(u))
        print(f"    {name:>8s}: {c:>10d}  ({c/total*100:5.2f}%)")

    abstain_rate = (M_val == -1).mean()
    print(f"  Overall abstain rate: {abstain_rate*100:.1f}%")

    # ── Find best top-k by validation agreement ─────────────────────
    print("\n" + "=" * 60)
    print("STAGE 5b: Finding best top-k programs (by val agreement)")
    print("=" * 60)

    ranked = np.argsort(best_accuracies)[::-1]
    ranked_surv = [i for i in ranked if best_accuracies[i] >= MIN_ACCURACY_THRESHOLD]

    best_k, best_agree = len(ranked_surv), 0.0
    for k in tqdm(range(1, len(ranked_surv) + 1), desc="Sweeping k"):
        selected = ranked_surv[:k]
        M_va_k = build_label_matrix(val_diffs, selected, best_thresholds)
        if k >= 3:
            try:
                lm_k = LabelModel(cardinality=SNORKEL_CARDINALITY, verbose=False)
                lm_k.fit(L_train=M_va_k, Y_dev=y_val,
                         n_epochs=200, l2=SNORKEL_L2, lr=SNORKEL_LR,
                         seed=SNORKEL_SEED)
                yh_k = lm_k.predict(L=M_va_k, tie_break_policy="random")
            except Exception:
                continue
        else:
            mv_k = MajorityLabelVoter(cardinality=SNORKEL_CARDINALITY)
            yh_k = mv_k.predict(L=M_va_k, tie_break_policy="random")
        v = yh_k != -1
        if v.sum() > 0:
            agree = accuracy_score(y_val[v], yh_k[v])
            if agree > best_agree:
                best_agree = agree
                best_k = k

    selected_programs = ranked_surv[:best_k]
    print(f"  >>> Best k = {best_k} / {len(ranked_surv)} surviving")
    print(f"  >>> Val agreement = {best_agree:.4f}")
    print(f"  >>> Top programs: {[judge_names[i] for i in selected_programs[:5]]}"
          f"{'...' if best_k > 5 else ''}")

    # Rebuild M with only the best top-k programs
    surviving = np.array(selected_programs)
    M_val   = build_label_matrix(val_diffs,   selected_programs, best_thresholds)
    M_test  = build_label_matrix(test_diffs,  selected_programs, best_thresholds)

    test_conflict = LFAnalysis(L=M_test).label_conflict()
    print(f"  >>> Top-{best_k} test label conflict:       {test_conflict:.4f}")

    _save_array("M_val",   M_val)
    _save_array("M_test",  M_test)

    print(f"  M_val shape (top-{best_k}): {M_val.shape}")

    # ── Detect all-abstain test samples ──────────────────────────
    test_all_abstain_mask = (M_test == -1).all(axis=1)
    abstained_test_indices = np.where(test_all_abstain_mask)[0].tolist()
    covered_test_mask = ~test_all_abstain_mask
    n_test_total = len(y_test)
    n_test_covered = int(covered_test_mask.sum())
    n_test_abstained = len(abstained_test_indices)
    test_coverage_rate = n_test_covered / n_test_total if n_test_total > 0 else 0.0

    abstained_json = {
        "dataset": TAG,
        "n_total_test": n_test_total,
        "n_all_abstained": n_test_abstained,
        "n_covered": n_test_covered,
        "test_coverage": round(test_coverage_rate, 6),
        "abstained_sample_indices": abstained_test_indices,
    }
    abstained_path = os.path.join(OUTPUT_DIR, "abstained_test_number.json")
    with open(abstained_path, "w") as f:
        json.dump(abstained_json, f, indent=2)
    print(f"\n  All-abstain test samples: {n_test_abstained} / {n_test_total}")
    print(f"  Test coverage: {test_coverage_rate:.4f} ({n_test_covered}/{n_test_total})")
    print(f"  Saved to {abstained_path}")

    M_test_covered = M_test[covered_test_mask]
    y_test_covered = y_test[covered_test_mask]

    print("\n" + "=" * 60)
    print(f"STAGE 6: Training Snorkel LabelModel (top-{best_k} programs)")
    print("=" * 60)

    label_model = LabelModel(cardinality=SNORKEL_CARDINALITY, verbose=True)
    label_model.fit(
        L_train=M_val,
        Y_dev=y_val,
        n_epochs=SNORKEL_EPOCHS,
        l2=SNORKEL_L2,
        lr=SNORKEL_LR,
        log_freq=100,
        seed=SNORKEL_SEED,
    )

    # ── Val coverage detection (mirror of the test-side handling) ──
    # LabelModel.predict with tie_break_policy="random" returns 0/1 (never -1)
    # for all-abstain rows, which would otherwise leak random guesses into the
    # in-sample LabelModel_val accuracy / coverage.  Restrict the prediction to
    # covered val rows (rows where at least one LF voted) and leave the rest
    # as -1 so _full_metrics filters them out — same pattern as for test.
    val_all_abstain_mask = (M_val == -1).all(axis=1)
    covered_val_mask = ~val_all_abstain_mask
    n_val_total = len(y_val)
    n_val_covered = int(covered_val_mask.sum())
    M_val_covered = M_val[covered_val_mask]

    Y_hat_val = np.full(n_val_total, -1, dtype=int)
    if n_val_covered > 0:
        Y_hat_val[covered_val_mask] = label_model.predict(
            L=M_val_covered, tie_break_policy="random")

    Y_hat_test = np.full(n_test_total, -1, dtype=int)
    Y_hat_test_proba = np.full((n_test_total, SNORKEL_CARDINALITY), np.nan)
    if n_test_covered > 0:
        Y_hat_test[covered_test_mask] = label_model.predict(
            L=M_test_covered, tie_break_policy="random")
        Y_hat_test_proba[covered_test_mask] = label_model.predict_proba(
            L=M_test_covered)
    Y_hat_test_soft = Y_hat_test_proba[:, 1]

    mv = MajorityLabelVoter(cardinality=SNORKEL_CARDINALITY)
    Y_mv_test = np.full(n_test_total, -1, dtype=int)
    if n_test_covered > 0:
        Y_mv_test[covered_test_mask] = mv.predict(
            L=M_test_covered, tie_break_policy="random")

    def _full_metrics(y_true, y_pred):
        """Compute agreement (accuracy), precision, recall, F1, coverage."""
        v = y_pred != -1
        cov = v.mean() if len(v) else 0.0
        if v.sum() == 0:
            return {"agreement": 0, "precision": 0, "recall": 0, "f1": 0, "coverage": 0}
        yt, yp = y_true[v], y_pred[v]
        return {
            "agreement": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, average="macro", zero_division=0),
            "recall":    recall_score(yt, yp, average="macro", zero_division=0),
            "f1":        f1_score(yt, yp, average="macro", zero_division=0),
            "coverage":  float(cov),
        }

    metrics_lm_test = _full_metrics(y_test, Y_hat_test)
    metrics_mv_test = _full_metrics(y_test, Y_mv_test)
    metrics_lm_val  = _full_metrics(y_val,  Y_hat_val)

    header = f"  {'Method':<25s} {'Agree':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Cov':>8s}"
    print(f"\n{header}")
    print("  " + "-" * 65)
    for name, m in [("LabelModel (test)", metrics_lm_test),
                    ("MajorityVote (test)", metrics_mv_test),
                    ("LabelModel (val)", metrics_lm_val)]:
        print(f"  {name:<25s} {m['agreement']:>8.4f} {m['precision']:>8.4f}"
              f" {m['recall']:>8.4f} {m['f1']:>8.4f} {m['coverage']:>8.3f}")

    stage6_results = {
        "LabelModel_test":  metrics_lm_test,
        "MajorityVote_test": metrics_mv_test,
        "LabelModel_val":   metrics_lm_val,
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(stage6_results, f, indent=2)

    # ── Inference-only throughput measurement ────────────────────────
    print("\n" + "=" * 60)
    print(f"STAGE 6b: Measuring inference throughput "
          f"(top-{best_k} programs + label model)")
    print("=" * 60)

    selected_judge_fns = [judges[c] for c in selected_programs]

    prog_start = time.time()
    for row in tqdm(test_data, desc="Top-k program calls", total=len(test_data)):
        query, ans1, ans2 = _extract_fields(row)
        for fn in selected_judge_fns:
            try:
                _ = float(fn(query, ans1))
                _ = float(fn(query, ans2))
            except Exception:
                pass
    prog_elapsed = time.time() - prog_start

    lm_start = time.time()
    if n_test_covered > 0:
        _ = label_model.predict(L=M_test_covered, tie_break_policy="random")
    lm_elapsed = time.time() - lm_start

    inference_elapsed = prog_elapsed + lm_elapsed
    samples_per_sec = (len(test_data) / inference_elapsed
                       if inference_elapsed > 0 else 0.0)
    covered_per_sec = (n_test_covered / inference_elapsed
                       if inference_elapsed > 0 else 0.0)

    print(f"  Top-k program calls : {prog_elapsed:>8.3f}s  "
          f"on {len(test_data)} test samples  ({best_k} programs each)")
    print(f"  Label model infer   : {lm_elapsed:>8.3f}s  "
          f"on {n_test_covered} covered samples")
    print(f"  Total inference time: {inference_elapsed:>8.3f}s")
    print(f"  Throughput          : {samples_per_sec:>8.2f} samples/sec  "
          f"({covered_per_sec:.2f} covered/sec)")

    # ── Unified summary JSON (comparable to LLM judge summaries) ──
    pipeline_elapsed = time.time() - pipeline_start
    n_test = len(y_test)
    n_val = len(y_val)
    selected_col_indices = [int(c) for c in selected_programs]
    selected_program_names = [judge_names[c] for c in selected_col_indices]
    selected_program_ids = []
    for name in selected_program_names:
        m_id = re.search(r"(\d+)", name)
        selected_program_ids.append(int(m_id.group(1)) if m_id else -1)
    selected_program_thresholds = [float(best_thresholds[c]) for c in selected_col_indices]
    selected_program_val_accs = [float(best_accuracies[c]) for c in selected_col_indices]

    summary = {
        "dataset": TAG,
        "method": "program_judge",
        "best_k": int(best_k),
        "n_total_programs": len(judges),
        "n_surviving_programs": len(surviving),
        "selected_program_ids": selected_program_ids,
        "selected_program_names": selected_program_names,
        "selected_program_col_indices": selected_col_indices,
        "selected_program_thresholds": selected_program_thresholds,
        "selected_program_val_accuracies": selected_program_val_accs,
        "label_matrix_diagnostics": {
            "test_conflict": round(test_conflict, 4),
            "n_test_total": n_test_total,
            "n_test_covered": n_test_covered,
            "n_test_all_abstained": n_test_abstained,
            "test_coverage": round(test_coverage_rate, 6),
        },
        "LabelModel_test": {
            "n_total": n_test,
            "n_covered": n_test_covered,
            "accuracy": round(metrics_lm_test["agreement"], 4),
            "precision": round(metrics_lm_test["precision"], 4),
            "recall": round(metrics_lm_test["recall"], 4),
            "f1": round(metrics_lm_test["f1"], 4),
            "coverage": round(metrics_lm_test["coverage"], 4),
        },
        "MajorityVote_test": {
            "n_total": n_test,
            "n_covered": n_test_covered,
            "accuracy": round(metrics_mv_test["agreement"], 4),
            "precision": round(metrics_mv_test["precision"], 4),
            "recall": round(metrics_mv_test["recall"], 4),
            "f1": round(metrics_mv_test["f1"], 4),
            "coverage": round(metrics_mv_test["coverage"], 4),
        },
        "LabelModel_val": {
            "n_total": n_val,
            "n_covered": n_val_covered,
            "accuracy": round(metrics_lm_val["agreement"], 4),
            "precision": round(metrics_lm_val["precision"], 4),
            "recall": round(metrics_lm_val["recall"], 4),
            "f1": round(metrics_lm_val["f1"], 4),
            "coverage": round(metrics_lm_val["coverage"], 4),
        },
        "throughput": {
            "n_test": n_test,
            "n_test_covered": n_test_covered,
            "n_top_k_programs": int(best_k),
            "top_k_program_time_sec": round(prog_elapsed, 4),
            "label_model_infer_time_sec": round(lm_elapsed, 4),
            "inference_elapsed_sec": round(inference_elapsed, 4),
            "samples_per_sec": round(samples_per_sec, 2),
            "covered_samples_per_sec": round(covered_per_sec, 2),
            "pipeline_elapsed_sec": round(pipeline_elapsed, 1),
        },
    }
    summary_path = os.path.join(OUTPUT_DIR, f"{TAG}_pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Pipeline summary saved to {summary_path}")
    print(f"  Inference: {inference_elapsed:.2f}s on {n_test} test samples  "
          f"({samples_per_sec:.1f} samples/sec)  "
          f"[pipeline wall-clock: {pipeline_elapsed:.1f}s]")

    _save_array("Y_hat_val",        Y_hat_val)
    _save_array("Y_hat_test_soft",  Y_hat_test_soft)
    label_model.save(os.path.join(OUTPUT_DIR, f"{TAG}_trained_label_model.pkl"))

    print("\n" + "=" * 60)
    print(f"Snorkel pipeline ({TAG}) complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
