"""
PAJAMA demo pipelines: program loading, scoring, and Snorkel aggregation.

**Live mode** — val-free aggregation (no validation set):
  score rows → normalize → fixed abstain band → LabelModel.

**Demo / mock mode** — production pipeline ported from ``pajama_workflow/run.py``:
  cached val scores → per-program threshold tuning → filter → top-k → LabelModel.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

ProgressCallback = Callable[[float, str], None] | None


# ── Program loading ──────────────────────────────────────────────────────


@dataclass
class Program:
    """A single judging program plus its manifest metadata."""

    program_id: int
    filename: str
    code: str
    heuristic_id: int | None = None
    heuristic_name: str = ""
    variant: int | None = None
    approach_summary: str = ""
    status: str = "success"
    selected: bool = True
    fn: Any = None  # compiled judging_function callable
    dirty: bool = False  # True if code has been edited since load (invalidates score cache)

    @property
    def display_name(self) -> str:
        return f"judge_{self.program_id}"


def compile_program(code: str):
    """Compile a judge program string and return its judging_function."""
    ns: dict = {}
    exec(compile(code, "<judge>", "exec"), ns)
    fn = ns.get("judging_function")
    if fn is None:
        raise ValueError("Program does not define judging_function(query, response)")
    return fn


# Deterministic mapping used by the main PAJAMA generator:
#   judge_1..8  -> heuristic 1 (variants 1..8),  judge_9..16 -> heuristic 2,  ...
HEURISTIC_NAMES_BY_ID: dict[int, str] = {
    1: "Relevance to the Query",
    2: "Language Quality and Readability",
    3: "Completeness and Coverage",
    4: "Factual Accuracy Indicators",
    5: "Logical Coherence and Argument Structure",
    6: "Clarity and Conciseness",
    7: "Reasoning Transparency and Step-wise Formulation",
    8: "Epistemic Calibration and Uncertainty Communication",
    9: "Structural Organization and Formatting",
    10: "Evidence Density and Specificity",
}


def _infer_heuristic(program_id: int) -> tuple[int, int]:
    """Map program_id -> (heuristic_id, variant) under the 10 x 8 layout."""
    h_id = (program_id - 1) // 8 + 1
    variant = (program_id - 1) % 8 + 1
    return h_id, variant


def _approach_keywords(code: str) -> str:
    code_lower = code.lower()
    markers = [
        ("flesch", "Flesch readability"),
        ("syllable", "syllable counting"),
        ("sentence_len", "sentence length"),
        ("avg_word_len", "word length"),
        ("type_token", "type-token ratio"),
        ("unique_words", "vocabulary diversity"),
        ("bullet", "bullet/list detection"),
        ("header", "header detection"),
        ("paragraph", "paragraph analysis"),
        ("overlap", "word overlap"),
        ("jaccard", "Jaccard similarity"),
        ("cosine", "cosine similarity"),
        ("tfidf", "TF-IDF"),
        ("ngram", "n-gram"),
        ("hedge", "hedging language"),
        ("confident", "confidence markers"),
        ("transition", "transition words"),
        ("regex", "regex patterns"),
        ("entropy", "entropy"),
        ("concrete", "concreteness"),
        ("trigram", "char trigrams"),
        ("repetition", "repetition penalty"),
        ("specific", "specificity"),
        ("citation", "citation markers"),
        ("step", "step-by-step markers"),
        ("organize", "structural organization"),
    ]
    tags = [label for marker, label in markers if marker in code_lower]
    if not tags:
        tags = ["general heuristic"]
    return ", ".join(tags[:4])


def load_programs_from_dir(judges_dir: str, manifest_path: str | None = None) -> list[Program]:
    """Load all judge_*.py from a directory; merge manifest metadata when available,
    otherwise infer (heuristic, variant) from the deterministic 10x8 program-id layout.
    """
    manifest_by_id: dict[int, dict] = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for entry in json.load(f):
                manifest_by_id[int(entry["program_id"])] = entry

    programs: list[Program] = []
    files = sorted(
        (f for f in os.listdir(judges_dir) if f.startswith("judge_") and f.endswith(".py")),
        key=lambda f: int(f[len("judge_") : -len(".py")]),
    )
    for fname in files:
        pid = int(fname[len("judge_") : -len(".py")])
        with open(os.path.join(judges_dir, fname)) as f:
            code = f.read()
        meta = manifest_by_id.get(pid, {})
        h_id = meta.get("heuristic_id")
        variant = meta.get("variant")
        if h_id is None or variant is None:
            inferred_h, inferred_v = _infer_heuristic(pid)
            h_id = h_id if h_id is not None else inferred_h
            variant = variant if variant is not None else inferred_v
        heuristic_name = meta.get("heuristic_name") or HEURISTIC_NAMES_BY_ID.get(h_id, "")
        approach_summary = meta.get("approach_summary") or _approach_keywords(code)
        try:
            fn = compile_program(code)
            status = meta.get("status", "success")
        except Exception as exc:
            fn = None
            status = f"compile_error: {exc}"
        programs.append(
            Program(
                program_id=pid,
                filename=fname,
                code=code,
                heuristic_id=h_id,
                heuristic_name=heuristic_name,
                variant=variant,
                approach_summary=approach_summary,
                status=status,
                fn=fn,
            )
        )
    return programs


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Score collection ─────────────────────────────────────────────────────


def collect_raw_scores(rows: list[dict], programs: list[Program], progress_cb=None):
    """Score every (query, response1) and (query, response2) with every program.

    Returns (s1, s2) of shape (len(rows), len(programs)). NaN on failure.
    """
    n, m = len(rows), len(programs)
    s1 = np.full((n, m), np.nan)
    s2 = np.full((n, m), np.nan)
    for i, row in enumerate(rows):
        q = str(row.get("query", ""))
        r1 = str(row.get("response1", ""))
        r2 = str(row.get("response2", ""))
        for j, p in enumerate(programs):
            if p.fn is None:
                continue
            try:
                s1[i, j] = float(p.fn(q, r1))
            except Exception:
                pass
            try:
                s2[i, j] = float(p.fn(q, r2))
            except Exception:
                pass
        if progress_cb is not None:
            progress_cb(i + 1, n)
    return s1, s2


# ── Aggregation (val-free) ───────────────────────────────────────────────


@dataclass
class AggregationResult:
    s1: np.ndarray
    s2: np.ndarray
    diff: np.ndarray  # normalized diff in [-1, 1], NaN where any score missing
    M: np.ndarray  # label matrix in {-1, 0, 1}
    weights: np.ndarray  # per-program learned accuracy from LabelModel
    coverage: np.ndarray  # per-program fraction non-abstain
    conflicts: np.ndarray  # per-program conflict rate from LFAnalysis
    polarities: np.ndarray  # majority polarity (0 or 1) per program
    hard: np.ndarray  # predicted label per row (-1 abstain, 0=R1, 1=R2)
    soft: np.ndarray  # P(class=1) per row (NaN if all-abstain)
    label_model: Any
    norm_bounds: list[tuple[float, float]]  # per-program (lo, hi)
    abstain_band: float
    program_ids: list[int] = field(default_factory=list)


def _robust_bounds(col: np.ndarray) -> tuple[float, float]:
    finite = col[np.isfinite(col)]
    if finite.size < 2 or np.ptp(finite) == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi == lo:
            hi = lo + 1.0
    return lo, hi


def aggregate(
    s1: np.ndarray,
    s2: np.ndarray,
    program_ids: list[int],
    abstain_band: float = 0.02,
    epochs: int = 500,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 123,
) -> AggregationResult:
    n, m = s1.shape

    pooled = np.vstack([s1, s2])
    bounds = [_robust_bounds(pooled[:, j]) for j in range(m)]

    diff = np.full((n, m), np.nan)
    for j, (lo, hi) in enumerate(bounds):
        rng = hi - lo
        if rng > 0:
            n1 = np.clip((s1[:, j] - lo) / rng, 0.0, 1.0)
            n2 = np.clip((s2[:, j] - lo) / rng, 0.0, 1.0)
            d = n1 - n2
            mask = np.isnan(s1[:, j]) | np.isnan(s2[:, j])
            d[mask] = np.nan
            diff[:, j] = d

    M = np.full(diff.shape, -1, dtype=int)
    M[diff > abstain_band] = 0
    M[diff < -abstain_band] = 1
    M[np.isnan(diff)] = -1

    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=M, n_epochs=epochs, lr=lr, l2=l2, seed=seed)

    hard = label_model.predict(L=M, tie_break_policy="abstain")
    soft_full = label_model.predict_proba(L=M)  # (n, 2)
    soft = soft_full[:, 1].copy()
    all_abstain = (M == -1).all(axis=1)
    soft[all_abstain] = np.nan
    hard[all_abstain] = -1

    weights = np.asarray(label_model.get_weights(), dtype=float)
    coverage = (M != -1).mean(axis=0)
    try:
        analysis = LFAnalysis(L=M).lf_summary()
        conflicts = analysis["Conflicts"].to_numpy()
        polarities_raw = analysis["Polarity"].tolist()
        polarities = np.array(
            [(p[0] if isinstance(p, (list, tuple)) and len(p) else -1) for p in polarities_raw]
        )
    except Exception:
        conflicts = np.zeros(m)
        polarities = np.full(m, -1)

    return AggregationResult(
        s1=s1,
        s2=s2,
        diff=diff,
        M=M,
        weights=weights,
        coverage=coverage,
        conflicts=conflicts,
        polarities=polarities,
        hard=hard,
        soft=soft,
        label_model=label_model,
        norm_bounds=bounds,
        abstain_band=abstain_band,
        program_ids=list(program_ids),
    )


def attach_predictions_to_rows(
    rows: list[dict], hard: np.ndarray, soft: np.ndarray
) -> list[dict]:
    """Attach PAJAMA predictions to row dicts (verdict 1=R1, 2=R2)."""
    out: list[dict] = []
    for i, row in enumerate(rows):
        pred = int(hard[i])
        prob = float(soft[i]) if not np.isnan(soft[i]) else None
        labeled = dict(row)
        labeled["pajama_predicted_verdict"] = {0: 1, 1: 2, -1: None}[pred]
        labeled["pajama_response2_prob"] = prob
        labeled["pajama_abstained"] = pred == -1
        out.append(labeled)
    return out


def label_jsonl_export(rows: list[dict], result: AggregationResult) -> list[dict]:
    """Return the rows with predicted labels attached (ready to write back to JSONL)."""
    return attach_predictions_to_rows(rows, result.hard, result.soft)


# ── Production pipeline (mock mode; run.py stages 2–6) ───────────────────

SNORKEL_CARDINALITY = 2
SNORKEL_EPOCHS = 500
SNORKEL_SEED = 123
SNORKEL_L2 = 0.01
SNORKEL_LR = 0.01
MIN_ACCURACY_THRESHOLD = 0.50
N_PROGRAMS = 80


def threshold_candidates(threshold_max: float) -> np.ndarray:
    """Per-program tuning grid; ``run.py`` uses 0.00–0.15 (step 0.01)."""
    tmax = float(np.clip(threshold_max, 0.0, 0.50))
    return np.arange(0.0, tmax + 0.005, 0.01)


def compute_norm_params(s1: np.ndarray, s2: np.ndarray) -> list[tuple[float, float]]:
    m = s1.shape[1]
    params: list[tuple[float, float]] = []
    for j in range(m):
        pooled = np.concatenate([
            s1[:, j][~np.isnan(s1[:, j])],
            s2[:, j][~np.isnan(s2[:, j])],
        ])
        if len(pooled) > 1 and np.ptp(pooled) > 0:
            lo = float(np.percentile(pooled, 1))
            hi = float(np.percentile(pooled, 99))
            if hi <= lo:
                lo, hi = float(pooled.min()), float(pooled.max())
        else:
            lo, hi = 0.0, 1.0
        params.append((lo, hi))
    return params


def normalize_and_diff(
    s1: np.ndarray, s2: np.ndarray, params: list[tuple[float, float]]
) -> np.ndarray:
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


def apply_threshold(diffs: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.full(diffs.shape, -1, dtype=int)
    labels[diffs > threshold] = 0
    labels[diffs < -threshold] = 1
    labels[np.isnan(diffs)] = -1
    return labels


def tune_thresholds(
    val_diffs: np.ndarray,
    y_val: np.ndarray,
    candidates: np.ndarray,
    progress_callback: ProgressCallback = None,
    progress_lo: float = 0.0,
    progress_hi: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    m = val_diffs.shape[1]
    best_t = np.zeros(m)
    best_acc = np.zeros(m)
    for j in range(m):
        top_acc, top_t = -1.0, 0.0
        for t in candidates:
            votes = apply_threshold(val_diffs[:, j : j + 1], float(t)).flatten()
            valid = votes != -1
            if valid.sum() == 0:
                continue
            a = accuracy_score(y_val[valid], votes[valid])
            if a > top_acc or (a == top_acc and t > top_t):
                top_acc, top_t = a, float(t)
        best_t[j] = top_t
        best_acc[j] = top_acc
        if progress_callback is not None:
            frac = progress_lo + (j + 1) / m * (progress_hi - progress_lo)
            progress_callback(frac, f"Tuning per-program thresholds… {j + 1}/{m}")
    return best_t, best_acc


def build_label_matrix(
    diffs: np.ndarray, indices: np.ndarray | list[int], thresholds: np.ndarray
) -> np.ndarray:
    n = diffs.shape[0]
    indices = list(indices)
    M = np.full((n, len(indices)), -1, dtype=int)
    for col, j in enumerate(indices):
        M[:, col] = apply_threshold(diffs[:, j : j + 1], float(thresholds[j])).flatten()
    return M


def _full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    v = y_pred != -1
    cov = float(v.mean()) if len(v) else 0.0
    if v.sum() == 0:
        return {"agreement": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "coverage": 0.0}
    yt, yp = y_true[v], y_pred[v]
    return {
        "agreement": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "coverage": cov,
    }


def _program_id_from_col(j: int) -> int:
    return j + 1


def _judge_name(j: int) -> str:
    return f"judge_{_program_id_from_col(j)}"


def load_demo_val_arrays(outputs_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    root = Path(outputs_dir)
    p1, p2, pg = root / "val_s1.npy", root / "val_s2.npy", root / "val_gold.npy"
    if not (p1.exists() and p2.exists() and pg.exists()):
        return None
    return np.load(p1), np.load(p2), np.load(pg)


def save_summary(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


@dataclass
class ProductionPipelineResult:
    """Outcome of one production-pipeline run at a given ``threshold_max``."""

    summary: dict
    label_model: Any
    threshold_max: float
    norm_params: list[tuple[float, float]]
    best_thresholds: np.ndarray
    best_accuracies: np.ndarray
    selected_col_indices: list[int]
    selected_program_ids: list[int]
    M_val: np.ndarray
    Y_hat_val: np.ndarray
    y_val: np.ndarray
    weights: np.ndarray
    coverage: np.ndarray
    conflicts: np.ndarray
    program_ids_for_display: list[int] = field(default_factory=list)


def run_from_cached_scores(
    val_s1: np.ndarray,
    val_s2: np.ndarray,
    y_val: np.ndarray,
    *,
    threshold_max: float = 0.14,
    min_accuracy: float = MIN_ACCURACY_THRESHOLD,
    tag: str = "judgelm",
    progress_callback: ProgressCallback = None,
) -> ProductionPipelineResult:
    """Run stages 2–6 of ``run.py`` on cached validation scores."""

    def report(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, frac)), msg)

    report(0.0, "Starting production pipeline…")
    candidates = threshold_candidates(threshold_max)

    report(0.05, "Normalizing scores (per-program min-max)…")
    norm_params = compute_norm_params(val_s1, val_s2)
    val_diffs = normalize_and_diff(val_s1, val_s2, norm_params)

    best_thresholds, best_accuracies = tune_thresholds(
        val_diffs,
        y_val,
        candidates,
        progress_callback=progress_callback,
        progress_lo=0.10,
        progress_hi=0.42,
    )

    report(0.45, "Filtering programs by validation accuracy…")
    surviving = np.where(best_accuracies >= min_accuracy)[0]
    ranked = np.argsort(best_accuracies)[::-1]
    ranked_surv = [int(i) for i in ranked if best_accuracies[i] >= min_accuracy]

    report(0.48, f"Selecting top-k programs (1–{N_PROGRAMS} of {N_PROGRAMS} judges)…")
    best_k, best_agree = len(ranked_surv), 0.0
    n_sweep = max(len(ranked_surv), 1)
    for k in range(1, len(ranked_surv) + 1):
        selected = ranked_surv[:k]
        M_va_k = build_label_matrix(val_diffs, selected, best_thresholds)
        if k >= 3:
            try:
                lm_k = LabelModel(cardinality=SNORKEL_CARDINALITY, verbose=False)
                lm_k.fit(
                    L_train=M_va_k,
                    Y_dev=y_val,
                    n_epochs=200,
                    l2=SNORKEL_L2,
                    lr=SNORKEL_LR,
                    seed=SNORKEL_SEED,
                )
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
        report(
            0.48 + 0.24 * (k / n_sweep),
            f"Top-k sweep… {k}/{N_PROGRAMS}",
        )

    selected_programs = ranked_surv[:best_k]
    M_val = build_label_matrix(val_diffs, selected_programs, best_thresholds)

    report(0.75, f"Training Snorkel LabelModel (top-{best_k} programs)…")
    label_model = LabelModel(cardinality=SNORKEL_CARDINALITY, verbose=False)
    label_model.fit(
        L_train=M_val,
        Y_dev=y_val,
        n_epochs=SNORKEL_EPOCHS,
        l2=SNORKEL_L2,
        lr=SNORKEL_LR,
        seed=SNORKEL_SEED,
    )

    report(0.92, "Computing validation predictions and metrics…")
    val_all_abstain = (M_val == -1).all(axis=1)
    covered_val = ~val_all_abstain
    n_val_total = len(y_val)
    n_val_covered = int(covered_val.sum())

    Y_hat_val = np.full(n_val_total, -1, dtype=int)
    if n_val_covered > 0:
        Y_hat_val[covered_val] = label_model.predict(
            L=M_val[covered_val], tie_break_policy="random"
        )

    metrics_lm_val = _full_metrics(y_val, Y_hat_val)

    weights = np.asarray(label_model.get_weights(), dtype=float)
    coverage = (M_val != -1).mean(axis=0)
    try:
        analysis = LFAnalysis(L=M_val).lf_summary()
        conflicts = analysis["Conflicts"].to_numpy()
    except Exception:
        conflicts = np.zeros(len(selected_programs))

    report(1.0, "Pipeline complete.")

    selected_col_indices = [int(c) for c in selected_programs]
    selected_program_names = [_judge_name(c) for c in selected_col_indices]
    selected_program_ids = [_program_id_from_col(c) for c in selected_col_indices]
    selected_program_thresholds = [float(best_thresholds[c]) for c in selected_col_indices]
    selected_program_val_accs = [float(best_accuracies[c]) for c in selected_col_indices]

    summary = {
        "dataset": tag,
        "method": "program_judge",
        "threshold_max": round(float(threshold_max), 4),
        "best_k": int(best_k),
        "n_total_programs": int(val_s1.shape[1]),
        "n_surviving_programs": int(len(surviving)),
        "selected_program_ids": selected_program_ids,
        "selected_program_names": selected_program_names,
        "selected_program_col_indices": selected_col_indices,
        "selected_program_thresholds": selected_program_thresholds,
        "selected_program_val_accuracies": selected_program_val_accs,
        "LabelModel_val": {
            "n_total": n_val_total,
            "n_covered": n_val_covered,
            "accuracy": round(metrics_lm_val["agreement"], 4),
            "precision": round(metrics_lm_val["precision"], 4),
            "recall": round(metrics_lm_val["recall"], 4),
            "f1": round(metrics_lm_val["f1"], 4),
            "coverage": round(metrics_lm_val["coverage"], 4),
        },
    }

    return ProductionPipelineResult(
        summary=summary,
        label_model=label_model,
        threshold_max=float(threshold_max),
        norm_params=norm_params,
        best_thresholds=best_thresholds,
        best_accuracies=best_accuracies,
        selected_col_indices=selected_col_indices,
        selected_program_ids=selected_program_ids,
        M_val=M_val,
        Y_hat_val=Y_hat_val,
        y_val=y_val,
        weights=weights,
        coverage=coverage,
        conflicts=conflicts,
        program_ids_for_display=selected_program_ids,
    )


def predict_on_scores(
    s1: np.ndarray,
    s2: np.ndarray,
    pipe: ProductionPipelineResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted production pipeline to arbitrary (n, 80) score matrices."""
    diffs = normalize_and_diff(s1, s2, pipe.norm_params)
    M = build_label_matrix(diffs, pipe.selected_col_indices, pipe.best_thresholds)

    n = M.shape[0]
    hard = np.full(n, -1, dtype=int)
    soft = np.full(n, np.nan)

    all_abstain = (M == -1).all(axis=1)
    covered = ~all_abstain
    if covered.any():
        hard[covered] = pipe.label_model.predict(L=M[covered], tie_break_policy="abstain")
        proba = pipe.label_model.predict_proba(L=M[covered])
        soft[covered] = proba[:, 1]
    hard[all_abstain] = -1
    soft[all_abstain] = np.nan
    return hard, soft
