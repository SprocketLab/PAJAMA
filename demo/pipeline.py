"""
Val-free Snorkel aggregation for PAJAMA-style pairwise preference labeling.

Pipeline (no validation set required):
  1. Score each (response1, response2) with all N programs -> (n, N) matrices s1, s2.
  2. Per-program robust normalization (P1/P99 over pooled scores).
  3. diff = norm(s1) - norm(s2) in [-1, 1].
  4. Vote with a small fixed abstain band:
        diff >  band -> 0  (response1 preferred)
        diff < -band -> 1  (response2 preferred)
        else         -> -1 (abstain)
  5. Fit Snorkel LabelModel directly on the (n, N) label matrix.
  6. Return per-program weights, coverage, conflict, plus hard/soft labels.

No threshold tuning, no top-k pruning, no validation set.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel


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


def label_jsonl_export(rows: list[dict], result: AggregationResult) -> list[dict]:
    """Return the rows with predicted labels attached (ready to write back to JSONL)."""
    out: list[dict] = []
    for i, row in enumerate(rows):
        pred = int(result.hard[i])
        prob = float(result.soft[i]) if not np.isnan(result.soft[i]) else None
        labeled = dict(row)
        labeled["pajama_predicted_verdict"] = {0: 1, 1: 2, -1: None}[pred]
        labeled["pajama_response2_prob"] = prob
        labeled["pajama_abstained"] = pred == -1
        out.append(labeled)
    return out
