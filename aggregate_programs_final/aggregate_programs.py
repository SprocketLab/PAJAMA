"""
aggregate_programs.py — Use Claude Opus 4.7 Adaptive Thinking to synthesize
selected judge programs into a small number of stronger aggregated programs.

Instead of Snorkel LabelModel aggregation (which barely beats majority vote),
this script sends the selected programs' source code (and optionally their
validation accuracy / coverage) to Claude and asks it to autonomously design
N optimal aggregated programs. Claude is NOT told what dimension or heuristic
each aggregated program should target — it decides the strategy itself.

What Claude is shown about each input program is controlled by --info-mode,
which has four supported levels:

    code        : source code only (no per-program acc/cov)
    acc         : source code + per-program validation accuracy
    cov         : source code + per-program val accuracy + coverage  (default)
    val-samples : source code only (NO per-program acc/coverage) +
                  coverage-preserving denoising context (failure/correct
                  cases, score calibration, ensemble role design)

In ALL info-modes, Claude is also shown:
    - The generation blueprint (the 10-heuristic × 8-variant outline that was
      used to generate the original 80-program pool), and
    - A "Designed for heuristic <id>: <name>" tag for each input program
      indicating which of the 10 dimensions that specific program targets.
These are design-time facts about the program pool — not per-program
performance numbers — so they are safe to expose in every mode (including
val-samples).

Programs are ordered consistently with the chosen mode:
    - 'acc' / 'cov'              : by validation accuracy descending
    - 'code' / 'val-samples'     : alphabetically (no accuracy leak)

Downstream evaluation tunes per-program decision thresholds on the validation
set, builds a Snorkel-style label matrix M of shape (n_samples, n_programs)
with values in {-1, 0, 1}, and then runs the same aggregation that
snorkel_pipeline_final.py uses on the original 80-program pool:
    1) Snorkel LabelModel  (cardinality=2, epochs=500, seed=123) trained on
       M_val/y_val and evaluated on M_val and the covered rows of M_test.
    2) Snorkel MajorityLabelVoter on the same M_val and covered M_test rows.
Both report accuracy/precision/recall/F1/coverage, so numbers are directly
comparable to the original pipeline's <tag>_pipeline_summary.json.

Pipeline outputs are read from the *_final directories, which exclude samples
where ALL judges abstained.

Uses adaptive thinking (required for Opus 4.7+; manual budget_tokens is not
supported). Effort level controls thinking depth: low / medium / high / xhigh / max.
Docs recommend xhigh for coding/agentic tasks on Opus 4.7.

Supported datasets: judgelm, pandalm, multipref, prometheus, hendrydong.
Use --dataset <name> to auto-fill --summary, --judges, --output, --eval-* paths.

Usage (preset mode — recommended):
    python aggregate_programs.py --dataset judgelm                     # cov mode
    python aggregate_programs.py --dataset judgelm --info-mode acc
    python aggregate_programs.py --dataset judgelm --info-mode code
    python aggregate_programs.py --dataset judgelm --info-mode val-samples

Usage (eval-only — re-evaluate existing programs without calling Claude):
    python aggregate_programs.py --dataset judgelm --info-mode code --n-programs 5 --eval-only
    python aggregate_programs.py --dataset judgelm --info-mode code --n-programs 5 --eval-only --eval-thresholds 0.50
    # --eval-thresholds sets the upper bound of threshold search (default 0.15)
    # Looks for programs in aggregated_programs_final_judgelm_5_code/
    # Errors if the directory or programs don't exist

Usage (manual mode — omit --output to get auto-naming with dataset/n/mode):
    python aggregate_programs.py \\
        --summary snorkel_label_model_pipeline_final_new/pipeline_outputs_judgelm_final/judgelm_pipeline_summary.json \\
        --judges  judge_programs_judgelm \\
        --n-programs 5 --effort xhigh --info-mode cov \\
        --eval-val judgelm_val_500.jsonl \\
        --eval-test judgelm_test_5000.jsonl
    # → output dir: aggregated_programs_final_5_cov/
"""

import os
import re
import json
import random
import argparse
import textwrap
import anthropic
import numpy as np
from pathlib import Path

from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── Snorkel evaluation hyperparameters ─────────────────────────────────────
# These match snorkel_label_model_pipeline_final_new/snorkel_pipeline_final.py
# so the aggregated programs are evaluated under the same conditions as the
# original 80-program pool.
SNORKEL_CARDINALITY = 2
SNORKEL_EPOCHS = 500
SNORKEL_SEED = 123
SNORKEL_L2 = 0.01
SNORKEL_LR = 0.01


# ── Original Judge Program Generation Blueprint ─────────────────────────────
# These constants mirror judge_programs/judge_programs_scripts/prompt.py
# and the per-dataset generate_judging_programs_*.py scripts. Every dataset's
# 80-program pool was generated by the SAME 10-heuristic × 8-variant outline,
# so this single blueprint is canonical across judgelm / pandalm / multipref /
# prometheus / hendrydong. We expose this blueprint to Claude so it knows the
# design provenance of the top-k input programs it is asked to aggregate.

PROGRAMS_PER_HEURISTIC = 8

HEURISTICS = {
    1: {
        "name": "Relevance to the Query",
        "description": (
            "Evaluate how semantically relevant the response is to the question asked. "
            "Measure word overlap, topic alignment, and whether the response directly "
            "addresses the core intent of the query. Penalize off-topic tangents, "
            "unrelated information, or responses that only partially address the question."
        ),
    },
    2: {
        "name": "Language Quality and Readability",
        "description": (
            "Evaluate language quality and readability. Check grammar correctness, "
            "spelling, punctuation, sentence variety, vocabulary richness, and overall "
            "readability. Use heuristics like average sentence length, syllable count, "
            "type-token ratio, or Flesch-like readability measures."
        ),
    },
    3: {
        "name": "Completeness and Coverage",
        "description": (
            "Evaluate completeness and thoroughness of the answer. Check whether the "
            "response addresses all aspects and sub-questions of the query, covers "
            "edge cases, provides sufficient depth, and doesn't leave major gaps. "
            "Penalize partial or superficial answers."
        ),
    },
    4: {
        "name": "Factual Accuracy Indicators",
        "description": (
            "Evaluate indicators of factual reliability. Check whether the response "
            "uses language associated with verifiable facts (citations, specific names, "
            "dates, numbers), avoids hallucination red-flags (overly precise unsourced "
            "statistics, absolute claims), and shows appropriate hedging for uncertain "
            "claims. Penalize sensationalism and conspiracy-style language."
        ),
    },
    5: {
        "name": "Logical Coherence and Argument Structure",
        "description": (
            "Evaluate logical coherence. Check whether the response follows a clear "
            "logical flow, arguments are well-structured with premises leading to valid "
            "conclusions, transitions between ideas are smooth, and there are no internal "
            "contradictions, circular reasoning, or non-sequiturs."
        ),
    },
    6: {
        "name": "Clarity and Conciseness",
        "description": (
            "Evaluate clarity and conciseness. Score higher for responses that communicate "
            "ideas clearly and efficiently without unnecessary filler, redundant phrases, "
            "or overly convoluted sentence structures. Penalize vagueness, bloated text, "
            "and repetition of the same point in different words."
        ),
    },
    7: {
        "name": "Reasoning Transparency and Step-wise Formulation",
        "description": (
            "Evaluate how transparently the response shows its reasoning process. "
            "Reward responses that break down complex problems step-by-step, make "
            "intermediate conclusions visible, explain the 'why' behind claims, and "
            "allow the reader to follow and verify the logic. Penalize opaque answers "
            "that jump directly to conclusions without showing reasoning."
        ),
    },
    8: {
        "name": "Epistemic Calibration and Uncertainty Communication",
        "description": (
            "Evaluate how well the response communicates confidence and uncertainty. "
            "Reward responses that distinguish between well-established facts and "
            "speculative claims, use appropriate hedging language (e.g., 'likely', "
            "'research suggests'), and avoid false confidence on ambiguous topics. "
            "Penalize overconfident claims and responses that present speculation as fact."
        ),
    },
    9: {
        "name": "Structural Organization and Formatting",
        "description": (
            "Evaluate the structural organization of the response. Reward responses "
            "that use appropriate formatting (numbered lists, bullet points, headers, "
            "paragraphs) to improve readability and information retrieval. Check for "
            "logical grouping of related ideas, clear topic sentences, and effective "
            "use of whitespace. Penalize wall-of-text responses and poorly organized "
            "information dumps."
        ),
    },
    10: {
        "name": "Evidence Density and Specificity",
        "description": (
            "Evaluate the density of concrete evidence and specific details in the "
            "response. Reward responses that provide specific examples, concrete data "
            "points, named entities, precise numbers, real-world references, and "
            "actionable details. Penalize vague, hand-wavy responses that use generic "
            "filler like 'many people think', 'it depends', or 'there are various "
            "factors' without actually specifying them."
        ),
    },
}


DATASET_PRESETS = {
    "judgelm": {
        "summary": "snorkel_label_model_pipeline_final_new/pipeline_outputs_judgelm_final/judgelm_pipeline_summary.json",
        "pipeline_outputs_dir": "snorkel_label_model_pipeline_final_new/pipeline_outputs_judgelm_final",
        "judges": "judge_programs_judgelm",
        "output_base": "aggregated_programs_final_judgelm",
        "eval_val": "judgelm_val_500.jsonl",
        "eval_test": "judgelm_test_5000.jsonl",
    },
    "pandalm": {
        "summary": "snorkel_label_model_pipeline_final_new/pipeline_outputs_pandalm_final/pandalm_pipeline_summary.json",
        "pipeline_outputs_dir": "snorkel_label_model_pipeline_final_new/pipeline_outputs_pandalm_final",
        "judges": "judge_programs_pandalm",
        "output_base": "aggregated_programs_final_pandalm",
        "eval_val": "pandalm_val_500_v2.jsonl",
        "eval_test": "pandalm_test_894.jsonl",
    },
    "multipref": {
        "summary": "snorkel_label_model_pipeline_final_new/pipeline_outputs_multipref_final/multipref_pipeline_summary.json",
        "pipeline_outputs_dir": "snorkel_label_model_pipeline_final_new/pipeline_outputs_multipref_final",
        "judges": "judge_programs_multipref",
        "output_base": "aggregated_programs_final_multipref",
        "eval_val": "multipref_val_170.jsonl",
        "eval_test": "multipref_test_1700.jsonl",
    },
    "prometheus": {
        "summary": "snorkel_label_model_pipeline_final_new/pipeline_outputs_prometheus_final/prometheus_pipeline_summary.json",
        "pipeline_outputs_dir": "snorkel_label_model_pipeline_final_new/pipeline_outputs_prometheus_final",
        "judges": "judge_programs_prometheus",
        "output_base": "aggregated_programs_final_prometheus",
        "eval_val": "prometheus_val_500.jsonl",
        "eval_test": "prometheus_test_5000.jsonl",
    },
    "hendrydong": {
        "summary": "snorkel_label_model_pipeline_final_new/pipeline_outputs_hendrydong_final/hendrydong_pipeline_summary.json",
        "pipeline_outputs_dir": "snorkel_label_model_pipeline_final_new/pipeline_outputs_hendrydong_final",
        "judges": "judge_programs_hendrydong",
        "output_base": "aggregated_programs_final_hendrydong",
        "eval_val": "hendrydong_val_500.jsonl",
        "eval_test": "hendrydong_test_5000.jsonl",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate selected judge programs via Claude Opus 4.7 Adaptive Thinking")
    p.add_argument("--dataset", default=None,
                   choices=list(DATASET_PRESETS.keys()),
                   help="Dataset name — auto-fills --summary, --judges, --output, "
                        "--eval-* from presets. Overrides can still be given explicitly.")
    p.add_argument("--summary", default=None,
                   help="Path to pipeline summary JSON (auto-filled by --dataset)")
    p.add_argument("--pipeline-outputs-dir", default=None,
                   help="Directory containing pipeline outputs (val_diffs.json, "
                        "best_thresholds.json, etc.) — auto-filled by --dataset")
    p.add_argument("--judges", default=None,
                   help="Directory containing judge_*.py programs (auto-filled by --dataset)")
    p.add_argument("--output", default=None,
                   help="Output directory for aggregated programs (auto-filled by --dataset)")
    p.add_argument("--n-programs", type=int, default=5,
                   help="Number of aggregated programs to generate (default: 5)")
    p.add_argument("--max-output-tokens", type=int, default=128000,
                   help="Max output tokens — ceiling for thinking+text combined. "
                        "Only actually-generated tokens are billed. (default: 128000)")
    p.add_argument("--effort", default="xhigh",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Effort level: low/medium/high/xhigh/max. "
                        "Opus 4.7 docs recommend xhigh for coding tasks. "
                        "(default: xhigh)")
    p.add_argument("--info-mode", default="cov",
                   choices=["code", "acc", "cov", "val-samples"],
                   help="What metadata to expose to Claude per input program: "
                        "'code' = source code only (no metadata), "
                        "'acc'  = source + per-program validation accuracy, "
                        "'cov'  = source + per-program val accuracy + coverage, "
                        "'val-samples' = source code only (NO per-program "
                        "acc/coverage) + coverage-preserving denoising "
                        "context (failure cases, correct cases, ensemble "
                        "roles, score calibration). "
                        "(default: cov)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (used by majority-vote tie-breaking; "
                        "ensures evaluation is reproducible). (default: 42)")
    p.add_argument("--eval-val", default=None,
                   help="Path to validation JSONL (auto-filled by --dataset)")
    p.add_argument("--eval-test", default=None,
                   help="Path to test JSONL (auto-filled by --dataset)")
    p.add_argument("--model", default="claude-opus-4-7",
                   help="Anthropic model ID (default: claude-opus-4-7)")
    p.add_argument("--tag", default=None,
                   help="Tag for output filenames (default: basename of --output)")
    p.add_argument("--max-resume-rounds", type=int, default=5,
                   help="Max API rounds for resume/retry (default: 5)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip generation, only evaluate existing programs in --output")
    p.add_argument("--eval-thresholds", type=float, default=0.15,
                   help="Upper bound of threshold search range for evaluation "
                        "(searched in steps of 0.01 from 0.00 to this value). "
                        "(default: 0.15)")

    args = p.parse_args()

    if args.dataset:
        preset = DATASET_PRESETS[args.dataset]
        if args.summary is None:
            args.summary = preset["summary"]
        if args.pipeline_outputs_dir is None:
            args.pipeline_outputs_dir = preset["pipeline_outputs_dir"]
        if args.judges is None:
            args.judges = preset["judges"]
        if args.output is None:
            args.output = f"{preset['output_base']}_{args.n_programs}_{args.info_mode}"
        if args.eval_val is None:
            args.eval_val = preset["eval_val"]
        if args.eval_test is None:
            args.eval_test = preset["eval_test"]

    if args.summary is None or args.judges is None:
        p.error("Either --dataset or both --summary and --judges are required.")
    if args.pipeline_outputs_dir is None:
        args.pipeline_outputs_dir = os.path.dirname(args.summary)
    if args.output is None:
        args.output = f"aggregated_programs_final_{args.n_programs}_{args.info_mode}"

    return args


def resolve(p):
    """Resolve a path relative to PROJECT_ROOT (not the script's own directory).

    All preset paths (summary, judges, eval data, etc.) are relative to the
    project root (~/pajama/), not to the program_aggregation_final/ subdirectory
    where this script lives.
    """
    return str(p) if os.path.isabs(p) else str(PROJECT_ROOT / p)


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def load_program_source(judge_dir, program_name):
    path = os.path.join(judge_dir, f"{program_name}.py")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_program_val_stats(pipeline_outputs_dir, summary):
    """Load per-program validation statistics directly from pipeline_outputs.

    Reads val_diffs.json (n_val x n_total_programs, normalized score diffs)
    from pipeline_outputs_dir. Combined with selected_program_col_indices and
    selected_program_thresholds from the summary, returns per-program coverage
    at threshold (i.e. fraction of validation samples where |diff| > threshold).

    Returns
    -------
    stats : dict
        {program_name: {"coverage": float, "n_total": int}}
    n_total_programs : int or None
        Total number of judge programs in the original pool (= number of columns
        in val_diffs). None if val_diffs.json could not be loaded.
    """
    fpath = os.path.join(pipeline_outputs_dir, "val_diffs.json")
    if not os.path.exists(fpath):
        print(f"  WARNING: {fpath} not found, skipping val stats")
        return {}, None

    with open(fpath) as f:
        val_diffs = np.array(json.load(f))

    n_total_programs = int(val_diffs.shape[1])

    names = summary["selected_program_names"]
    col_indices = summary["selected_program_col_indices"]
    thresholds = summary["selected_program_thresholds"]

    stats = {}
    for i, name in enumerate(names):
        col_idx = col_indices[i]
        thr = thresholds[i]
        diffs = val_diffs[:, col_idx]
        d_valid = diffs[~np.isnan(diffs)]
        n_total = len(diffs)

        if len(d_valid) == 0 or n_total == 0:
            stats[name] = {"coverage": 0.0, "n_total": n_total}
            continue

        n_covered = int((np.abs(d_valid) > thr).sum())
        stats[name] = {
            "coverage": n_covered / n_total,
            "n_total": n_total,
        }

    return stats, n_total_programs


def _heuristic_for_judge_name(name):
    """Map a judge program filename to its source heuristic.

    The 80 original programs were generated as 10 heuristics × 8 variants in
    a fixed deterministic order: judge_1..judge_8 → heuristic 1,
    judge_9..judge_16 → heuristic 2, ..., judge_73..judge_80 → heuristic 10.

    Parameters
    ----------
    name : str
        Program name, e.g. ``"judge_4"`` or ``"judge_4.py"``.

    Returns
    -------
    (int | None, dict | None)
        ``(heuristic_id, heuristic_dict)`` if recognized; ``(None, None)``
        if the name does not follow the standard convention or the parsed
        number is outside ``1..len(HEURISTICS)*PROGRAMS_PER_HEURISTIC``.
    """
    m = re.search(r"(\d+)", name)
    if not m:
        return None, None
    n = int(m.group(1))
    if n < 1 or n > len(HEURISTICS) * PROGRAMS_PER_HEURISTIC:
        return None, None
    h_id = (n - 1) // PROGRAMS_PER_HEURISTIC + 1
    if h_id not in HEURISTICS:
        return None, None
    return h_id, HEURISTICS[h_id]


def _format_generation_blueprint():
    """Render the 10-heuristic blueprint as a prompt-ready markdown section.

    Explains to Claude how the original 80-program pool was constructed
    (10 quality-evaluation heuristics × 8 variants per heuristic) and what
    each heuristic measures.

    The returned string has NO leading whitespace on any line. This keeps
    things clean when it is interpolated into a parent ``textwrap.dedent``
    block — interpolating a partially-indented block would otherwise leak
    into the parent's common-leading-whitespace calculation and produce
    inconsistent rendering.
    """
    total = len(HEURISTICS) * PROGRAMS_PER_HEURISTIC

    heuristic_lines = []
    for h_id, h in HEURISTICS.items():
        heuristic_lines.append(
            f"  {h_id}. **{h['name']}** — {h['description']}")
    heuristic_list_text = "\n".join(heuristic_lines)

    return (
        "## How the Input Programs Were Generated\n"
        "\n"
        "The original judge program pool (from which the input programs below\n"
        "were selected) was generated by an LLM following a fixed\n"
        "single-dimension blueprint:\n"
        "\n"
        f"- **10 quality-evaluation heuristics × {PROGRAMS_PER_HEURISTIC} variants per "
        f"heuristic = {total} programs total.**\n"
        "- Each program is intentionally NARROW: it focuses STRICTLY on ONE\n"
        "  heuristic dimension. Variants within a heuristic use meaningfully\n"
        "  DIFFERENT algorithms, features, and scoring formulas but target the\n"
        "  same dimension.\n"
        "- Programs are pure Python with only ``re``, ``math``, ``collections``,\n"
        "  ``string``, ``statistics`` allowed (no ML/NLP libraries).\n"
        "- The per-program \"Designed for heuristic\" tag below tells you which\n"
        "  of the 10 dimensions each selected input program targets.\n"
        "\n"
        "The 10 heuristics (this is the canonical taxonomy — same across all\n"
        "datasets):\n"
        "\n"
        f"{heuristic_list_text}\n"
        "\n"
        "Implications for your aggregation:\n"
        "- Each selected input program is, by design, a SINGLE-DIMENSION scorer.\n"
        "  It is intentionally NOT trying to capture all aspects of quality.\n"
        f"- The top-k subset was filtered from the {total}-program pool based on\n"
        "  validation performance, which may leave some heuristics over- or\n"
        "  under-represented in the subset you see.\n"
        "- When designing aggregated programs, consider: (1) which heuristics\n"
        "  are present vs missing among the inputs, (2) how to FUSE multiple\n"
        "  heuristics into a multi-dimensional judge (e.g. relevance × clarity ×\n"
        "  specificity), and (3) how cross-heuristic interactions matter (a\n"
        "  response that is relevant but poorly structured may still be the\n"
        "  better choice)."
    )


def _order_programs(program_sources, summary, info_mode):
    """Return list of (name, source) ordered consistently with info_mode.

    - 'acc' / 'cov': sort by validation accuracy descending.
    - 'code' / 'val-samples': sort alphabetically by program name.

    Note: 'val-samples' uses alphabetical ordering so that Claude cannot
    infer per-program accuracy from position (since this mode hides
    per-program acc/cov metadata entirely).
    """
    name_to_idx = {n: idx for idx, n in enumerate(summary["selected_program_names"])}
    accs = summary.get("selected_program_val_accuracies", [])

    def _acc(name):
        idx = name_to_idx.get(name)
        if idx is None or idx >= len(accs):
            return float("-inf")
        return float(accs[idx])

    items = list(program_sources.items())
    if info_mode in ("code", "val-samples"):
        items.sort(key=lambda kv: kv[0])
    else:
        items.sort(key=lambda kv: _acc(kv[0]), reverse=True)
    return items


def find_wrong_val_samples(pipeline_outputs_dir, val_jsonl_path, n=10, seed=42):
    """Find validation samples where the LabelModel prediction was wrong.

    Uses Y_hat_val.json (LabelModel predictions on val) from the pipeline
    outputs and compares with ground-truth labels from the val JSONL.

    Returns a list of dicts, each containing the sample's text fields,
    ground-truth label, and the LabelModel's (wrong) prediction.
    """
    y_hat_path = os.path.join(pipeline_outputs_dir, "Y_hat_val.json")
    if not os.path.exists(y_hat_path):
        print(f"  WARNING: {y_hat_path} not found, cannot load wrong samples")
        return []

    with open(y_hat_path) as f:
        y_hat_val = json.load(f)

    rows = []
    labels = []
    with open(val_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            labels.append(_extract_label(row))

    if len(y_hat_val) != len(labels):
        print(f"  WARNING: Y_hat_val length ({len(y_hat_val)}) != "
              f"val JSONL length ({len(labels)}). Cannot align.")
        return []

    wrong_indices = [i for i in range(len(labels))
                     if y_hat_val[i] != labels[i]]
    if not wrong_indices:
        print("  No wrong samples found (LabelModel was 100% correct on val).")
        return []

    rng = random.Random(seed)
    sampled = rng.sample(wrong_indices, min(n, len(wrong_indices)))

    results = []
    for idx in sampled:
        query, ans1, ans2 = _extract_fields(rows[idx])
        gt = labels[idx]
        pred = y_hat_val[idx]
        results.append({
            "idx": idx,
            "query": query[:500],
            "response_a": ans1[:600],
            "response_b": ans2[:600],
            "ground_truth": "Response A is better" if gt == 0
                            else "Response B is better",
            "lm_prediction": "Response A is better" if pred == 0
                             else "Response B is better",
        })

    return results


def find_correct_val_samples(pipeline_outputs_dir, val_jsonl_path, summary,
                             n=5, seed=42):
    """Find high-confidence correct validation samples.

    Picks samples where: (1) LabelModel prediction == ground truth, and
    (2) the selected programs have high mean |diff| (confident agreement).
    These serve as "patterns to preserve" in the denoising prompt.
    """
    y_hat_path = os.path.join(pipeline_outputs_dir, "Y_hat_val.json")
    val_diffs_path = os.path.join(pipeline_outputs_dir, "val_diffs.json")
    if not os.path.exists(y_hat_path) or not os.path.exists(val_diffs_path):
        return []

    with open(y_hat_path) as f:
        y_hat_val = json.load(f)
    with open(val_diffs_path) as f:
        val_diffs = np.array(json.load(f))

    rows, labels = [], []
    with open(val_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            labels.append(_extract_label(row))

    if len(y_hat_val) != len(labels):
        return []

    col_indices = summary.get("selected_program_col_indices", [])
    if not col_indices or val_diffs.shape[1] <= max(col_indices):
        return []

    selected_diffs = val_diffs[:, col_indices]
    confidence = np.nanmean(np.abs(selected_diffs), axis=1)

    correct_with_conf = [(i, float(confidence[i]))
                         for i in range(len(labels))
                         if y_hat_val[i] == labels[i]]
    if not correct_with_conf:
        return []

    correct_with_conf.sort(key=lambda x: x[1], reverse=True)
    top_pool = correct_with_conf[:max(n, len(correct_with_conf) // 4)]

    rng = random.Random(seed)
    sampled = rng.sample(top_pool, min(n, len(top_pool)))

    results = []
    for idx, conf in sampled:
        query, ans1, ans2 = _extract_fields(rows[idx])
        gt = labels[idx]
        results.append({
            "idx": idx,
            "query": query[:500],
            "response_a": ans1[:600],
            "response_b": ans2[:600],
            "ground_truth": ("Response A is better" if gt == 0
                             else "Response B is better"),
            "confidence": conf,
        })
    return results


def format_denoising_context(wrong_samples, correct_samples, n_programs):
    """Build the full coverage-preserving denoising prompt for val-samples mode.

    Includes: denoising objective, score calibration, ensemble role design,
    failure cases (with anti-abstention framing), high-confidence correct
    cases (patterns to preserve), and forbidden failure modes.
    """
    if not wrong_samples and not correct_samples:
        return ""

    # ── Ensemble role design (adaptive to N) ──
    if n_programs < 4:
        role_text = (
            f"Each of the {n_programs} programs should balance broad coverage "
            f"with targeted denoising. Every program should vote on most "
            f"ordinary examples while incorporating targeted corrections "
            f"for known failure modes."
        )
    else:
        n_broad = max(2, round(n_programs * 0.6))
        n_denoise = n_programs - n_broad
        role_text = (
            f"- Programs 1-{n_broad}: Coverage-preserving broad scorers.\n"
            f"  Vote on most ordinary examples with broad, general quality "
            f"signals.\n"
            f"  Include targeted gates against known failure modes but do "
            f"NOT become conservative.\n"
            f"- Programs {n_broad + 1}-{n_programs}: Denoising specialists.\n"
            f"  Target systematic failure patterns from the validation "
            f"errors below.\n"
            f"  Correct misleading broad heuristics — do NOT merely "
            f"abstain.\n\n"
            f"The final set should combine: broad coverage + targeted "
            f"denoising + diverse error patterns + calibrated score margins."
        )

    # ── Format failure cases ──
    wrong_parts = []
    for i, s in enumerate(wrong_samples, 1):
        wrong_parts.append(
            f"--- Failure Case {i} (val index {s['idx']}) ---\n"
            f"Query: {s['query']}\n\n"
            f"Response A:\n{s['response_a']}\n\n"
            f"Response B:\n{s['response_b']}\n\n"
            f"Ground truth: {s['ground_truth']}\n"
            f"Pipeline predicted: {s['lm_prediction']}  <- WRONG"
        )
    wrong_text = "\n\n".join(wrong_parts) if wrong_parts else "(none available)"

    # ── Format correct cases ──
    correct_parts = []
    for i, s in enumerate(correct_samples, 1):
        correct_parts.append(
            f"--- Correct Case {i} (val index {s['idx']}) ---\n"
            f"Query: {s['query']}\n\n"
            f"Response A:\n{s['response_a']}\n\n"
            f"Response B:\n{s['response_b']}\n\n"
            f"Ground truth: {s['ground_truth']}\n"
            f"Pipeline predicted: {s['ground_truth']}  <- CORRECT "
            f"(high confidence)"
        )
    correct_text = ("\n\n".join(correct_parts)
                    if correct_parts else "(none available)")

    return textwrap.dedent(f"""\

    ## Main Objective: Coverage-Preserving Denoising

    Your job is to synthesize stronger programs that DENOISE the original
    top-k while PRESERVING broad coverage.

    Critical constraint:
        Do NOT improve accuracy by becoming conservative.
        Coverage collapse is considered a FAILURE.

    Denoising means:
    - Identify misleading heuristic patterns from the failure cases below
    - Downweight those patterns ONLY when context indicates they are misleading
    - Keep the same heuristic active when it is usually useful
    - Preserve useful score margins so downstream thresholding can still vote

    Denoising does NOT mean:
    - Shrinking most scores toward a single neutral midpoint
    - Requiring extremely strong evidence before assigning a meaningful score
    - Abstaining on ordinary cases
    - Deleting broad heuristics because they failed in a few examples

    ## Score Calibration

    Your scores must SPREAD across responses of different quality so that
    higher-quality responses receive clearly higher scores than lower-quality
    ones. The exact numeric range you use does not matter — use whatever
    range is natural for your heuristic — but the SPREAD across responses
    does matter.

    Differentiate strongly:
    - Clearly excellent responses should receive a high score
    - Clearly poor responses should receive a low score
    - Mediocre / borderline responses fall in between, but should still
      be distinguishable from one another when there is a real quality
      difference

    Do NOT collapse most scores to a narrow band around a single neutral
    value. A program that is accurate only because it is neutral on many
    samples is BAD — it cannot discriminate.

    ## Ensemble Role Design

    {role_text}

    ## Validation Failure Cases

    The following {len(wrong_samples)} examples are where the pipeline
    predicted INCORRECTLY.

    IMPORTANT: These are NOT abstention examples. For many of them, the
    correct behavior is to still vote, but in the CORRECT direction by
    recognizing which heuristic was misleading.

    For each failure case, infer:
    1. What signal misled the original programs?
    2. What stronger signal should have overridden it?
    3. How can this correction be a general rule (not memorization)?
    4. How can this rule preserve coverage on ordinary cases?

    {wrong_text}

    ## High-Confidence Correct Cases

    The following {len(correct_samples)} examples show where the pipeline
    was CORRECT with high confidence. These patterns should be PRESERVED.

    When fixing failure cases, do NOT overcorrect in ways that break
    these successful patterns. The goal is targeted denoising, not global
    caution.

    {correct_text}

    ## Forbidden Failure Mode

    Do NOT globally reduce confidence. The following are FAILURES:
    - Returning the same neutral midpoint score for many normal responses
    - Requiring too many conditions before giving meaningful scores
    - Deleting broad heuristics because they appeared in one failure case
    - Treating all failure examples as reasons to abstain

    The correct approach:
    - Keep broad useful heuristics
    - Add targeted gates for known misleading patterns
    - Preserve score variance and coverage
    - A 1% accuracy gain is NOT worth a large coverage drop
    """)


def build_prompt(program_sources, n_programs, summary, program_val_stats,
                 info_mode="cov", n_total_programs=None,
                 denoising_context=""):
    """Build the prompt that asks Claude to synthesize aggregated programs.

    Information levels (controlled by ``info_mode``):
      - 'code'       : only the source code of each input program (no metadata).
      - 'acc'        : source code + per-program validation accuracy.
      - 'cov'        : source code + per-program validation accuracy + coverage.
      - 'val-samples': source code ONLY + coverage-preserving denoising
        context (failure cases, correct cases, ensemble role design,
        score calibration). Per-program acc/coverage are NOT shown so
        Claude must rely on the actual failure/correct cases rather than
        per-program performance numbers.

    Claude is never told the per-program threshold, score-diff statistics,
    or s1/s2 distributions in any mode.

    Programs are ordered so the prompt is internally consistent:
      - 'acc'/'cov'           : by validation accuracy descending (told to Claude).
      - 'code'/'val-samples'  : alphabetically (we do NOT claim any acc ordering).
    """
    name_to_idx = {n: idx for idx, n in enumerate(summary["selected_program_names"])}
    accs = summary.get("selected_program_val_accuracies", [])

    ordered_items = _order_programs(program_sources, summary, info_mode)

    programs_block = []
    for i, (name, source) in enumerate(ordered_items, 1):
        meta_lines = []
        h_id, h = _heuristic_for_judge_name(name)
        if h is not None:
            meta_lines.append(
                f"Designed for heuristic {h_id}: {h['name']}")
        if info_mode in ("acc", "cov"):
            orig_idx = name_to_idx.get(name)
            acc = float(accs[orig_idx]) if (orig_idx is not None
                                            and orig_idx < len(accs)) else 0.0
            meta_lines.append(f"Validation accuracy: {acc:.4f}")
        if info_mode == "cov":
            stats = program_val_stats.get(name, {})
            if stats:
                meta_lines.append(
                    f"Validation coverage: {stats['coverage']:.2%} "
                    f"(non-abstain rate on {stats['n_total']} val samples)")
        meta_block = ("\n".join(meta_lines) + "\n") if meta_lines else ""

        programs_block.append(
            f"### Program {i}: {name}\n"
            f"{meta_block}"
            f"```python\n{source}\n```"
        )
    programs_text = "\n\n".join(programs_block)

    if info_mode != "code":
        lm_val = summary.get("LabelModel_val", {})
        pipeline_context = (
            f"Pipeline baseline performance on validation set "
            f"(n={lm_val.get('n_total', '?')}, "
            f"coverage={lm_val.get('coverage', '?')}):\n"
            f"  - LabelModel val accuracy: {lm_val.get('accuracy', '?')}"
        )
    else:
        pipeline_context = ""

    if info_mode in ("code", "val-samples"):
        if n_total_programs:
            selection_phrase = (
                f"I selected {len(program_sources)} programs "
                f"(from {n_total_programs} total)."
            )
        else:
            selection_phrase = (
                f"I selected {len(program_sources)} programs."
            )
    else:
        if n_total_programs:
            selection_phrase = (
                f"I selected the top {len(program_sources)} programs "
                f"(from {n_total_programs} total) based on their "
                f"validation accuracy."
            )
        else:
            selection_phrase = (
                f"I selected the top {len(program_sources)} programs "
                f"based on their validation accuracy."
            )

    metadata_lines = [
        "- **Source code** — the actual Python judge program",
        "- **Designed for heuristic <id>: <name>** — which of the 10 "
        "single-dimension heuristics this program targets (see blueprint "
        "section above)",
    ]
    if info_mode in ("acc", "cov"):
        metadata_lines.append(
            "- **Validation accuracy** — accuracy on non-abstained validation samples")
    if info_mode == "cov":
        metadata_lines.append(
            "- **Validation coverage** — fraction of validation samples this "
            "program votes on (rather than abstaining)")
    metadata_desc = "\n    ".join(metadata_lines)

    think_lines = []
    if info_mode in ("acc", "cov"):
        think_lines.append(
            "- Which programs have the highest accuracy and WHY their heuristics work")
    if info_mode == "cov":
        think_lines.append(
            "- Which programs have high coverage vs. low coverage, and how a "
            "high-accuracy/low-coverage program may complement a "
            "moderate-accuracy/high-coverage one in an ensemble")
    if info_mode == "val-samples":
        think_lines.append(
            "- What patterns the failure cases reveal and how to fix them "
            "WITHOUT reducing coverage on ordinary cases")
        think_lines.append(
            "- Which heuristics to preserve (shown in the correct cases) vs. "
            "which misleading signals to gate/suppress (shown in the "
            "failure cases)")
    if info_mode in ("code", "val-samples"):
        think_lines.append(
            "- What each program's logic actually measures (read the code carefully)")
        think_lines.append(
            "- Which heuristics seem brittle vs. robust, and how to combine them "
            "so weaknesses cancel out")
    think_lines.append(
        "- What specific heuristics (keyword overlap, sentence structure, "
        "information density, etc.) are most predictive of quality differences")
    think_text = "\n    ".join(think_lines)

    if info_mode in ("acc", "cov"):
        order_clause = "ordered by validation accuracy (highest first)"
    else:
        order_clause = "ordered alphabetically by name"

    generation_blueprint = _format_generation_blueprint()

    prompt = textwrap.dedent(f"""\
    # Task: Synthesize {len(program_sources)} Judge Programs into {n_programs} Optimal Aggregated Programs

    ## Context

    I have a "Program as a Judge" system for evaluating LLM response quality.
    Given a (query, response1, response2) triple, each judge program independently
    scores response1 and response2. The score difference determines which
    response is preferred:
    - If score_diff > threshold → response1 wins
    - If score_diff < -threshold → response2 wins
    - Otherwise → abstain (the program is not confident enough to vote)

    {selection_phrase} These programs use diverse heuristics: relevance overlap,
    readability indices, factual indicators, clarity metrics, structural analysis,
    etc.

    {pipeline_context}

{generation_blueprint}

    ## Your Task

    You are given **exactly one parameter**: synthesize these {len(program_sources)}
    input programs into **exactly {n_programs} new aggregated programs**.

    Below you will find each input program's:
    {metadata_desc}

    Study ALL of this information carefully. Think deeply about:
    {think_text}

    Then autonomously design the **optimal {n_programs} aggregated programs** that
    combine the best heuristics. You decide the strategy for each program — there
    are no pre-assigned dimensions or roles.

    ## Requirements

    1. **Follow the exact interface**: `def judging_function(query, response):`
       returning a float score (higher = better). The numeric range is up to
       you — pick whatever is natural for your heuristic.
    2. **Be pure Python** — only `re`, `math`, `collections`, `string` allowed.
       No ML models, no numpy, sklearn, etc.
    3. **Be robust** — wrap everything in try/except, never crash, return a neutral
       score on error
    4. **Be diverse from each other** — each aggregated program should capture
       different complementary signals so that when they vote together, they
       maximize ensemble accuracy
    5. **Be substantive — target ~200 lines per program (roughly 150-250
       lines of Python).** Each aggregated program should be a substantial
       standalone judge that thoughtfully combines multiple heuristics,
       gating logic, edge-case handling, and score calibration. Do NOT
       shorten programs just because you are asked to generate more of
       them — per-program complexity should NOT scale down with N.
       Programs that are only ~50-100 lines are too thin to denoise
       effectively. Each program independently deserves the same level
       of depth regardless of the total number requested.

    ## Output Format

    Output each program in a clearly delimited block:

    ```
    === AGGREGATED_PROGRAM_1 ===
    [description: what strategy this program uses, which input programs inspired it, and why]

    ```python
    def judging_function(query, response):
        ...
    ```

    === AGGREGATED_PROGRAM_2 ===
    ...
    ```

    ## Input Programs

    Below are the {len(program_sources)} selected programs, {order_clause}:

    {programs_text}
    """)

    if denoising_context:
        prompt += denoising_context

    return prompt


def call_claude(prompt, model, max_output_tokens, effort="xhigh"):
    """Call Claude API with adaptive thinking.

    Opus 4.7 API structure:
      - thinking: {"type": "adaptive", "display": "summarized"}
      - output_config: {"effort": "xhigh"}   ← effort lives HERE, not in thinking
      - Manual budget_tokens is NOT supported on Opus 4.7 (returns 400)

    Uses streaming (required by SDK when max_tokens > 21333).
    Collects response manually from stream events for robustness.

    Returns (thinking_text, output_text, stop_reason).
    """
    client = anthropic.Anthropic()

    print(f"  Calling {model} ...")
    print(f"    max_tokens (total ceiling):  {max_output_tokens:,}")
    print(f"    thinking mode:               adaptive")
    print(f"    effort:                      {effort}")
    print(f"    thinking display:            summarized")

    with client.messages.stream(
        model=model,
        max_tokens=max_output_tokens,
        thinking={
            "type": "adaptive",
            "display": "summarized",
        },
        output_config={
            "effort": effort,
        },
        messages=[
            {"role": "user", "content": prompt}
        ],
    ) as stream:
        response = stream.get_final_message()

    thinking_text = ""
    output_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text += (block.thinking or "") + "\n"
        elif block.type == "text":
            output_text += block.text
        elif block.type == "redacted_thinking":
            thinking_text += "[REDACTED]\n"

    usage = response.usage
    input_tok = usage.input_tokens
    output_tok = usage.output_tokens
    stop_reason = response.stop_reason

    print(f"\n  API response:")
    print(f"    Input tokens:  {input_tok:,}")
    print(f"    Output tokens: {output_tok:,}  (billed: full thinking + text)")
    print(f"    Stop reason:   {stop_reason}")
    print(f"    Thinking summary length: {len(thinking_text):,} chars")
    print(f"    Text output length:      {len(output_text):,} chars")

    cost_in = input_tok / 1_000_000 * 5.0
    cost_out = output_tok / 1_000_000 * 25.0
    print(f"    Est. cost:     ${cost_in:.2f} (input) + ${cost_out:.2f} (output) "
          f"= ${cost_in + cost_out:.2f}")

    return thinking_text, output_text, stop_reason


def parse_programs(output_text, n_expected):
    """Parse generated programs using a line-by-line state machine.

    Handles code that contains backtick patterns (e.g. regex matching ```)
    by only treating a line as a closing fence if it starts with ``` and is
    NOT indented (real code fences are always at column 0).
    """
    programs = {}
    lines = output_text.split("\n")
    i = 0

    current_id = None
    in_code = False
    code_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        header = re.match(r'===\s*AGGREGATED_PROGRAM_(\d+)\s*===', stripped)
        if header:
            if current_id is not None and code_lines:
                programs[f"aggregated_judge_{current_id}"] = \
                    "\n".join(code_lines).strip()
            current_id = int(header.group(1))
            in_code = False
            code_lines = []
            i += 1
            continue

        if not in_code and stripped.startswith("```python"):
            in_code = True
            code_lines = []
            i += 1
            continue

        if in_code:
            if re.match(r'^```\s*$', line):
                in_code = False
                if current_id is not None:
                    programs[f"aggregated_judge_{current_id}"] = \
                        "\n".join(code_lines).strip()
                    code_lines = []
            else:
                code_lines.append(line)

        i += 1

    if current_id is not None and code_lines:
        programs[f"aggregated_judge_{current_id}"] = \
            "\n".join(code_lines).strip()

    if not programs:
        fallback = re.findall(
            r'```python\s*\n(def judging_function.*?)(?:\n```|\Z)',
            output_text, re.DOTALL)
        for idx, code in enumerate(fallback, 1):
            programs[f"aggregated_judge_{idx}"] = code.strip()

    if not programs:
        print("  WARNING: Could not parse any programs from output.")
        print("  Raw output (first 2000 chars):")
        print(output_text[:2000])

    return programs


def load_existing_programs(output_dir, n_programs):
    """Scan output_dir for existing aggregated_judge_*.py files.
    Returns dict of {name: code} for programs that pass basic syntax check."""
    existing = {}
    for prog_id in range(1, n_programs + 1):
        name = f"aggregated_judge_{prog_id}"
        path = os.path.join(output_dir, f"{name}.py")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            code = f.read().strip()
        if not code:
            continue
        try:
            import ast
            ast.parse(code)
            if "def judging_function" not in code:
                print(f"  {name}: exists but missing judging_function, will regenerate")
                continue
            existing[name] = code
        except SyntaxError:
            print(f"  {name}: exists but has syntax errors (truncated?), will regenerate")
    return existing


def build_resume_prompt(missing_ids, completed_programs, program_sources,
                        n_programs, summary, program_val_stats,
                        info_mode="cov", n_total_programs=None,
                        denoising_context=""):
    """Build a prompt asking Claude to generate ONLY the missing programs.

    Mirrors ``build_prompt`` exactly in terms of what metadata is exposed —
    controlled by ``info_mode`` — so that the resume round receives the same
    information level as the initial round (no silent information loss).

    For val-samples mode the same denoising context (failure cases, correct
    cases, ensemble roles, score calibration) is appended.

    Includes already-completed programs as context for diversity.
    """
    completed_block = ""
    if completed_programs:
        parts = []
        for name, code in sorted(completed_programs.items()):
            parts.append(
                f"### {name} (ALREADY COMPLETED — do NOT regenerate)\n"
                f"```python\n{code}\n```"
            )
        completed_block = (
            "## Already Completed Programs\n\n"
            "The following programs have ALREADY been generated and validated. "
            "Do NOT regenerate them. They are shown here so you can ensure "
            "the new programs are DIVERSE and COMPLEMENTARY to these.\n\n"
            + "\n\n".join(parts)
        )

    missing_nums = sorted(missing_ids)
    missing_str = ", ".join(str(i) for i in missing_nums)

    name_to_idx = {n: idx for idx, n in enumerate(summary["selected_program_names"])}
    accs = summary.get("selected_program_val_accuracies", [])
    ordered_items = _order_programs(program_sources, summary, info_mode)

    input_block = []
    for i, (name, source) in enumerate(ordered_items, 1):
        meta_lines = []
        h_id, h = _heuristic_for_judge_name(name)
        if h is not None:
            meta_lines.append(
                f"Designed for heuristic {h_id}: {h['name']}")
        if info_mode in ("acc", "cov"):
            orig_idx = name_to_idx.get(name)
            acc = float(accs[orig_idx]) if (orig_idx is not None
                                            and orig_idx < len(accs)) else 0.0
            meta_lines.append(f"Validation accuracy: {acc:.4f}")
        if info_mode == "cov":
            stats = program_val_stats.get(name, {})
            if stats:
                meta_lines.append(
                    f"Validation coverage: {stats['coverage']:.2%} "
                    f"(non-abstain rate on {stats['n_total']} val samples)")
        meta_block = ("\n".join(meta_lines) + "\n") if meta_lines else ""

        input_block.append(
            f"### Program {i}: {name}\n"
            f"{meta_block}"
            f"```python\n{source}\n```"
        )
    input_text = "\n\n".join(input_block)

    missing_descs = []
    for mid in missing_nums:
        missing_descs.append(
            f"  - AGGREGATED_PROGRAM_{mid}: design an optimal, unique strategy "
            f"complementary to the already-completed programs")
    missing_desc_text = "\n".join(missing_descs)

    if info_mode == "code":
        study_hint = (
            "Read each input program's source code carefully, and use the "
            "per-program heuristic tag (which of the 10 generation "
            "dimensions it targets) to decide the best strategy for each "
            "missing program."
        )
    elif info_mode == "acc":
        study_hint = (
            "Use each input program's source code, its heuristic tag, and "
            "validation accuracy to decide the best strategy for each missing "
            "program."
        )
    elif info_mode == "val-samples":
        study_hint = (
            "Read each input program's source code carefully, use its "
            "heuristic tag (which of the 10 generation dimensions it "
            "targets), AND use the coverage-preserving denoising context "
            "below (failure cases, correct cases, ensemble roles, score "
            "calibration) to decide the best strategy for each missing "
            "program. Fix systematic failure modes while preserving broad "
            "coverage."
        )
    else:
        study_hint = (
            "Use each input program's source code, its heuristic tag, "
            "validation accuracy, and validation coverage to decide the "
            "best strategy for each missing program."
        )

    if n_total_programs:
        from_clause = f" (from {n_total_programs} total)"
    else:
        from_clause = ""

    if info_mode in ("acc", "cov"):
        order_clause = "ordered by validation accuracy (highest first)"
    else:
        order_clause = "ordered alphabetically by name"

    generation_blueprint = _format_generation_blueprint()

    prompt = textwrap.dedent(f"""\
    # Task: Generate Missing Aggregated Judge Programs

    I am synthesizing {n_programs} aggregated judge programs from {len(program_sources)}
    input programs{from_clause}. Some programs were already generated successfully,
    but the following program(s) still need to be generated: **{missing_str}**

{generation_blueprint}

    {completed_block}

    ## Programs to Generate

    Generate ONLY the following program(s). Use the same output format:

    {missing_desc_text}

    {study_hint} Make them complementary to the already-completed programs.

    ## Requirements (same as before)

    - Each program MUST be a complete, self-contained Python function
    - Each program MUST start with `def judging_function(query, response):`
    - All imports MUST be inside the function body (re, math, collections, string only)
    - The function MUST return a float (higher = better); the numeric range
      is up to you — pick whatever is natural for your heuristic
    - Be robust: wrap in try/except, never crash
    - Be DIVERSE from the already-completed programs above
    - Be SUBSTANTIVE — target ~200 lines per program (roughly 150-250 lines).
      Per-program complexity must NOT scale down with N — each program
      independently deserves the same level of depth as if it were the only
      one being generated. Combine multiple heuristics, gating logic, and
      edge-case handling. Programs of only ~50-100 lines are too thin.

    ## Output Format

    ```
    === AGGREGATED_PROGRAM_{missing_nums[0]} ===
    [description]

    ```python
    def judging_function(query, response):
        ...
    ```
    ```

    ## Input Programs (for reference, {order_clause})

    {input_text}
    """)

    if denoising_context:
        prompt += denoising_context

    return prompt


def save_programs(programs, output_dir, tag):
    """Save each aggregated program as a separate .py file."""
    os.makedirs(output_dir, exist_ok=True)

    for name, code in programs.items():
        path = os.path.join(output_dir, f"{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n")
        print(f"  Saved: {path}")

    manifest = {
        "tag": tag,
        "n_programs": len(programs),
        "programs": list(programs.keys()),
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {manifest_path}")


def validate_programs(programs):
    """Validate that each program can be loaded and called."""
    import importlib.util
    import tempfile

    results = {}
    test_query = "What is the capital of France?"
    test_response = "The capital of France is Paris, a city known for its culture."

    for name, code in programs.items():
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                             delete=False) as f:
                f.write(code)
                tmp_path = f.name

            spec = importlib.util.spec_from_file_location(name, tmp_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            score = float(mod.judging_function(test_query, test_response))
            results[name] = {"valid": True, "test_score": score}
            print(f"  {name}: valid (test_score={score:.2f})")

            os.unlink(tmp_path)
        except Exception as e:
            results[name] = {"valid": False, "error": str(e)}
            print(f"  {name}: INVALID — {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return results


def load_programs_as_funcs(programs):
    """Load program code strings into callable functions."""
    import importlib.util
    import tempfile

    loaded = {}
    for name, code in programs.items():
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                             delete=False) as f:
                f.write(code)
                tmp_path = f.name
            spec = importlib.util.spec_from_file_location(name, tmp_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            loaded[name] = mod.judging_function
            os.unlink(tmp_path)
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return loaded


def _parse_pandalm_input_sequence(input_seq):
    """Parse PandaLM input_sequence to extract query, response1, response2."""
    instruction = ""
    extra_input = ""
    response1 = ""
    response2 = ""

    parts = re.split(
        r'### (Instruction|Input|Response 1|Response 2|Evaluation):\s*\n?',
        input_seq)
    for i, part in enumerate(parts):
        if part == "Instruction" and i + 1 < len(parts):
            instruction = parts[i + 1].strip()
        elif part == "Input" and i + 1 < len(parts):
            extra_input = parts[i + 1].strip()
        elif part == "Response 1" and i + 1 < len(parts):
            response1 = parts[i + 1].strip()
        elif part == "Response 2" and i + 1 < len(parts):
            response2 = parts[i + 1].strip()

    query = instruction
    if extra_input:
        query = f"{instruction}\n\nInput: {extra_input}"
    return query, response1, response2


def _extract_fields(row):
    """Extract (query, ans1, ans2) from a dataset row, auto-detecting format."""
    if "input_sequence" in row:
        return _parse_pandalm_input_sequence(row["input_sequence"])
    elif "instruction" in row and "response1" in row:
        instruction = row.get("instruction", "")
        extra_input = row.get("input", "")
        query = instruction
        if extra_input:
            query = f"{instruction}\n\nInput: {extra_input}"
        return query, row.get("response1", ""), row.get("response2", "")
    else:
        return (row.get("question_body", ""),
                row.get("answer1_body", ""),
                row.get("answer2_body", ""))


def _extract_label(row):
    """Extract binary label from a dataset row, auto-detecting format.
    Returns 0 if response1 wins, 1 if response2 wins."""
    if "output_sequence" in row:
        verdict = row["output_sequence"].strip().split("\n")[0].strip()
        return 0 if verdict == "1" else 1
    elif "annotator1" in row:
        from statistics import mode as stat_mode
        votes = [row["annotator1"], row["annotator2"], row["annotator3"]]
        majority = stat_mode(votes)
        return 0 if majority == 1 else 1
    else:
        s1, s2 = float(row["score"][0]), float(row["score"][1])
        if s1 > s2:
            return 0
        elif s2 > s1:
            return 1
        else:
            return np.random.choice([0, 1])


def load_jsonl(path):
    """Load JSONL dataset. Auto-detects format (standard / PandaLM).
    Returns (data_rows, labels_array)."""
    data, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            data.append(row)
            labels.append(_extract_label(row))
    return data, np.array(labels)


def score_dataset(data, func_list, func_names, split_name="data"):
    """Score every (query, ans1) and (query, ans2) pair. Returns s1_mat, s2_mat."""
    from tqdm import tqdm

    n, m = len(data), len(func_list)
    s1_mat = np.full((n, m), np.nan)
    s2_mat = np.full((n, m), np.nan)

    for i, row in tqdm(enumerate(data), total=n, desc=f"  Scoring {split_name}"):
        query, ans1, ans2 = _extract_fields(row)
        for j, fn in enumerate(func_list):
            try:
                s1_mat[i, j] = float(fn(query, ans1))
                s2_mat[i, j] = float(fn(query, ans2))
            except Exception:
                pass

    return s1_mat, s2_mat


def compute_norm_params(s1_mat, s2_mat):
    """Per-program robust min-max bounds (P1/P99) from the given score matrices."""
    m = s1_mat.shape[1]
    params = []
    for j in range(m):
        pooled = np.concatenate([
            s1_mat[:, j][~np.isnan(s1_mat[:, j])],
            s2_mat[:, j][~np.isnan(s2_mat[:, j])],
        ])
        if len(pooled) > 1 and (np.max(pooled) - np.min(pooled)) > 0:
            lo = float(np.percentile(pooled, 1))
            hi = float(np.percentile(pooled, 99))
            if hi <= lo:
                lo, hi = float(pooled.min()), float(pooled.max())
        else:
            lo, hi = 0.0, 1.0
        params.append((lo, hi))
    return params


def normalize_and_diff(s1_mat, s2_mat, norm_params):
    """Normalize each program's scores to [0,1], then compute diff in [-1,1]."""
    n, m = s1_mat.shape
    diffs = np.full((n, m), np.nan)
    for j in range(m):
        lo, hi = norm_params[j]
        rng = hi - lo
        if rng > 0:
            n1 = np.clip((s1_mat[:, j] - lo) / rng, 0.0, 1.0)
            n2 = np.clip((s2_mat[:, j] - lo) / rng, 0.0, 1.0)
        else:
            n1 = np.full(n, 0.5)
            n2 = np.full(n, 0.5)
        diffs[:, j] = n1 - n2
        mask = np.isnan(s1_mat[:, j]) | np.isnan(s2_mat[:, j])
        diffs[mask, j] = np.nan
    return diffs


def tune_thresholds(diffs, labels, func_names, threshold_max=0.15):
    """Per-program threshold tuning on a labelled set (val).

    Returns ``{program_name: {accuracy, best_threshold, coverage}}``.
    """
    candidates = np.arange(0.0, threshold_max + 0.005, 0.01)
    n, m = diffs.shape
    results = {}

    for j, name in enumerate(func_names):
        d = diffs[:, j]
        best_acc, best_t = 0.0, 0.0
        for t in candidates:
            votes = np.full(n, -1, dtype=int)
            votes[d > t] = 0
            votes[d < -t] = 1
            votes[np.isnan(d)] = -1
            valid = votes != -1
            if valid.sum() == 0:
                continue
            a = accuracy_score(labels[valid], votes[valid])
            if a > best_acc or (a == best_acc and t > best_t):
                best_acc, best_t = a, t

        coverage = float((~np.isnan(d) & (np.abs(d) > best_t)).mean())
        results[name] = {
            "accuracy": round(best_acc, 4),
            "best_threshold": round(best_t, 2),
            "coverage": round(coverage, 4),
        }
        print(f"    {name}: acc={best_acc:.4f} (T={best_t:.2f}, cov={coverage:.2%})")

    return results


def apply_and_evaluate(diffs, labels, func_names, thresholds_dict, split_name="test"):
    """Apply FIXED per-program thresholds (from val) and report per-program
    accuracy / coverage on the given split.

    This is a *diagnostic* helper.  Aggregate evaluation (Snorkel LabelModel +
    MajorityLabelVoter) is handled separately by :func:`snorkel_evaluate`.

    Returns ``{program_name: {accuracy, threshold_from_val, coverage}}``.
    """
    n, m = diffs.shape
    results = {}

    for j, name in enumerate(func_names):
        d = diffs[:, j]
        t = thresholds_dict[name]["best_threshold"]

        votes = np.full(n, -1, dtype=int)
        votes[d > t] = 0
        votes[d < -t] = 1
        votes[np.isnan(d)] = -1

        valid = votes != -1
        if valid.sum() > 0:
            acc = accuracy_score(labels[valid], votes[valid])
            cov = float(valid.mean())
        else:
            acc, cov = 0.0, 0.0

        results[name] = {
            "accuracy": round(acc, 4),
            "threshold_from_val": round(t, 2),
            "coverage": round(cov, 4),
        }
        print(f"    {name}: acc={acc:.4f} (T={t:.2f} from val, cov={cov:.2%})")

    return results


# ── Snorkel-based aggregation evaluation ───────────────────────────────────

def build_label_matrix_from_diffs(diffs, func_names, thresholds_dict):
    """Convert an (n, m) diff matrix into a Snorkel-style int label matrix
    using the per-program tuned thresholds from ``thresholds_dict``.

    Values in {-1, 0, 1}:
        0   → response 1 wins   (diff >  t)
        1   → response 2 wins   (diff < -t)
        -1  → abstain           (|diff| <= t, or NaN)

    Mirrors ``build_label_matrix`` in snorkel_pipeline_final.py.
    """
    n, m = diffs.shape
    M = np.full((n, m), -1, dtype=int)
    for j, name in enumerate(func_names):
        t = float(thresholds_dict[name]["best_threshold"])
        d = diffs[:, j]
        col = np.full(n, -1, dtype=int)
        col[d > t] = 0
        col[d < -t] = 1
        col[np.isnan(d)] = -1
        M[:, j] = col
    return M


def _full_metrics(y_true, y_pred):
    """Agreement (accuracy), macro precision/recall/F1 and coverage, computed
    only on non-abstain predictions.

    Mirrors the ``_full_metrics`` helper in snorkel_pipeline_final.py.
    """
    v = y_pred != -1
    cov = float(v.mean()) if len(v) else 0.0
    if v.sum() == 0:
        return {
            "accuracy":  0.0,
            "precision": 0.0,
            "recall":    0.0,
            "f1":        0.0,
            "coverage":  0.0,
        }
    yt, yp = y_true[v], y_pred[v]
    return {
        "accuracy":  round(float(accuracy_score(yt, yp)), 4),
        "precision": round(float(precision_score(
            yt, yp, average="macro", zero_division=0)), 4),
        "recall":    round(float(recall_score(
            yt, yp, average="macro", zero_division=0)), 4),
        "f1":        round(float(f1_score(
            yt, yp, average="macro", zero_division=0)), 4),
        "coverage":  round(cov, 4),
    }


def snorkel_evaluate(M_val, y_val, M_test, y_test, n_programs):
    """Snorkel-style aggregate evaluation of the aggregated programs.

    Trains a Snorkel ``LabelModel`` on ``M_val``/``y_val`` (when available)
    and predicts on val and on the *covered* test rows (rows where at least
    one program voted).  In parallel, runs ``MajorityLabelVoter`` on both
    splits.  All numbers are computed via :func:`_full_metrics` so that they
    are directly comparable to the original Snorkel pipeline's summary JSON.

    Falls back to training on test (with a warning) when no val matrix is
    provided.  Skips ``LabelModel`` entirely when ``n_programs < 3`` since
    Snorkel's LabelModel requires at least 3 labeling functions.

    Returns a dict with any of these keys that apply:
        ``LabelModel_val``, ``LabelModel_test``,
        ``MajorityVote_val``, ``MajorityVote_test``,
        ``label_matrix_diagnostics``.
    """
    has_val  = M_val  is not None and y_val  is not None
    has_test = M_test is not None and y_test is not None
    if not (has_val or has_test):
        return {}

    print("\n  Snorkel aggregate evaluation "
          f"(cardinality={SNORKEL_CARDINALITY}, epochs={SNORKEL_EPOCHS}, "
          f"seed={SNORKEL_SEED}):")

    label_model = None
    if n_programs < 3:
        print(f"    Skipping LabelModel: requires >=3 programs "
              f"(got {n_programs}); MajorityLabelVoter still runs.")
    elif has_val:
        print(f"    Training LabelModel on M_val "
              f"(n={M_val.shape[0]}, k={n_programs}) ...")
        label_model = LabelModel(
            cardinality=SNORKEL_CARDINALITY, verbose=False)
        label_model.fit(
            L_train=M_val, Y_dev=y_val,
            n_epochs=SNORKEL_EPOCHS, l2=SNORKEL_L2, lr=SNORKEL_LR,
            log_freq=100, seed=SNORKEL_SEED,
        )
    else:
        print("    WARNING: no val matrix — training LabelModel on M_test "
              "(in-sample; results will be optimistic).")
        label_model = LabelModel(
            cardinality=SNORKEL_CARDINALITY, verbose=False)
        label_model.fit(
            L_train=M_test, Y_dev=y_test,
            n_epochs=SNORKEL_EPOCHS, l2=SNORKEL_L2, lr=SNORKEL_LR,
            log_freq=100, seed=SNORKEL_SEED,
        )

    mv = MajorityLabelVoter(cardinality=SNORKEL_CARDINALITY)

    results = {}
    diag = {}

    def _print_row(label, m):
        print(f"    {label:<20s}  acc={m['accuracy']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  cov={m['coverage']:.4f}")

    # ── Val ────────────────────────────────────────────────────────────
    if has_val:
        n_val = M_val.shape[0]
        val_covered = ~(M_val == -1).all(axis=1)
        n_val_covered = int(val_covered.sum())
        diag["val_n_total"]      = int(n_val)
        diag["val_n_covered"]    = n_val_covered
        diag["val_coverage"]     = round(
            n_val_covered / n_val if n_val else 0.0, 4)
        diag["val_abstain_rate"] = round(float((M_val == -1).mean()), 4)

        if label_model is not None and n_val_covered > 0:
            y_lm = np.full(n_val, -1, dtype=int)
            y_lm[val_covered] = label_model.predict(
                L=M_val[val_covered], tie_break_policy="random")
            m = _full_metrics(y_val, y_lm)
            m["n_total"]   = int(n_val)
            m["n_covered"] = n_val_covered
            results["LabelModel_val"] = m
            _print_row("LabelModel (val)", m)

        if n_val_covered > 0:
            y_mv = np.full(n_val, -1, dtype=int)
            y_mv[val_covered] = mv.predict(
                L=M_val[val_covered], tie_break_policy="random")
            m = _full_metrics(y_val, y_mv)
            m["n_total"]   = int(n_val)
            m["n_covered"] = n_val_covered
            results["MajorityVote_val"] = m
            _print_row("MajorityVote (val)", m)

    # ── Test ───────────────────────────────────────────────────────────
    if has_test:
        n_test = M_test.shape[0]
        test_covered = ~(M_test == -1).all(axis=1)
        n_test_covered = int(test_covered.sum())
        diag["test_n_total"]      = int(n_test)
        diag["test_n_covered"]    = n_test_covered
        diag["test_coverage"]     = round(
            n_test_covered / n_test if n_test else 0.0, 4)
        diag["test_abstain_rate"] = round(float((M_test == -1).mean()), 4)

        if label_model is not None and n_test_covered > 0:
            y_lm = np.full(n_test, -1, dtype=int)
            y_lm[test_covered] = label_model.predict(
                L=M_test[test_covered], tie_break_policy="random")
            m = _full_metrics(y_test, y_lm)
            m["n_total"]   = int(n_test)
            m["n_covered"] = n_test_covered
            results["LabelModel_test"] = m
            _print_row("LabelModel (test)", m)

        if n_test_covered > 0:
            y_mv = np.full(n_test, -1, dtype=int)
            y_mv[test_covered] = mv.predict(
                L=M_test[test_covered], tie_break_policy="random")
            m = _full_metrics(y_test, y_mv)
            m["n_total"]   = int(n_test)
            m["n_covered"] = n_test_covered
            results["MajorityVote_test"] = m
            _print_row("MajorityVote (test)", m)

    if diag:
        results["label_matrix_diagnostics"] = diag

    return results


def main():
    args = parse_args()

    np.random.seed(args.seed)

    summary_path = resolve(args.summary)
    pipeline_outputs_dir = resolve(args.pipeline_outputs_dir)
    judge_dir = resolve(args.judges)
    output_dir = (str(args.output) if os.path.isabs(args.output)
                  else str(SCRIPT_DIR / args.output))
    tag = args.tag if args.tag else os.path.basename(output_dir)

    print("=" * 60)
    print("Aggregate Programs via Claude Adaptive Thinking")
    print("=" * 60)
    if args.dataset:
        print(f"  Dataset:    {args.dataset}")
    print(f"  Summary:    {summary_path}")
    print(f"  Pipeline:   {pipeline_outputs_dir}")
    print(f"  Judges:     {judge_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  N programs: {args.n_programs}")
    print(f"  Model:      {args.model}")
    print(f"  Effort:     {args.effort}")
    print(f"  Info mode:  {args.info_mode}  "
          f"(code = source only; "
          f"acc = +val accuracy; "
          f"cov = +val accuracy +val coverage; "
          f"val-samples = source only + denoising context "
          f"(NO per-program acc/cov))")
    print(f"  Seed:       {args.seed}")
    print(f"  Threshold:  0.00 ~ {args.eval_thresholds:.2f} (step 0.01)")
    if args.eval_only:
        print(f"  Mode:       EVAL-ONLY (skip generation)")

    summary = load_summary(summary_path)
    selected_names = summary["selected_program_names"]

    # ── eval-only fast path ───────────────────────────────────────────
    # Skip all generation-related loading (input programs, val stats).
    # Only load the already-generated aggregated programs and run eval.
    if args.eval_only:
        print("\n" + "=" * 60)
        print(f"EVAL-ONLY mode")
        print(f"  Looking for programs in: {output_dir}")
        print("=" * 60)

        if not os.path.isdir(output_dir):
            print(f"\nERROR: Output directory does not exist: {output_dir}")
            print(f"  (dataset={args.dataset}, n_programs={args.n_programs}, "
                  f"info_mode={args.info_mode})")
            print(f"  Run without --eval-only first to generate programs.")
            return

        all_programs = load_existing_programs(output_dir, args.n_programs)
        if not all_programs:
            print(f"\nERROR: No valid programs found in {output_dir}")
            print(f"  Expected aggregated_judge_1.py .. "
                  f"aggregated_judge_{args.n_programs}.py")
            return

        print(f"\n  Validating {len(all_programs)} program(s) ...")
        existing_val = validate_programs(all_programs)
        invalid_names = [n for n, v in existing_val.items()
                         if not v.get("valid")]
        for name in invalid_names:
            del all_programs[name]
            print(f"  Removing invalid: {name}")

        if not all_programs:
            print(f"\nERROR: All programs in {output_dir} failed validation.")
            return

        missing_ids = [i for i in range(1, args.n_programs + 1)
                       if f"aggregated_judge_{i}" not in all_programs]
        if missing_ids:
            print(f"\n  WARNING: {len(missing_ids)} of {args.n_programs} "
                  f"programs missing: {missing_ids}")
            print(f"  Proceeding with {len(all_programs)} available programs.")

        print(f"\n  Using {len(all_programs)} / {args.n_programs} "
              f"valid programs for evaluation")

        programs = all_programs
        validation_results = existing_val
        valid_programs = dict(programs)

    else:
        # ── Normal mode: load input programs, generate, then evaluate ─

        # Load per-program validation statistics from pipeline_outputs.
        print(f"\n  Loading validation stats from pipeline_outputs ...")
        program_val_stats, n_total_programs = load_program_val_stats(
            pipeline_outputs_dir, summary)
        if program_val_stats:
            print(f"  Loaded val stats for {len(program_val_stats)} programs"
                  + (f" (pool size: {n_total_programs})"
                     if n_total_programs else ""))
        else:
            print("  WARNING: No val stats loaded (val_diffs.json not found?)")

        diag = summary.get("label_matrix_diagnostics", {})
        print(f"\n  Selected programs ({len(selected_names)}):")
        print(f"  Test coverage: {diag.get('n_test_covered', '?')}"
              f"/{diag.get('n_test_total', '?')} "
              f"({diag.get('test_coverage', '?')})")
        for i, name in enumerate(selected_names):
            acc = summary["selected_program_val_accuracies"][i]
            thr = summary["selected_program_thresholds"][i]
            st = program_val_stats.get(name, {})
            cov_str = f", cov={st['coverage']:.2%}" if st else ""
            print(f"    {i+1:2d}. {name:<20s} "
                  f"(val_acc={acc:.4f}, thr={thr:.2f}{cov_str})")

        print("\n  Loading program source code ...")
        program_sources = {}
        for name in selected_names:
            source = load_program_source(judge_dir, name)
            if source is not None:
                program_sources[name] = source

        print(f"  Loaded {len(program_sources)} / "
              f"{len(selected_names)} programs")

        if not program_sources:
            print("ERROR: No programs loaded. Exiting.")
            return

        # ── Load denoising context for val-samples mode ─────────────
        denoising_context = ""
        if args.info_mode == "val-samples":
            if args.eval_val:
                val_path_for_samples = resolve(args.eval_val)
                print(f"\n  Loading denoising samples from val set ...")

                wrong_samples = find_wrong_val_samples(
                    pipeline_outputs_dir, val_path_for_samples,
                    n=40, seed=args.seed)
                correct_samples = find_correct_val_samples(
                    pipeline_outputs_dir, val_path_for_samples,
                    summary, n=10, seed=args.seed)

                if wrong_samples or correct_samples:
                    denoising_context = format_denoising_context(
                        wrong_samples, correct_samples, args.n_programs)
                    print(f"  Loaded {len(wrong_samples)} failure cases + "
                          f"{len(correct_samples)} correct cases for "
                          f"denoising context")
                else:
                    print("  WARNING: No denoising samples loaded. "
                          "val-samples mode will behave like code mode "
                          "(source only, no metadata, no denoising context).")
            else:
                print("  WARNING: --info-mode=val-samples but no --eval-val. "
                      "Cannot load samples. Behaving like code mode "
                      "(source only, no metadata, no denoising context).")

        os.makedirs(output_dir, exist_ok=True)

        # ── Resume: check for already-completed programs ──────────────
        print("\n" + "=" * 60)
        print("Checking for existing programs (resume) ...")
        print("=" * 60)

        all_programs = load_existing_programs(output_dir, args.n_programs)
        all_validation = {}

        if all_programs:
            print(f"\n  Found {len(all_programs)} existing program(s), "
                  f"validating ...")
            existing_val = validate_programs(all_programs)
            all_validation.update(existing_val)
            invalid_names = [n for n, v in existing_val.items()
                             if not v.get("valid")]
            for name in invalid_names:
                del all_programs[name]
                print(f"  Removing invalid program: {name}")
            print(f"  Valid existing programs: "
                  f"{len(all_programs)} / {args.n_programs}")
        else:
            print("  No existing programs found. Starting fresh.")
        # ── Generation loop with resume ───────────────────────────────
        MAX_RESUME_ROUNDS = args.max_resume_rounds
        api_round = 0

        while len(all_programs) < args.n_programs and api_round < MAX_RESUME_ROUNDS:
            api_round += 1
            needed_ids = [i for i in range(1, args.n_programs + 1)
                          if f"aggregated_judge_{i}" not in all_programs]

            print("\n" + "=" * 60)
            if api_round == 1 and not all_programs:
                print(f"API ROUND {api_round}: Generating all "
                      f"{args.n_programs} programs ...")
            else:
                print(f"API ROUND {api_round}: Generating {len(needed_ids)} "
                      f"missing program(s): {needed_ids}")
            print("=" * 60)

            if api_round == 1 and not all_programs:
                prompt = build_prompt(
                    program_sources, args.n_programs, summary,
                    program_val_stats, args.info_mode, n_total_programs,
                    denoising_context)
            else:
                prompt = build_resume_prompt(
                    needed_ids, all_programs, program_sources,
                    args.n_programs, summary, program_val_stats,
                    args.info_mode, n_total_programs,
                    denoising_context)

            prompt_chars = len(prompt)
            print(f"  Prompt length: {prompt_chars:,} characters "
                  f"(~{prompt_chars // 4:,} tokens)")

            prompt_path = os.path.join(
                output_dir, f"{tag}_prompt_round{api_round}.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"  Saved prompt to: {prompt_path}")

            print("\n  Calling Claude API ...")
            thinking_text, output_text, stop_reason = call_claude(
                prompt, args.model, args.max_output_tokens, args.effort
            )

            thinking_path = os.path.join(
                output_dir, f"{tag}_thinking_round{api_round}.txt")
            with open(thinking_path, "w", encoding="utf-8") as f:
                f.write(thinking_text)
            print(f"  Saved thinking to: {thinking_path}")

            output_path = os.path.join(
                output_dir, f"{tag}_output_round{api_round}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"  Saved output to: {output_path}")

            if stop_reason == "max_tokens":
                print("\n  ! Output TRUNCATED (stop_reason=max_tokens). "
                      "Will resume missing programs in next round.")

            print("\n  Parsing generated programs ...")
            new_programs = parse_programs(output_text, len(needed_ids))
            print(f"  Parsed {len(new_programs)} program(s) this round")

            if not new_programs:
                print("  WARNING: No programs parsed this round.")
                continue

            needed_names = {f"aggregated_judge_{pid}" for pid in needed_ids}
            remapped = {}
            unmatched_codes = []
            for parsed_name, code in new_programs.items():
                if parsed_name in needed_names:
                    remapped[parsed_name] = code
                else:
                    unmatched_codes.append(code)
            remaining_ids = [pid for pid in needed_ids
                             if f"aggregated_judge_{pid}" not in remapped]
            for pid, code in zip(remaining_ids, unmatched_codes):
                remapped[f"aggregated_judge_{pid}"] = code
            new_programs = remapped

            print("\n  Validating new programs ...")
            new_val = validate_programs(new_programs)
            all_validation.update(new_val)

            for name, code in new_programs.items():
                if new_val.get(name, {}).get("valid", False):
                    all_programs[name] = code
                    path = os.path.join(output_dir, f"{name}.py")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(code + "\n")
                    print(f"  Saved: {path}")
                else:
                    print(f"  Skipping invalid: {name}")

            print(f"\n  Progress: {len(all_programs)} / {args.n_programs} "
                  f"valid programs")

        if len(all_programs) < args.n_programs:
            missing = [i for i in range(1, args.n_programs + 1)
                       if f"aggregated_judge_{i}" not in all_programs]
            print(f"\n  WARNING: After {MAX_RESUME_ROUNDS} rounds, still "
                  f"missing programs: {missing}")

        programs = all_programs
        validation_results = all_validation
        valid_programs = dict(programs)

        print("\n" + "=" * 60)
        print("Saving final manifest ...")
        print("=" * 60)

        save_programs(programs, output_dir, tag)

    # ── Evaluation: val → tune thresholds → apply to test → Snorkel agg ──
    eval_results = {}
    val_thresholds = None
    norm_params = None
    loaded_funcs = None
    diffs_val = None
    diffs_test = None
    y_val = None
    y_test = None
    func_names = None

    if args.eval_val or args.eval_test:
        loaded_funcs = load_programs_as_funcs(valid_programs)
        if not loaded_funcs:
            print("  No valid programs to evaluate. Skipping evaluation.")
        else:
            func_names = list(loaded_funcs.keys())
            func_list = [loaded_funcs[n] for n in func_names]

    if loaded_funcs and args.eval_val:
        val_path = resolve(args.eval_val)
        print("\n" + "=" * 60)
        print(f"EVAL STAGE 1: Scoring validation set ({val_path})")
        print("=" * 60)
        val_data, y_val = load_jsonl(val_path)
        s1_val, s2_val = score_dataset(val_data, func_list, func_names, "val")

        print(f"\n  Computing norm params from VAL set ...")
        norm_params = compute_norm_params(s1_val, s2_val)

        print("\n" + "=" * 60)
        print("EVAL STAGE 2: Tuning thresholds on validation set")
        print("=" * 60)
        diffs_val = normalize_and_diff(s1_val, s2_val, norm_params)
        val_thresholds = tune_thresholds(diffs_val, y_val, func_names,
                                        args.eval_thresholds)

        # NOTE: this is IN-SAMPLE — thresholds were tuned on these same val
        # samples, so val accuracy here is an optimistic upper bound on what
        # we'd expect on truly held-out data. Test accuracy below is the
        # honest out-of-sample number.
        print("\n  Val per-program results "
              "(IN-SAMPLE — thresholds tuned on this set):")
        val_agg = apply_and_evaluate(
            diffs_val, y_val, func_names, val_thresholds, "val")
        eval_results["val"] = val_agg

    if loaded_funcs and args.eval_test:
        test_path = resolve(args.eval_test)
        print("\n" + "=" * 60)
        print(f"EVAL STAGE 3: Evaluating on test set ({test_path})")
        print("=" * 60)
        test_data, y_test = load_jsonl(test_path)
        s1_test, s2_test = score_dataset(
            test_data, func_list, func_names, "test")

        if norm_params is None:
            print("  WARNING: No val set provided — computing norm params "
                  "from test (not recommended)")
            norm_params = compute_norm_params(s1_test, s2_test)

        diffs_test = normalize_and_diff(s1_test, s2_test, norm_params)

        if val_thresholds is not None:
            print("\n  Applying thresholds from VALIDATION set to test:")
            test_results = apply_and_evaluate(
                diffs_test, y_test, func_names, val_thresholds, "test")
        else:
            print("\n  WARNING: No val thresholds — tuning on test directly "
                  "(not recommended, provide --eval-val)")
            val_thresholds = tune_thresholds(
                diffs_test, y_test, func_names, args.eval_thresholds)
            test_results = apply_and_evaluate(
                diffs_test, y_test, func_names, val_thresholds, "test")

        eval_results["test"] = test_results

    # ── EVAL STAGE 4: Snorkel LabelModel + MajorityLabelVoter aggregation ─
    if loaded_funcs and val_thresholds is not None and (
            diffs_val is not None or diffs_test is not None):
        print("\n" + "=" * 60)
        print("EVAL STAGE 4: Snorkel aggregate evaluation "
              "(LabelModel + MajorityLabelVoter)")
        print("=" * 60)
        M_val_agg = (
            build_label_matrix_from_diffs(diffs_val, func_names, val_thresholds)
            if diffs_val is not None else None)
        M_test_agg = (
            build_label_matrix_from_diffs(diffs_test, func_names, val_thresholds)
            if diffs_test is not None else None)
        agg_metrics = snorkel_evaluate(
            M_val=M_val_agg, y_val=y_val,
            M_test=M_test_agg, y_test=y_test,
            n_programs=len(func_names),
        )
        if agg_metrics:
            eval_results["aggregate"] = agg_metrics

    if eval_results:
        eval_path = os.path.join(output_dir, f"{tag}_evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n  Evaluation results saved to: {eval_path}")

    agg_summary = {
        "tag": tag,
        "model": args.model,
        "info_mode": args.info_mode,
        "eval_only": args.eval_only,
        "eval_thresholds_max": args.eval_thresholds,
        "seed": args.seed,
        "n_input_programs": len(selected_names),
        "n_aggregated_programs": len(programs),
        "n_valid_programs": len(valid_programs),
        "input_summary": os.path.basename(summary_path),
        "validation_results": {k: v for k, v in validation_results.items()
                               if isinstance(v, dict)},
        "evaluation": eval_results,
        "evaluation_notes": {
            "val_is_in_sample": True,
            "test_is_out_of_sample": True,
            "aggregation_method": (
                "Snorkel LabelModel + MajorityLabelVoter "
                "(cardinality=2, epochs=500, l2=0.01, lr=0.01, "
                "seed=123, tie_break_policy=random)"
            ),
        },
        "input_pipeline_metrics": {
            "LabelModel_test_accuracy": summary.get("LabelModel_test", {}).get("accuracy"),
            "MajorityVote_test_accuracy": summary.get("MajorityVote_test", {}).get("accuracy"),
        },
    }
    agg_summary_path = os.path.join(output_dir, f"{tag}_aggregation_summary.json")
    with open(agg_summary_path, "w") as f:
        json.dump(agg_summary, f, indent=2)
    print(f"\n  Aggregation summary saved to: {agg_summary_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  {len(programs)} aggregated programs in {output_dir}/")
    agg = eval_results.get("aggregate", {})
    if agg.get("LabelModel_val"):
        m = agg["LabelModel_val"]
        print(f"  Aggregated LabelModel  val  acc: {m['accuracy']:.4f}  "
              f"(cov={m['coverage']:.4f})")
    if agg.get("MajorityVote_val"):
        m = agg["MajorityVote_val"]
        print(f"  Aggregated MajorityVote val acc: {m['accuracy']:.4f}  "
              f"(cov={m['coverage']:.4f})")
    if agg.get("LabelModel_test"):
        m = agg["LabelModel_test"]
        print(f"  Aggregated LabelModel  test acc: {m['accuracy']:.4f}  "
              f"(cov={m['coverage']:.4f})")
    if agg.get("MajorityVote_test"):
        m = agg["MajorityVote_test"]
        print(f"  Aggregated MajorityVote test acc: {m['accuracy']:.4f}  "
              f"(cov={m['coverage']:.4f})")
    orig_lm = summary.get("LabelModel_test", {}).get("accuracy")
    orig_mv = summary.get("MajorityVote_test", {}).get("accuracy")
    if orig_lm:
        print(f"  Original LabelModel test accuracy:  {orig_lm}")
    if orig_mv:
        print(f"  Original MajorityVote test accuracy: {orig_mv}")


if __name__ == "__main__":
    main()
