#!/usr/bin/env python3
"""
split_dataset.py — Unified dataset splitter for PAJAMA
======================================================
Merges five dataset-specific splitters into a single CLI:

    python split_dataset.py pandalm    [--raw ... --seed 42 ...]
    python split_dataset.py multipref  [--seed 42]
    python split_dataset.py judgelm    [--seed 42]
    python split_dataset.py prometheus [--seed 42]
    python split_dataset.py hendrydong [--seed 42]
    python split_dataset.py all        [--seed 42]

Each subcommand loads raw data, applies dataset-specific filtering, and
writes train/val/test JSONL splits.  Run '<subcommand> -h' for options.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════
#  Common utilities
# ═══════════════════════════════════════════════════════════════════════════

def write_jsonl(path: str, items: list[dict], *, ensure_ascii: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in items:
            f.write(json.dumps(row, ensure_ascii=ensure_ascii) + "\n")


def _require_datasets():
    """Lazy-import and return ``datasets.load_dataset``."""
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets",
              file=sys.stderr)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  PandaLM
# ═══════════════════════════════════════════════════════════════════════════

_PANDALM_SECTION_RE = re.compile(
    r'### (Instruction|Input|Response 1|Response 2|Evaluation):\s*\n?'
)


def _pandalm_parse_input_seq(input_seq: str) -> tuple[str, str, str, str]:
    instruction = extra_input = response1 = response2 = ""
    parts = _PANDALM_SECTION_RE.split(input_seq)
    for i, part in enumerate(parts):
        if i + 1 >= len(parts):
            continue
        nxt = parts[i + 1]
        if part == "Instruction":
            instruction = nxt.strip()
        elif part == "Input":
            extra_input = nxt.strip()
        elif part == "Response 1":
            response1 = nxt.strip()
        elif part == "Response 2":
            response2 = nxt.strip()
    return instruction, extra_input, response1, response2


def _pandalm_parse_verdict(output_seq: str) -> str:
    for line in output_seq.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def split_pandalm(*, raw=None, train_out=None, val_out=None, bert_out=None,
                  test_raw=None, test_out=None,
                  n_total=20_000, n_val=500, n_bert=5_000, seed=42) -> int:
    """Filter and split PandaLM train set + filter test set."""
    print("\n" + "=" * 60)
    print("  PandaLM")
    print("=" * 60)

    if raw is None:
        raw = os.path.join(SCRIPT_DIR, "pandalm_train_raw.json")
    if train_out is None:
        train_out = os.path.join(SCRIPT_DIR, "pandalm_train_19500_v2.jsonl")
    if val_out is None:
        val_out = os.path.join(SCRIPT_DIR, "pandalm_val_500_v2.jsonl")
    if bert_out is None:
        bert_out = os.path.join(SCRIPT_DIR, "pandalm_train_bert_v2.jsonl")
    if test_raw is None:
        test_raw = os.path.join(SCRIPT_DIR, "testset-v1.json")
    if test_out is None:
        test_out = os.path.join(SCRIPT_DIR, "pandalm_test_894.jsonl")

    n_train_target = n_total - n_val
    if n_val >= n_total:
        print(f"ERROR: n_val ({n_val}) must be < n_total ({n_total})", file=sys.stderr)
        return 2
    if n_train_target != 19500 or n_val != 500:
        print(f"[note] producing {n_train_target} train + {n_val} val samples")

    if not os.path.exists(raw):
        print(f"ERROR: raw file not found: {raw}", file=sys.stderr)
        return 2

    print(f"Loading {raw} ...")
    with open(raw, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  loaded {len(raw_data):,} raw records")

    verdict_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    kept: list[dict] = []

    for row in raw_data:
        if not isinstance(row, dict):
            drop_counts["not_dict"] += 1
            continue
        in_seq = row.get("input_sequence")
        out_seq = row.get("output_sequence")
        if not isinstance(in_seq, str) or not isinstance(out_seq, str):
            drop_counts["missing_fields"] += 1
            continue

        verdict = _pandalm_parse_verdict(out_seq)
        verdict_counts[verdict] += 1
        if verdict not in ("1", "2"):
            drop_counts["verdict_not_binary"] += 1
            continue

        instruction, _extra, r1, r2 = _pandalm_parse_input_seq(in_seq)
        if not instruction:
            drop_counts["no_instruction"] += 1
            continue
        if not r1 or not r2:
            drop_counts["missing_response"] += 1
            continue
        if r1 == r2:
            drop_counts["identical_responses"] += 1
            continue

        kept.append({"input_sequence": in_seq, "output_sequence": out_seq})

    print("\nVerdict distribution (first line of output_sequence):")
    for v, c in verdict_counts.most_common(10):
        print(f"  {v!r:<12s} : {c:,}")
    print("\nDropped:")
    for reason, c in drop_counts.most_common():
        print(f"  {reason:<22s} : {c:,}")
    print(f"\n  strict-binary survivors : {len(kept):,}")

    if len(kept) < n_total:
        print(f"ERROR: only {len(kept):,} strict-binary samples available, "
              f"need {n_total:,}", file=sys.stderr)
        return 3

    rng = random.Random(seed)
    rng.shuffle(kept)
    selected = kept[:n_total]

    sel_counts = Counter(_pandalm_parse_verdict(r["output_sequence"]) for r in selected)
    print(f"\nSelected {len(selected):,} samples; "
          f"label balance -> 1={sel_counts['1']:,}  2={sel_counts['2']:,}")

    val_samples = selected[:n_val]
    train_samples = selected[n_val:]
    assert len(train_samples) == n_train_target

    remaining_for_bert = kept[n_total:]
    bert_size = min(n_bert, len(remaining_for_bert))
    bert_samples = remaining_for_bert[:bert_size]

    print(f"\n  Train size      : {len(train_samples):,}")
    print(f"  Val size        : {len(val_samples):,}")
    print(f"  Train-BERT size : {bert_size:,}  "
          f"(of {len(remaining_for_bert):,} remaining filtered samples)")

    write_jsonl(train_out, train_samples, ensure_ascii=True)
    write_jsonl(val_out, val_samples, ensure_ascii=True)

    train_counts = Counter(_pandalm_parse_verdict(r["output_sequence"]) for r in train_samples)
    val_counts = Counter(_pandalm_parse_verdict(r["output_sequence"]) for r in val_samples)
    print(f"\nWrote {train_out}  ({len(train_samples):,} lines, "
          f"1={train_counts['1']:,}  2={train_counts['2']:,})")
    print(f"Wrote {val_out}   ({len(val_samples):,} lines, "
          f"1={val_counts['1']:,}  2={val_counts['2']:,})")

    if bert_size > 0:
        bert_path = bert_out.replace(".jsonl", f"_{bert_size}.jsonl")
        write_jsonl(bert_path, bert_samples, ensure_ascii=True)
        bert_counts = Counter(_pandalm_parse_verdict(r["output_sequence"]) for r in bert_samples)
        print(f"Wrote {bert_path}  ({bert_size:,} lines, "
              f"1={bert_counts['1']:,}  2={bert_counts['2']:,})")
    else:
        print("  No remaining samples for BERT training set.")

    # ── Test set: filter ties from human-annotated test data ─────────────
    if os.path.exists(test_raw):
        print(f"\n{'-'*60}")
        print(f"Loading test set: {test_raw} ...")
        with open(test_raw, "r", encoding="utf-8") as f:
            test_raw_data = json.load(f)
        print(f"  loaded {len(test_raw_data):,} raw test records")

        test_kept: list[dict] = []
        test_drop_counts: Counter[str] = Counter()
        test_verdict_counts: Counter[int] = Counter()

        for row in test_raw_data:
            if not isinstance(row, dict):
                test_drop_counts["not_dict"] += 1
                continue
            votes = [row.get("annotator1"), row.get("annotator2"), row.get("annotator3")]
            if any(v is None for v in votes):
                test_drop_counts["missing_annotators"] += 1
                continue

            counts = Counter(votes)
            majority_val, majority_cnt = counts.most_common(1)[0]
            if majority_cnt < 2:
                test_drop_counts["no_majority"] += 1
                continue

            test_verdict_counts[majority_val] += 1
            if majority_val not in (1, 2):
                test_drop_counts["tie_majority_0"] += 1
                continue

            test_kept.append(row)

        print("\nTest set majority-vote distribution:")
        for v, c in sorted(test_verdict_counts.items()):
            label = {0: "tie", 1: "resp1 wins", 2: "resp2 wins"}.get(v, str(v))
            print(f"  {v} ({label:<12s}) : {c:,}")
        print("\nTest set dropped:")
        for reason, c in test_drop_counts.most_common():
            print(f"  {reason:<22s} : {c:,}")
        print(f"\n  non-tie test survivors : {len(test_kept):,}")

        write_jsonl(test_out, test_kept, ensure_ascii=True)
        test_label_counts: Counter[int] = Counter()
        for row in test_kept:
            votes = [row["annotator1"], row["annotator2"], row["annotator3"]]
            majority_val = Counter(votes).most_common(1)[0][0]
            test_label_counts[majority_val] += 1
        print(f"\nWrote {test_out}  ({len(test_kept):,} lines, "
              f"1={test_label_counts.get(1, 0):,}  2={test_label_counts.get(2, 0):,})")
    else:
        print(f"\n[skip] test raw file not found: {test_raw}")

    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  MultiPref
# ═══════════════════════════════════════════════════════════════════════════

_MULTIPREF_EXCLUDE_CATEGORIES = {"Coding"}
_MULTIPREF_EXCLUDE_SUBJECTS = {"Mathematics", "Computer Science"}
_MULTIPREF_WINNER_SCORE = 1
_MULTIPREF_LOSER_SCORE = 0


def _multipref_extract_response_text(messages):
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list) or len(messages) == 0:
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "")
    if isinstance(messages[-1], dict):
        return messages[-1].get("content", "")
    return ""


def _multipref_is_excluded_subject(subject):
    if not subject:
        return False
    s_lower = subject.lower()
    for excl in _MULTIPREF_EXCLUDE_SUBJECTS:
        if excl.lower() in s_lower:
            return True
    return False


def _multipref_to_pajama(comparison_id, query, chosen_text, rejected_text, rng):
    if rng.random() < 0.5:
        return {
            "comparison_id": comparison_id,
            "question_body": query,
            "answer1_body": chosen_text,
            "answer2_body": rejected_text,
            "score": [_MULTIPREF_WINNER_SCORE, _MULTIPREF_LOSER_SCORE],
        }
    else:
        return {
            "comparison_id": comparison_id,
            "question_body": query,
            "answer1_body": rejected_text,
            "answer2_body": chosen_text,
            "score": [_MULTIPREF_LOSER_SCORE, _MULTIPREF_WINNER_SCORE],
        }


def split_multipref(*, seed=42, total_target=8_500,
                    val_size=170, test_size=1_700) -> int:
    """Split allenai/multipref into train/val/test with human GT."""
    print("\n" + "=" * 60)
    print("  MultiPref")
    print("=" * 60)

    load_dataset = _require_datasets()
    random.seed(seed)
    rng = random.Random(seed)

    print("Loading allenai/multipref (3 configs) ...")
    default_ds = load_dataset("allenai/multipref", "default", split="train")
    human_ds = load_dataset("allenai/multipref", "human_overall_binarized", split="train")
    gpt4_ds = load_dataset("allenai/multipref", "gpt4_overall_binarized", split="train")
    print(f"  Total samples: {len(default_ds)}")

    default_map = {row["comparison_id"]: row for row in default_ds}
    human_map = {row["comparison_id"]: row for row in human_ds}
    gpt4_map = {row["comparison_id"]: row for row in gpt4_ds}

    # ── Filter: remove code/math ──
    print(f"\nFiltering out code/math ...")
    print(f"  Exclude categories: {_MULTIPREF_EXCLUDE_CATEGORIES}")
    print(f"  Exclude subjects:   {_MULTIPREF_EXCLUDE_SUBJECTS}")

    all_cats = Counter(row["category"] for row in default_map.values())
    all_subjs = Counter(row.get("subject_study") or "N/A" for row in default_map.values())
    print(f"\n  All categories:  {dict(all_cats.most_common())}")
    print(f"  All subjects:    {dict(all_subjs.most_common())}")

    valid_ids = []
    excluded_cat = 0
    excluded_subj = 0
    excluded_subj_vals: Counter[str] = Counter()
    for cid, row in default_map.items():
        if row["category"] in _MULTIPREF_EXCLUDE_CATEGORIES:
            excluded_cat += 1
            continue
        subj = row.get("subject_study") or ""
        if _multipref_is_excluded_subject(subj):
            excluded_subj += 1
            excluded_subj_vals[subj] += 1
            continue
        valid_ids.append(cid)

    print(f"\n  Excluded by category: {excluded_cat}")
    print(f"  Excluded by subject:  {excluded_subj}  {dict(excluded_subj_vals)}")
    print(f"  Remaining: {len(valid_ids)}")

    # ── Split into pools ──
    pool_human_clear: list = []
    pool_human_tie: list = []
    missing_human = 0
    missing_gpt4 = 0

    for cid in valid_ids:
        h = human_map.get(cid)
        if h is None:
            missing_human += 1
            continue
        if gpt4_map.get(cid) is None:
            missing_gpt4 += 1
            continue
        if h["tie_is_common"]:
            pool_human_tie.append(cid)
        else:
            pool_human_clear.append(cid)

    if missing_human or missing_gpt4:
        print(f"  Skipped: {missing_human} missing from human config, "
              f"{missing_gpt4} missing from gpt4 config")
    print(f"\n  Pool A (human clear, for val/test): {len(pool_human_clear)}")
    print(f"  Pool B (human tie, train-only):     {len(pool_human_tie)}")

    # ── Shuffle and split ──
    rng_split = random.Random(seed)
    rng_split.shuffle(pool_human_clear)
    rng_split.shuffle(pool_human_tie)

    actual_val = min(val_size, len(pool_human_clear))
    actual_test = min(test_size, len(pool_human_clear) - actual_val)

    val_ids = pool_human_clear[:actual_val]
    test_ids = pool_human_clear[actual_val:actual_val + actual_test]

    train_ids_from_a = pool_human_clear[actual_val + actual_test:]
    train_ids = train_ids_from_a + pool_human_tie
    rng_split.shuffle(train_ids)

    train_budget = total_target - actual_val - actual_test
    if len(train_ids) > train_budget:
        train_ids = train_ids[:train_budget]
        print(f"\n  Capped train to {train_budget} to meet {total_target} total target")

    print(f"\n  Val size:   {len(val_ids):>6,}  (human GT)")
    print(f"  Test size:  {len(test_ids):>6,}  (human GT)")
    print(f"  Train size: {len(train_ids):>6,}  (human GT)")
    print(f"  Total:      {len(val_ids) + len(test_ids) + len(train_ids):>6,}")

    # ── Convert to PAJAMA format ──
    print("\nConverting to PAJAMA format ...")

    def convert_split(cid_list, use_human_gt):
        records = []
        skipped = 0
        for cid in cid_list:
            src = human_map if use_human_gt else gpt4_map
            binarized = src.get(cid)
            if binarized is None:
                skipped += 1
                continue
            query = default_map[cid]["text"]
            chosen_text = _multipref_extract_response_text(binarized["chosen"])
            rejected_text = _multipref_extract_response_text(binarized["rejected"])
            if not chosen_text.strip() or not rejected_text.strip():
                skipped += 1
                continue
            rec = _multipref_to_pajama(cid, query, chosen_text, rejected_text, rng)
            records.append(rec)
        if skipped:
            print(f"    Skipped {skipped} samples (empty response or missing)")
        return records

    val_records = convert_split(val_ids, use_human_gt=True)
    test_records = convert_split(test_ids, use_human_gt=True)
    train_records = convert_split(train_ids, use_human_gt=True)

    # ── Save ──
    val_path = os.path.join(SCRIPT_DIR, f"multipref_val_{len(val_records)}.jsonl")
    test_path = os.path.join(SCRIPT_DIR, f"multipref_test_{len(test_records)}.jsonl")
    train_path = os.path.join(SCRIPT_DIR, f"multipref_train_{len(train_records)}.jsonl")

    write_jsonl(val_path, val_records, ensure_ascii=False)
    write_jsonl(test_path, test_records, ensure_ascii=False)
    write_jsonl(train_path, train_records, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {val_path}   ({len(val_records)} samples, human GT)")
    print(f"  {test_path}  ({len(test_records)} samples, human GT)")
    print(f"  {train_path} ({len(train_records)} samples, human GT)")

    # ── Sanity checks ──
    print("\nSanity checks:")
    for name, records in [("val", val_records), ("test", test_records), ("train", train_records)]:
        ans1_wins = sum(1 for r in records if r["score"][0] > r["score"][1])
        ans2_wins = len(records) - ans1_wins
        print(f"  [{name}] ans1_wins={ans1_wins} ({ans1_wins/len(records)*100:.1f}%), "
              f"ans2_wins={ans2_wins} ({ans2_wins/len(records)*100:.1f}%)")

    for name, records in [("val", val_records), ("test", test_records), ("train", train_records)]:
        gaps = [abs(r["score"][0] - r["score"][1]) for r in records]
        print(f"  [{name}] score gap: min={min(gaps)}, max={max(gaps)}, "
              f"all=={_MULTIPREF_WINNER_SCORE - _MULTIPREF_LOSER_SCORE}: "
              f"{all(g == _MULTIPREF_WINNER_SCORE - _MULTIPREF_LOSER_SCORE for g in gaps)}")

    val_cids = set(r["comparison_id"] for r in val_records)
    test_cids = set(r["comparison_id"] for r in test_records)
    train_cids = set(r["comparison_id"] for r in train_records)
    assert len(val_cids & test_cids) == 0, "Val/Test overlap!"
    assert len(val_cids & train_cids) == 0, "Val/Train overlap!"
    assert len(test_cids & train_cids) == 0, "Test/Train overlap!"
    print("  No overlap between splits: OK")

    agree_clear, total_clear = 0, 0
    agree_all, total_all = 0, 0
    for cid in train_ids:
        h = human_map.get(cid)
        g = gpt4_map.get(cid)
        if h is None or g is None:
            continue
        total_all += 1
        if h.get("chosen_model") == g.get("chosen_model"):
            agree_all += 1
        if not h.get("tie_is_common", True):
            total_clear += 1
            if h.get("chosen_model") == g.get("chosen_model"):
                agree_clear += 1
    if total_all:
        print(f"  Human-GPT4 agreement on train (all):   {agree_all}/{total_all} "
              f"({agree_all/total_all*100:.1f}%)")
    if total_clear:
        print(f"  Human-GPT4 agreement on train (clear): {agree_clear}/{total_clear} "
              f"({agree_clear/total_clear*100:.1f}%)")
    for name, records in [("val", val_records), ("test", test_records), ("train", train_records)]:
        cats = [default_map[r["comparison_id"]]["category"] for r in records]
        top3 = Counter(cats).most_common(3)
        top3_str = ", ".join(f"{c}:{n}" for c, n in top3)
        print(f"  [{name}] top categories: {top3_str}")

    print("\nDone!")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  JudgeLM
# ═══════════════════════════════════════════════════════════════════════════

def split_judgelm(*, seed=42, score_gap_threshold=6, total_samples=25_000,
                  test_size=5_000, val_size=500, bert_train_size=5_000) -> int:
    """Filter JudgeLM-100K by score gap and split into train/val/test/bert."""
    print("\n" + "=" * 60)
    print("  JudgeLM")
    print("=" * 60)

    load_dataset = _require_datasets()

    train_size = total_samples - test_size - val_size

    print("Loading JudgeLM-100K from HuggingFace ...")
    dataset = load_dataset(
        "json",
        data_files="hf://datasets/BAAI/JudgeLM-100K/judgelm_train_100k.jsonl",
        split="train",
    )
    print(f"  Total rows in JudgeLM-100K: {len(dataset)}")

    print(f"Computing score differences and filtering for gap >= {score_gap_threshold} ...")

    def compute_score_diff(example):
        scores = example["score"]
        if scores and len(scores) == 2:
            example["score_diff"] = abs(float(scores[0]) - float(scores[1]))
        else:
            example["score_diff"] = 0.0
        return example

    dataset = dataset.map(compute_score_diff)
    filtered = dataset.filter(lambda x: x["score_diff"] >= score_gap_threshold)
    print(f"  Rows with score gap >= {score_gap_threshold}: {len(filtered)}")

    if len(filtered) < total_samples:
        print(f"  Only {len(filtered)} samples available (need {total_samples}). "
              f"Using ALL and adjusting proportionally.")
        available = len(filtered)
        actual_test = int(available * test_size / total_samples)
        actual_val = int(available * val_size / total_samples)
        actual_train = available - actual_test - actual_val
    else:
        available = total_samples
        actual_test = test_size
        actual_val = val_size
        actual_train = train_size

    print(f"Shuffling and selecting {available} samples (seed={seed}) ...")
    shuffled = filtered.shuffle(seed=seed)
    selected = shuffled.select(range(available))

    test_dataset = selected.select(range(actual_test))
    remaining = selected.select(range(actual_test, available))
    val_dataset = remaining.select(range(actual_val))
    train_dataset = remaining.select(range(actual_val, actual_val + actual_train))

    n_leftover = len(shuffled) - available
    bert_size = min(bert_train_size, n_leftover)
    bert_dataset = shuffled.select(range(available, available + bert_size)) if bert_size > 0 else None

    print(f"\n  Train size      : {len(train_dataset):>6,}")
    print(f"  Val size        : {len(val_dataset):>6,}")
    print(f"  Test size       : {len(test_dataset):>6,}")
    print(f"  Train-BERT size : {bert_size:>6,}  "
          f"(of {n_leftover:,} remaining threshold-passing samples)")
    print(f"  Total           : "
          f"{len(train_dataset) + len(val_dataset) + len(test_dataset) + bert_size:>6,}")

    train_path = os.path.join(SCRIPT_DIR, "judgelm_train_19500.jsonl")
    val_path = os.path.join(SCRIPT_DIR, "judgelm_val_500.jsonl")
    test_path = os.path.join(SCRIPT_DIR, "judgelm_test_5000.jsonl")
    bert_path = os.path.join(SCRIPT_DIR, f"judgelm_train_bert_{bert_size}.jsonl")

    train_dataset.to_json(train_path)
    val_dataset.to_json(val_path)
    test_dataset.to_json(test_path)

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    if bert_dataset is not None:
        bert_dataset.to_json(bert_path)
        print(f"  {bert_path}")

    splits = [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]
    if bert_dataset is not None:
        splits.append(("train_bert", bert_dataset))
    for name, ds in splits:
        min_diff = min(ds["score_diff"])
        print(f"  [{name}] min score_diff = {min_diff}  (should be >= {score_gap_threshold})")

    print("\nDone!")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Prometheus
# ═══════════════════════════════════════════════════════════════════════════

def split_prometheus(*, seed=42, score_gap_threshold=3,
                     total_samples=25_000, test_size=5_000, val_size=500) -> int:
    """Split prometheus-eval/Preference-Collection by score gap."""
    print("\n" + "=" * 60)
    print("  Prometheus")
    print("=" * 60)

    load_dataset = _require_datasets()

    train_size = total_samples - test_size - val_size

    print("Loading prometheus-eval/Preference-Collection from HuggingFace ...")
    dataset = load_dataset("prometheus-eval/Preference-Collection", split="train")
    print(f"  Total rows: {len(dataset)}")

    print(f"\nFiltering for |orig_score_A - orig_score_B| >= {score_gap_threshold} ...")
    filtered = dataset.filter(
        lambda x: abs(int(x["orig_score_A"]) - int(x["orig_score_B"])) >= score_gap_threshold)
    print(f"  Rows after gap filter: {len(filtered)}")

    gap_dist = Counter(
        abs(int(a) - int(b))
        for a, b in zip(filtered["orig_score_A"], filtered["orig_score_B"]))
    print("  |A-B| distribution (after filter):")
    for k in sorted(gap_dist):
        print(f"    {k}: {gap_dist[k]}")

    available = len(filtered)
    if available >= total_samples:
        actual_train, actual_val, actual_test = train_size, val_size, test_size
        actual_total = total_samples
    else:
        print(f"\n  Only {available} samples available (target was {total_samples}).")
        print(f"     Using ALL {available} samples, splitting proportionally.")
        actual_test = int(available * test_size / total_samples)
        actual_val = int(available * val_size / total_samples)
        actual_train = available - actual_test - actual_val
        actual_total = available

    print(f"\n  Target  -> Train={train_size}  Val={val_size}  Test={test_size}"
          f"  Total={total_samples}")
    print(f"  Actual  -> Train={actual_train}  Val={actual_val}  Test={actual_test}"
          f"  Total={actual_total}")

    print(f"\nShuffling and selecting {actual_total} samples (seed={seed}) ...")
    selected = filtered.shuffle(seed=seed).select(range(actual_total))

    test_ds = selected.select(range(actual_test))
    rest = selected.select(range(actual_test, actual_total))
    val_ds = rest.select(range(actual_val))
    train_ds = rest.select(range(actual_val, actual_val + actual_train))

    print(f"\n  Train size : {len(train_ds):>6,}")
    print(f"  Val size   : {len(val_ds):>6,}")
    print(f"  Test size  : {len(test_ds):>6,}")
    print(f"  Total      : {len(train_ds) + len(val_ds) + len(test_ds):>6,}")

    def normalise_row(row):
        return {
            "question_body": row["orig_instruction"],
            "answer1_body": row["orig_response_A"],
            "answer2_body": row["orig_response_B"],
            "score": [int(row["orig_score_A"]), int(row["orig_score_B"])],
            "orig_criteria": row["orig_criteria"],
            "orig_reference_answer": row["orig_reference_answer"],
            "orig_preference": row["orig_preference"],
        }

    splits_info = [
        ("train", train_ds, f"prometheus_train_{len(train_ds)}.jsonl"),
        ("val", val_ds, f"prometheus_val_{len(val_ds)}.jsonl"),
        ("test", test_ds, f"prometheus_test_{len(test_ds)}.jsonl"),
    ]

    for name, ds, filename in splits_info:
        path = os.path.join(SCRIPT_DIR, filename)
        records = [normalise_row(row) for row in ds]
        write_jsonl(path, records, ensure_ascii=False)
        print(f"  Saved {path}")

    # ── Sanity checks ──
    print("\nSanity checks:")
    for name, ds, filename in splits_info:
        path = os.path.join(SCRIPT_DIR, filename)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        print(f"  [{name}] {len(lines)} lines")

        first = json.loads(lines[0])
        pref_counts: dict[str, int] = {"A": 0, "B": 0}
        score_diffs = []
        for line in lines:
            r = json.loads(line)
            pref_counts[r["orig_preference"]] += 1
            score_diffs.append(abs(r["score"][0] - r["score"][1]))
        print(f"    Preference distribution: A={pref_counts['A']}  B={pref_counts['B']}")
        print(f"    min |score gap| = {min(score_diffs)}  (should be >= {score_gap_threshold})")
        print(f"    score example: {first['score']}")

    print("\nDone!")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  HendryDong (hendrydong/preference_700K)
# ═══════════════════════════════════════════════════════════════════════════

_HENDRYDONG_CODE_KEYWORDS = [
    r'\bcode\b', r'\bcoding\b', r'\bprogram\b', r'\bprogramming\b',
    r'\bfunction\b', r'\bscript\b', r'\bsnippet\b', r'\bclass\b',
    r'\bdef\b', r'\bimport\b', r'\breturn\b', r'\bprint\(',
    r'\bpython\b', r'\bjavascript\b', r'\bjava\b', r'\bc\+\+\b',
    r'\bruby\b', r'\brust\b', r'\bswift\b', r'\btypescript\b',
    r'\bhtml\b', r'\bcss\b', r'\bsql\b', r'\bbash\b',
    r'\bapi\b', r'\bjson\b', r'\bxml\b', r'\bhttp\b',
    r'\bgithub\b', r'\bgit\b', r'\bdocker\b', r'\bkubernetes\b',
    r'\bpytorch\b', r'\btensorflow\b', r'\bnumpy\b', r'\bpandas\b',
    r'\bregex\b', r'\bfor loop\b', r'\bwhile loop\b',
    r'\bcompile\b', r'\bdebug\b', r'\bsyntax\b', r'\bvariable\b',
    r'\balgorithm\b', r'\bdata structure\b', r'\bbinary search\b',
    r'\bsorting\b', r'\blinked list\b', r'\btree\b', r'\bgraph\b',
    r'\brecursion\b', r'\brecursive\b',
    r'```', r'\bvoid\b', r'\bint\s+main\b',
    r'\bnpm\b', r'\bpip\b', r'\bcargo\b', r'\bmaven\b', r'\bgradle\b',
    r'\bIDE\b', r'\bcompiler\b', r'\binterpreter\b',
    r'\barray\b', r'\bstack\b', r'\bqueue\b', r'\bhash\s*map\b',
    r'\bwebsite\b', r'\bfrontend\b', r'\bbackend\b',
    r'\breact\b', r'\bangular\b', r'\bvue\b', r'\bnode\.?js\b',
    r'\bflask\b', r'\bdjango\b', r'\bfastapi\b',
    r'\bESPHome\b', r'\bMATLAB\b', r'\bR\s+language\b',
]

_HENDRYDONG_MATH_KEYWORDS = [
    r'\bcalculate\b', r'\bcompute\b', r'\bsolve\b',
    r'\bequation\b', r'\bformula\b', r'\bproof\b', r'\bprove\b',
    r'\btheorem\b', r'\blemma\b', r'\bcorollary\b',
    r'\bderivative\b', r'\bintegral\b', r'\bcalculus\b',
    r'\balgebra\b', r'\bgeometry\b', r'\btrigonometry\b',
    r'\bmatrix\b', r'\bmatrices\b', r'\bdeterminant\b',
    r'\beigenvalue\b', r'\beigenvector\b',
    r'\bprobability\b', r'\bstatistic\b', r'\bstatistics\b',
    r'\bmathematic\b', r'\barithmetic\b',
    r'\bpolynomial\b', r'\bquadratic\b', r'\blinear algebra\b',
    r'\bfactorial\b', r'\bprime number\b', r'\bfibonacci\b',
    r'\blog(arithm)?\b', r'\bexponent\b',
    r'\bsin\b', r'\bcos\b', r'\btan\b',
    r'\d+\s*[\+\-\*\/\^]\s*\d+',
    r'\bx\s*[=<>]\s*\d', r'\d\s*[=<>]\s*x\b',
    r'\bGCD\b', r'\bLCM\b',
]

_HENDRYDONG_COMPLEX_TASK_KEYWORDS = [
    r'\bIn this task\b', r'\bGiven the task definition\b',
    r'\bTask:\s', r'\bInstruction:\s',
    r'\btranslate\b.*\bto\b.*\b(English|Chinese|French|German|Spanish|Japanese|Korean|Russian|Arabic|Hindi|Portuguese|Gujarati)\b',
    r'\btranslation\b',
    r'\bconvert\b.*\b(sentence|text|paragraph)\b.*\bto\b',
    r'\b(Q|A):\s',
    r'\bInput:\s.*\bOutput:\s',
    r'\b\[Q\]:', r'\b\[A\]:',
    r'\bgiven this background information\b',
    r'\bwrite\b.*\b(essay|article|report|proposal|paper)\b',
    r'\bsummarize the book\b', r'\bbook summarizer\b',
    r'\bwrite a\s+\d+[\-\s]word\b',
    r'\byaml\b', r'\bwiring\b.*\bESP\b',
    r'\bpuzzle\b', r'\briddle\b', r'\bunscramble\b',
    r'\brole[\s-]*play\b', r'\byou are a\b.*\b(expert|professional|specialist|engineer|developer|scientist)\b',
    r'\bExplain like I\'m five\b', r'\bELI5\b',
]

_HENDRYDONG_CODE_PAT = re.compile('|'.join(_HENDRYDONG_CODE_KEYWORDS), re.IGNORECASE)
_HENDRYDONG_MATH_PAT = re.compile('|'.join(_HENDRYDONG_MATH_KEYWORDS), re.IGNORECASE)
_HENDRYDONG_COMPLEX_PAT = re.compile('|'.join(_HENDRYDONG_COMPLEX_TASK_KEYWORDS), re.IGNORECASE)


def _hendrydong_is_single_turn(messages):
    if not isinstance(messages, list):
        return False
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return roles.count("user") == 1 and roles.count("assistant") == 1


def _hendrydong_is_simple_general_qa(prompt_text):
    if _HENDRYDONG_CODE_PAT.search(prompt_text):
        return False
    if _HENDRYDONG_MATH_PAT.search(prompt_text):
        return False
    if _HENDRYDONG_COMPLEX_PAT.search(prompt_text):
        return False
    return True


def _hendrydong_get_user_text(messages):
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "")
    return ""


def _hendrydong_get_assistant_text(messages):
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def split_hendrydong(*, seed=42, target_train=20_000, target_val=500,
                     target_test=5_000, winner_score=5, loser_score=1) -> int:
    """Filter hendrydong/preference_700K and split into train/val/test."""
    print("\n" + "=" * 60)
    print("  HendryDong")
    print("=" * 60)

    load_dataset = _require_datasets()
    random.seed(seed)

    target_total = target_train + target_val + target_test

    print("Loading hendrydong/preference_700K from HuggingFace ...")
    ds = load_dataset("hendrydong/preference_700K", split="train")
    total = len(ds)
    print(f"  Total rows: {total:,}")

    # ── Step 1: non-null scores ──
    print("\nStep 1: Filtering for non-null rejected_score and chosen_score ...")
    scored = ds.filter(
        lambda x: x["rejected_score"] is not None and x["chosen_score"] is not None
    )
    print(f"  After score-not-null filter: {len(scored):,}  "
          f"(removed {total - len(scored):,})")

    # ── Step 2: single-turn ──
    print("\nStep 2: Filtering for single-turn conversations ...")
    single = scored.filter(
        lambda x: _hendrydong_is_single_turn(x["chosen"]) and _hendrydong_is_single_turn(x["rejected"])
    )
    print(f"  After single-turn filter: {len(single):,}  "
          f"(removed {len(scored) - len(single):,})")

    # ── Step 3: score ratio ──
    print("\nStep 3: Filtering for max_score >= 2 * min_score ...")

    def score_filter(x):
        cs, rs = float(x["chosen_score"]), float(x["rejected_score"])
        mx = max(cs, rs)
        mn = min(cs, rs)
        if mn <= 0:
            return mx > 0
        return mx >= 2.0 * mn

    quality = single.filter(score_filter)
    print(f"  After score-ratio filter: {len(quality):,}  "
          f"(removed {len(single) - len(quality):,})")

    # ── Step 4: simple general QA ──
    print("\nStep 4: Filtering for simple general Q&A "
          "(no coding, math, complex tasks) ...")
    general = quality.filter(
        lambda x: _hendrydong_is_simple_general_qa(_hendrydong_get_user_text(x["chosen"]))
    )
    print(f"  After general-QA filter: {len(general):,}  "
          f"(removed {len(quality) - len(general):,})")

    available = len(general)
    print(f"\n  Total available after all filters: {available:,}")

    # ── Determine split sizes ──
    if available >= target_total:
        train_size = target_train
        val_size = target_val
        test_size = target_test
    else:
        print(f"  Only {available:,} samples (need {target_total:,}). "
              f"Adjusting proportionally.")
        test_size = int(available * target_test / target_total)
        val_size = max(int(available * target_val / target_total), 100)
        train_size = available - test_size - val_size

    use_total = train_size + val_size + test_size

    # ── Shuffle ──
    print(f"\nShuffling all {available:,} filtered samples (seed={seed}) ...")
    shuffled = general.shuffle(seed=seed)

    # ── Convert to PAJAMA format ──
    print("Converting to PAJAMA format ...")
    records = []
    for row in shuffled.select(range(use_total)):
        question = _hendrydong_get_user_text(row["chosen"])
        chosen_text = _hendrydong_get_assistant_text(row["chosen"])
        rejected_text = _hendrydong_get_assistant_text(row["rejected"])

        if random.random() < 0.5:
            rec = {
                "question_body": question,
                "answer1_body": chosen_text,
                "answer2_body": rejected_text,
                "score": [winner_score, loser_score],
            }
        else:
            rec = {
                "question_body": question,
                "answer1_body": rejected_text,
                "answer2_body": chosen_text,
                "score": [loser_score, winner_score],
            }
        records.append(rec)

    test_records = records[:test_size]
    val_records = records[test_size:test_size + val_size]
    train_records = records[test_size + val_size:]

    print(f"\n  Train size : {len(train_records):>6,}")
    print(f"  Val size   : {len(val_records):>6,}")
    print(f"  Test size  : {len(test_records):>6,}")
    print(f"  Total      : {len(train_records) + len(val_records) + len(test_records):>6,}")

    train_path = os.path.join(SCRIPT_DIR, f"hendrydong_train_{len(train_records)}.jsonl")
    val_path = os.path.join(SCRIPT_DIR, f"hendrydong_val_{len(val_records)}.jsonl")
    test_path = os.path.join(SCRIPT_DIR, f"hendrydong_test_{len(test_records)}.jsonl")

    write_jsonl(train_path, train_records, ensure_ascii=False)
    write_jsonl(val_path, val_records, ensure_ascii=False)
    write_jsonl(test_path, test_records, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    # ── Optional: train_full (all filtered minus test/val) ──
    train_full_start = test_size + val_size
    train_full_count = available - train_full_start
    if train_full_count > train_size:
        print(f"\nConverting train_full ({train_full_count:,} samples = "
              f"all filtered - test - val) ...")
        random.seed(seed + 1)
        extra_records = []
        for row in shuffled.select(range(use_total, available)):
            question = _hendrydong_get_user_text(row["chosen"])
            chosen_text = _hendrydong_get_assistant_text(row["chosen"])
            rejected_text = _hendrydong_get_assistant_text(row["rejected"])

            if random.random() < 0.5:
                rec = {
                    "question_body": question,
                    "answer1_body": chosen_text,
                    "answer2_body": rejected_text,
                    "score": [winner_score, loser_score],
                }
            else:
                rec = {
                    "question_body": question,
                    "answer1_body": rejected_text,
                    "answer2_body": chosen_text,
                    "score": [loser_score, winner_score],
                }
            extra_records.append(rec)

        train_full_records = train_records + extra_records
        train_full_path = os.path.join(
            SCRIPT_DIR, f"hendrydong_train_full_{len(train_full_records)}.jsonl")
        write_jsonl(train_full_path, train_full_records, ensure_ascii=False)
        print(f"  {train_full_path}")
        print(f"  train_full size: {len(train_full_records):,}  "
              f"(train_20k [{len(train_records):,}] + extra [{len(extra_records):,}])")

        n_a1 = sum(1 for r in train_full_records if r["score"][0] > r["score"][1])
        n_a2 = len(train_full_records) - n_a1
        print(f"  [train_full] answer1 wins: {n_a1}, answer2 wins: {n_a2}  "
              f"(ratio {n_a1/max(len(train_full_records),1):.2%} / "
              f"{n_a2/max(len(train_full_records),1):.2%})")
    else:
        print("\n  No extra samples beyond train_20k for train_full.")

    # ── Sanity checks ──
    print("\nSanity checks:")
    for name, recs in [("train", train_records),
                       ("val", val_records),
                       ("test", test_records)]:
        n_a1_wins = sum(1 for r in recs if r["score"][0] > r["score"][1])
        n_a2_wins = len(recs) - n_a1_wins
        print(f"  [{name}] answer1 wins: {n_a1_wins}, "
              f"answer2 wins: {n_a2_wins}  "
              f"(ratio {n_a1_wins/max(len(recs),1):.2%} / "
              f"{n_a2_wins/max(len(recs),1):.2%})")

    print("\nDone!")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  CLI dispatch
# ═══════════════════════════════════════════════════════════════════════════

DATASETS = {
    "pandalm": split_pandalm,
    "multipref": split_multipref,
    "judgelm": split_judgelm,
    "prometheus": split_prometheus,
    "hendrydong": split_hendrydong,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unified dataset splitter for PAJAMA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="dataset", help="Dataset to split")
    sub.required = True

    # ── pandalm ──
    p = sub.add_parser("pandalm", help="Split PandaLM train + test")
    p.add_argument("--raw", default=os.path.join(SCRIPT_DIR, "pandalm_train_raw.json"))
    p.add_argument("--train_out", default=os.path.join(SCRIPT_DIR, "pandalm_train_19500_v2.jsonl"))
    p.add_argument("--val_out", default=os.path.join(SCRIPT_DIR, "pandalm_val_500_v2.jsonl"))
    p.add_argument("--bert_out", default=os.path.join(SCRIPT_DIR, "pandalm_train_bert_v2.jsonl"))
    p.add_argument("--test_raw", default=os.path.join(SCRIPT_DIR, "testset-v1.json"))
    p.add_argument("--test_out", default=os.path.join(SCRIPT_DIR, "pandalm_test_894.jsonl"))
    p.add_argument("--n_total", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--n_bert", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    # ── multipref ──
    p = sub.add_parser("multipref", help="Split MultiPref dataset")
    p.add_argument("--seed", type=int, default=42)

    # ── judgelm ──
    p = sub.add_parser("judgelm", help="Split JudgeLM-100K dataset")
    p.add_argument("--seed", type=int, default=42)

    # ── prometheus ──
    p = sub.add_parser("prometheus", help="Split Prometheus Preference-Collection")
    p.add_argument("--seed", type=int, default=42)

    # ── hendrydong ──
    p = sub.add_parser("hendrydong", help="Split hendrydong/preference_700K")
    p.add_argument("--seed", type=int, default=42)

    # ── all ──
    p = sub.add_parser("all", help="Split all datasets sequentially")
    p.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.dataset == "all":
        results: dict[str, int] = {}
        for name, func in DATASETS.items():
            try:
                if name == "pandalm":
                    rc = func(seed=args.seed)
                else:
                    rc = func(seed=args.seed)
                results[name] = rc
            except Exception as e:
                print(f"\nERROR in {name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                results[name] = 1

        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        for name, rc in results.items():
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  {name:<12s} : {status}")
        return max(results.values()) if results else 0

    elif args.dataset == "pandalm":
        return split_pandalm(
            raw=args.raw, train_out=args.train_out, val_out=args.val_out,
            bert_out=args.bert_out, test_raw=args.test_raw, test_out=args.test_out,
            n_total=args.n_total, n_val=args.n_val, n_bert=args.n_bert,
            seed=args.seed,
        )

    else:
        return DATASETS[args.dataset](seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
