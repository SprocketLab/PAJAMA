#!/usr/bin/env python3
"""
llm_judge_server.py

Call a running vLLM OpenAI-compatible server to judge pairwise preferences
across 5 datasets (preference_700K, judgelm, multipref, pandalm, prometheus)
on val and test splits.

Data is loaded from the PAJAMA Hugging Face dataset by default:
    https://huggingface.co/datasets/sprocket-lab/PAJAMA

GT annotation sources:
  preference_700K -> human
  judgelm         -> GPT-4
  multipref       -> human
  pandalm         -> GPT-3.5-Turbo (val), human (test)
  prometheus      -> GPT-4

Prerequisites:
    A vLLM server running (e.g. via vllm_server.sh)

Usage:
    python llm_judge_server.py --base-url http://localhost:8000/v1 --model judge-model
    python llm_judge_server.py --datasets preference_700K judgelm --splits test
    python llm_judge_server.py --datasets pandalm --splits val test --concurrency 32
    python llm_judge_server.py --constrained-decoding   # use vLLM guided_choice
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'datasets'. Install with:\n"
        "  pip install datasets huggingface_hub"
    ) from exc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HF_REPO_ID = os.environ.get("PAJAMA_HF_REPO", "sprocket-lab/PAJAMA")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EVALUATOR_PROMPT = """You are an expert evaluator assessing AI-generated responses. Your task is to determine which of two responses better serves the user's needs.

<question>
{question}
</question>

<response_a>
{response_a}
</response_a>

<response_b>
{response_b}
</response_b>

Reply with ONLY "A" or "B"."""

MAX_QUESTION_CHARS = 4000
MAX_RESPONSE_CHARS = 6000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len("\n...[TRUNCATED]..."))
    return text[:keep] + "\n...[TRUNCATED]..."


def build_messages(question: str, response_a: str, response_b: str) -> list[dict]:
    prompt = EVALUATOR_PROMPT.format(
        question=truncate_text(question, MAX_QUESTION_CHARS),
        response_a=truncate_text(response_a, MAX_RESPONSE_CHARS),
        response_b=truncate_text(response_b, MAX_RESPONSE_CHARS),
    )
    return [{"role": "user", "content": prompt}]


def parse_judgment(text: str) -> int:
    """Return 0 for A, 1 for B, -1 on failure. Strips <think> blocks."""
    if not text or not text.strip():
        return -1
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        return -1
    cleaned = cleaned.strip().strip("\"'").rstrip(".").strip()
    if cleaned == "A":
        return 0
    if cleaned == "B":
        return 1
    return -1


def compute_metrics(y_true: list[int], y_pred: list[int], label: str) -> dict:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_total = len(y_true_arr)

    valid = y_pred_arr != -1
    n_valid = int(valid.sum())
    n_failed = n_total - n_valid
    pred_counts = Counter(y_pred)

    metrics: dict[str, Any] = {
        "label": label,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_failed": n_failed,
        "pred_A": int(pred_counts.get(0, 0)),
        "pred_B": int(pred_counts.get(1, 0)),
    }

    if n_valid == 0:
        metrics["error"] = "all_failed"
        return metrics

    yt = y_true_arr[valid]
    yp = y_pred_arr[valid]

    metrics.update(
        accuracy=round(float(accuracy_score(yt, yp)), 4),
        strict_accuracy=round(float((yt == yp).sum() / n_total), 4),
        precision=round(float(precision_score(yt, yp, average="macro", zero_division=0)), 4),
        recall=round(float(recall_score(yt, yp, average="macro", zero_division=0)), 4),
        f1=round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        coverage=round(float(n_valid / n_total), 4),
    )
    return metrics


def print_metrics(m: dict) -> None:
    if "error" in m:
        print(f"    all failed: {m['n_failed']} / {m['n_total']}")
        return
    print(f"    accuracy        : {m['accuracy']:.4f}  (on {m['n_valid']} parsed samples)")
    print(f"    strict_accuracy : {m['strict_accuracy']:.4f}  (parse failures = wrong, N={m['n_total']})")
    print(f"    precision (mac) : {m['precision']:.4f}")
    print(f"    recall    (mac) : {m['recall']:.4f}")
    print(f"    f1        (mac) : {m['f1']:.4f}")
    print(f"    coverage        : {m['coverage']:.4f}  ({m['n_valid']}/{m['n_total']}, {m['n_failed']} parse failures)")
    print(f"    pred distrib    : A={m['pred_A']}  B={m['pred_B']}  fail={m['n_failed']}")


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
    "preference_700K": {"hf_config": "preference_700K", "gt_source": {"val": "human",          "test": "human"}},
    "judgelm":         {"hf_config": "judgelm",         "gt_source": {"val": "GPT-4",           "test": "GPT-4"}},
    "multipref":       {"hf_config": "multipref",       "gt_source": {"val": "human",          "test": "human"}},
    "pandalm":         {"hf_config": "pandalm",         "gt_source": {"val": "GPT-3.5-Turbo",  "test": "human"}},
    "prometheus":      {"hf_config": "prometheus",      "gt_source": {"val": "GPT-4",           "test": "GPT-4"}},
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())
ALL_SPLITS = ["val", "test"]


# ---------------------------------------------------------------------------
# Dataset loader
# Returns: list of (question, resp_a, resp_b, gt_label), skipped_count
# gt_label: 0 = A wins (verdict=1), 1 = B wins (verdict=2)
# ---------------------------------------------------------------------------

def load_hf_split(dataset_key: str, split: str, repo_id: str = HF_REPO_ID):
    """Load one split from the PAJAMA HuggingFace dataset (unified schema)."""
    hf_config = DATASET_CONFIGS[dataset_key]["hf_config"]
    hf_split = "validation" if split == "val" else split
    print(f"  Loading HF: {repo_id}  config={hf_config}  split={hf_split}")
    ds = load_dataset(repo_id, hf_config, split=hf_split)
    samples, skipped = [], 0
    for row in ds:
        verdict = row.get("verdict")
        if verdict in (1, "1"):
            gt = 0  # response1 wins = A
        elif verdict in (2, "2"):
            gt = 1  # response2 wins = B
        else:
            skipped += 1
            continue
        samples.append((
            str(row.get("query", "")),
            str(row.get("response1", "")),
            str(row.get("response2", "")),
            gt,
        ))
    return samples, skipped


# ---------------------------------------------------------------------------
# Async inference against vLLM OpenAI-compatible server
# ---------------------------------------------------------------------------

async def call_judge(
    client,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    constrained: bool,
    max_retries: int = 3,
) -> str:
    extra: dict = {}
    if constrained:
        extra["extra_body"] = {"guided_choice": ["A", "B"]}

    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.0,
                    **extra,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"    [WARN] API error after {max_retries} attempts: {e}")
                    return ""
                await asyncio.sleep(2 ** attempt)
    return ""


async def run_inference(
    client,
    model: str,
    conversations: list[list[dict]],
    concurrency: int,
    constrained: bool,
) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        call_judge(client, model, msgs, semaphore, constrained)
        for msgs in conversations
    ]
    results = []
    batch_size = max(concurrency * 4, 100)
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        chunk_results = await asyncio.gather(*chunk)
        results.extend(chunk_results)
        done = min(start + batch_size, len(tasks))
        print(f"    progress: {done}/{len(tasks)}", end="\r", flush=True)
    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge via vLLM OpenAI-compatible server (all datasets, all splits)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="judge-model",
        help="Model name as served by vLLM (--served-model-name in vllm_server.sh)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        choices=ALL_SPLITS,
        help="Splits to evaluate (default: val test)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent API requests (default: 64)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save per-split outputs and summaries (default: llm_judge_server_results/)",
    )
    parser.add_argument(
        "--constrained-decoding",
        action="store_true",
        help="Use vLLM guided_choice=[A,B] in extra_body to constrain output tokens",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"HuggingFace dataset repo (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the server (default: EMPTY, as used by vLLM)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag used for output directory and file names (default: model name with / replaced by _)",
    )
    return parser.parse_args()


async def async_main(args, output_dir: str, model_tag: str) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise SystemExit("openai package not found. Install it: pip install openai")

    # Single client for the entire run — avoids event-loop lifecycle issues
    # that occur when AsyncOpenAI is shared across multiple asyncio.run() calls.
    async with AsyncOpenAI(base_url=args.base_url, api_key=args.api_key) as client:
        all_summaries: dict[str, dict] = {}

        for ds_name in args.datasets:
            ds_summaries: dict[str, dict] = {}

            for split in args.splits:
                cfg = DATASET_CONFIGS[ds_name]
                gt_source = cfg["gt_source"][split]

                samples, skipped = load_hf_split(ds_name, split, repo_id=args.repo_id)
                if not samples:
                    print(f"\n[SKIP] {ds_name}/{split}: no samples after filtering")
                    continue

                y_true = [s[3] for s in samples]
                gt_counts = Counter(y_true)

                label = f"{ds_name}/{split}"
                print(f"\n{'=' * 72}")
                print(f"Dataset: {ds_name}  Split: {split}  GT: {gt_source}")
                print(f"{'=' * 72}")
                print(f"  samples: {len(samples)}  (skipped ties: {skipped})")
                print(f"  GT dist: A={gt_counts.get(0, 0)}  B={gt_counts.get(1, 0)}")

                conversations = [build_messages(q, r1, r2) for q, r1, r2, _ in samples]

                print(f"  Inference ({len(conversations)} prompts, concurrency={args.concurrency}) ...")
                t0 = time.time()
                raw_texts = await run_inference(
                    client,
                    args.model,
                    conversations,
                    args.concurrency,
                    args.constrained_decoding,
                )
                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.1f}s  ({len(conversations) / elapsed:.1f} samples/s)")

                preds = [parse_judgment(t) for t in raw_texts]
                m = compute_metrics(y_true, preds, label)
                print("\n  Metrics:")
                print_metrics(m)

                # Per-sample output
                per_sample = []
                for i, (q, r1, r2, gt) in enumerate(samples):
                    per_sample.append({
                        "idx": i,
                        "gt": gt,
                        "pred": preds[i],
                        "raw_output": raw_texts[i],
                        "question_preview": q[:200],
                        "resp_a_preview": r1[:100],
                        "resp_b_preview": r2[:100],
                    })

                prefix = os.path.join(output_dir, f"{ds_name}_{split}_{model_tag}")
                with open(prefix + "_outputs.json", "w", encoding="utf-8") as f:
                    json.dump(per_sample, f, indent=2, ensure_ascii=False)

                split_summary = {
                    "dataset": ds_name,
                    "split": split,
                    "gt_source": gt_source,
                    "model": args.model,
                    "base_url": args.base_url,
                    "constrained_decoding": args.constrained_decoding,
                    "metrics": m,
                    "skipped_gt_ties": skipped,
                    "elapsed_sec": round(elapsed, 1),
                    "samples_per_sec": round(len(samples) / elapsed, 2),
                }
                with open(prefix + "_summary.json", "w", encoding="utf-8") as f:
                    json.dump(split_summary, f, indent=2, ensure_ascii=False)

                ds_summaries[split] = split_summary

            all_summaries[ds_name] = ds_summaries

    # -----------------------------------------------------------------------
    # Cross-dataset summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print(f"  Model: {args.model}   Server: {args.base_url}")
    print("=" * 90)
    header = (
        f"  {'Dataset':<14} {'Split':<6} {'GT Source':<16}"
        f" {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Cov':>7} {'N':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ds_name, splits_data in all_summaries.items():
        for split, s in splits_data.items():
            m = s["metrics"]
            if "error" in m:
                print(f"  {ds_name:<14} {split:<6} {s['gt_source']:<16} ALL FAILED")
                continue
            print(
                f"  {ds_name:<14} {split:<6} {s['gt_source']:<16}"
                f" {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f}"
                f" {m['f1']:>7.4f} {m['coverage']:>7.4f} {m['n_total']:>6}"
            )

    combined_path = os.path.join(output_dir, f"all_summaries_{model_tag}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to: {output_dir}")
    print(f"Combined summary: {combined_path}")


def main():
    args = parse_args()

    model_tag = args.tag or args.model.replace("/", "_")
    output_dir = args.output_dir or os.path.join(os.getcwd(), model_tag)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("LLM-as-a-Judge  |  vLLM server mode")
    print(f"  server      : {args.base_url}")
    print(f"  model       : {args.model}")
    print(f"  tag         : {model_tag}")
    print(f"  hf repo     : {args.repo_id}")
    print(f"  datasets    : {args.datasets}")
    print(f"  splits      : {args.splits}")
    print(f"  concurrency : {args.concurrency}")
    print(f"  constrained : {args.constrained_decoding}")
    print(f"  output      : {output_dir}")
    print("=" * 80)

    asyncio.run(async_main(args, output_dir, model_tag))


if __name__ == "__main__":
    main()
