"""
converting_data.py

Prepare reward-model training data for judgelm and prometheus.

Two labeling modes
  snorkel  -- labels from program judges + Snorkel label model
               reads Y_hat_{train,val}.npy; coverage gated by M_{train,val}.npy
  gt       -- ground-truth GPT-4 preference labels
               derived from the 'score' field in the original JSONL files
               coverage still gated by M_{train,val}.npy

Coverage rule (both modes):
  A sample is kept only if at least one program judge gave a non-abstain vote,
  i.e. (M[i] != -1).any().  This is determined from the label matrix saved by
  the Snorkel pipeline.  For the test split the coverage comes from
  Y_hat_test_abstain.npy (rows with -1 are uncovered).

Output format per split (train.jsonl / val.jsonl / test.jsonl):
  {"chosen":   [{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
   "rejected": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Output directories
  snorkel: rm_data_judgelm/       rm_data_prometheus/
  gt:      rm_data_judgelm_gpt4/    rm_data_prometheus_gpt4/

Usage:
  python make_rm_data.py                            # both datasets, both modes
  python make_rm_data.py --datasets judgelm         # one dataset only
  python make_rm_data.py --modes gt                 # one mode only
"""

import argparse
import glob
import json
import os

import numpy as np
import tiktoken

GPT4_INPUT_PRICE_PER_1M  = 30.0  # GPT-4,          USD/1M input  tokens
OPUS_OUTPUT_PRICE_PER_1M = 25.0  # Claude Opus 4.6, USD/1M output tokens (no thinking)
_ENC = tiktoken.get_encoding("cl100k_base")

# ── per-sample instruction prompts (injected into every API call) ────────────
# Token counts of these templates are added to every sample's input cost.
# Placeholders ({question}, {answer_1}, {answer_2}) are stripped before counting
# because the actual question / answer tokens are counted separately per row.

PROMETHEUS_INSTRUCTION = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: Feedback: (write a feedback for criteria)
[RESULT] (an integer number between 1 and 5)
4. Please do not generate any other opening, closing, and explanations.
"""

JUDGELM_INSTRUCTION = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question}
[The Start of Assistant 1's Answer]
{answer_1}
[The End of Assistant 1's Answer]
[The Start of Assistant 2's Answer]
{answer_2}
[The End of Assistant 2's Answer]
[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
"""

INSTRUCTION_PROMPTS = {
    "judgelm":    JUDGELM_INSTRUCTION,
    "prometheus": PROMETHEUS_INSTRUCTION,
}


def _instruction_overhead_tokens(name):
    """Token count of the static instruction template (placeholders stripped)."""
    tmpl = INSTRUCTION_PROMPTS[name]
    for ph in ("{question}", "{answer_1}", "{answer_2}"):
        tmpl = tmpl.replace(ph, "")
    return len(_ENC.encode(tmpl))

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.join(SCRIPT_DIR, "..", "datasets")
_PAJAMA_DIR  = os.path.join(SCRIPT_DIR, "..", "pajama_workflow")

# ── file layout ──────────────────────────────────────────────────────────────

def _pipeline(tag, filename):
    """Path to a file in pajama_workflow/<tag>_outputs/."""
    return os.path.join(_PAJAMA_DIR, f"{tag}_outputs", filename)

DATASETS = {
    "judgelm": {
        "train_jsonl":  os.path.join(_DATA_DIR, "judgelm_train_19500.jsonl"),
        "val_jsonl":    os.path.join(_DATA_DIR, "judgelm_val_500.jsonl"),
        "test_jsonl":   os.path.join(_DATA_DIR, "judgelm_test_5000.jsonl"),
        "y_hat_train":  _pipeline("judgelm", "Y_hat_train.npy"),
        "y_hat_val":    _pipeline("judgelm", "Y_hat_val.npy"),
        "m_train":      _pipeline("judgelm", "M_train.npy"),
        "m_val":        _pipeline("judgelm", "M_val.npy"),
        "m_test":       _pipeline("judgelm", "M_test.npy"),
    },
    "prometheus": {
        "train_jsonl":  os.path.join(_DATA_DIR, "prometheus_train_19500.jsonl"),
        "val_jsonl":    os.path.join(_DATA_DIR, "prometheus_val_500.jsonl"),
        "test_jsonl":   os.path.join(_DATA_DIR, "prometheus_test_5000.jsonl"),
        "y_hat_train":  _pipeline("prometheus", "Y_hat_train.npy"),
        "y_hat_val":    _pipeline("prometheus", "Y_hat_val.npy"),
        "m_train":      _pipeline("prometheus", "M_train.npy"),
        "m_val":        _pipeline("prometheus", "M_val.npy"),
        "m_test":       _pipeline("prometheus", "M_test.npy"),
    },
}

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  saved {len(data):>6d} pairs  ->  {path}")


# ── row loader ────────────────────────────────────────────────────────────────

def load_rows(jsonl_path):
    """Parse judgelm/prometheus JSONL into (question, resp1, resp2, gt) tuples.

    Every line in the file yields exactly one tuple so that row indices stay
    aligned with the M / Y_hat arrays produced by the Snorkel pipeline.

    gt: 0 = resp1 wins, 1 = resp2 wins, -1 = tie (skipped by converters).
    """
    rows = []
    for row in load_jsonl(jsonl_path):
        s1, s2 = float(row["score"][0]), float(row["score"][1])
        if s1 > s2:
            gt = 0
        elif s2 > s1:
            gt = 1
        else:
            gt = -1
        rows.append((
            row["question_body"].strip(),
            row["answer1_body"].strip(),
            row["answer2_body"].strip(),
            gt,
        ))
    return rows

# ── pair builder ──────────────────────────────────────────────────────────────

def make_pair(question, resp1, resp2, winner):
    """winner: 0 = resp1 wins, 1 = resp2 wins."""
    chosen, rejected = (resp1, resp2) if winner == 0 else (resp2, resp1)
    return {
        "chosen":   [{"role": "user",      "content": question},
                     {"role": "assistant", "content": chosen}],
        "rejected": [{"role": "user",      "content": question},
                     {"role": "assistant", "content": rejected}],
    }

# ── coverage ──────────────────────────────────────────────────────────────────

def coverage_mask(M):
    """Boolean array: True where at least one program gave a non-abstain vote."""
    return (M != -1).any(axis=1)

# ── converters ────────────────────────────────────────────────────────────────

def convert_snorkel(rows, y_hat, covered):
    """Label from Snorkel Y_hat; keep only covered, non-empty rows.

    Y_hat_train/val use random tie-break so every row has a 0/1 label even for
    all-abstain samples.  covered[] (derived from M) is the true coverage gate.
    """
    pairs, skipped = [], 0
    for i, (q, r1, r2, _gt) in enumerate(rows):
        if i >= len(y_hat) or not covered[i]:
            skipped += 1
            continue
        label = int(y_hat[i])
        if label not in (0, 1) or not q or not r1 or not r2:
            skipped += 1
            continue
        pairs.append(make_pair(q, r1, r2, label))
    return pairs, skipped


def convert_gt(rows, covered):
    """Label from GPT-4 GT; keep only covered, non-tie, non-empty rows."""
    pairs, skipped = [], 0
    for i, (q, r1, r2, gt) in enumerate(rows):
        if not covered[i] or gt not in (0, 1) or not q or not r1 or not r2:
            skipped += 1
            continue
        pairs.append(make_pair(q, r1, r2, gt))
    return pairs, skipped


def label_accuracy(rows, y_hat, covered):
    """Agreement between Y_hat and GT, restricted to covered non-tie rows."""
    correct = total = 0
    for i, (_q, _r1, _r2, gt) in enumerate(rows):
        if i >= len(y_hat) or not covered[i]:
            continue
        label = int(y_hat[i])
        if label not in (0, 1) or gt not in (0, 1):
            continue
        total += 1
        correct += label == gt
    return (correct / total if total else 0.0), correct, total

# ── labeling cost benchmark ───────────────────────────────────────────────────

def count_gpt4_input_tokens(rows, instruction_tokens=0):
    """GPT-4 input tokens per sample: question + answer1 + answer2 + instruction.

    instruction_tokens accounts for the dataset-specific judging prompt that is
    prepended to every API call (e.g. Prometheus task description, JudgeLM
    rating template).  It is charged once per sample.
    """
    total = 0
    for (q, r1, r2, _gt) in rows:
        total += len(_ENC.encode(q))
        total += len(_ENC.encode(r1))
        total += len(_ENC.encode(r2))
        total += instruction_tokens
    return total


def count_judge_program_tokens(ds):
    """Output tokens for the generated judge programs (Claude Opus 4.6)."""
    prog_dir = os.path.join(SCRIPT_DIR, "..", "synthesized_programmatic_judges", ds)
    py_files = sorted(glob.glob(os.path.join(prog_dir, "judge_*.py")))
    total = sum(len(_ENC.encode(open(f, encoding="utf-8").read())) for f in py_files)
    return len(py_files), total


def labeling_cost_summary(dataset_names):
    W = 70
    print(f"\n{'='*W}")
    print("Labeling cost benchmark")
    print(f"  GPT-4 labeling : ${GPT4_INPUT_PRICE_PER_1M:.2f} / 1M  input tokens  "
          f"(instruction + question + 2 responses, per sample)")
    print(f"  Program gen    : ${OPUS_OUTPUT_PRICE_PER_1M:.2f} / 1M output tokens  "
          f"(Claude Opus 4.6, 80 programs per dataset)")
    print(f"{'='*W}")
    fmt = "  {:<14} {:<12} {:>16,}  ${:>10.4f}  {}"
    print(f"  {'Dataset':<14} {'Method':<12} {'Tokens':>16}  {'Cost (USD)':>11}  Note")
    print(f"  {'-'*(W-2)}")

    for name in dataset_names:
        cfg = DATASETS[name]
        # GPT-4: labeled every sample in train + val
        all_rows   = load_rows(cfg["train_jsonl"]) + load_rows(cfg["val_jsonl"])
        instr_tok  = _instruction_overhead_tokens(name)
        gpt4_tok   = count_gpt4_input_tokens(all_rows, instruction_tokens=instr_tok)
        gpt4_cost  = gpt4_tok / 1e6 * GPT4_INPUT_PRICE_PER_1M
        print(fmt.format(name, "GPT-4", gpt4_tok, gpt4_cost,
                         f"{len(all_rows):,} samples (train+val), "
                         f"+{instr_tok} instr tok/sample"))

        # Programs: one-time generation cost
        n_files, prog_tok = count_judge_program_tokens(name)
        prog_cost = prog_tok / 1e6 * OPUS_OUTPUT_PRICE_PER_1M
        print(fmt.format(name, "Programs", prog_tok, prog_cost,
                         f"{n_files} generated programs"))

    print(f"{'='*W}")


# ── main processing ───────────────────────────────────────────────────────────

def process(name, cfg, mode):
    suffix  = "_gpt4" if mode == "gt" else ""
    out_dir = os.path.join(SCRIPT_DIR, f"rm_data_{name}{suffix}")

    label_desc = "GPT-4 GT" if mode == "gt" else "Snorkel Y_hat"
    print(f"\n{'='*60}")
    print(f"  {name}  |  mode={mode} ({label_desc})  ->  {out_dir}/")
    print(f"{'='*60}")

    # ── train ────────────────────────────────────────────────────
    train_rows   = load_rows(cfg["train_jsonl"])
    covered_train = coverage_mask(np.load(cfg["m_train"]))
    print(f"  train rows loaded: {len(train_rows)}"
          f"  |  covered: {covered_train.sum()}/{len(covered_train)}")

    if mode == "snorkel":
        y_hat = np.load(cfg["y_hat_train"])
        train_pairs, skip = convert_snorkel(train_rows, y_hat, covered_train)
        acc, cor, tot = label_accuracy(train_rows, y_hat, covered_train)
        print(f"  Y_hat vs GT (train): {acc:.4f}  ({cor}/{tot})")
    else:
        train_pairs, skip = convert_gt(train_rows, covered_train)

    print(f"  train: {len(train_pairs)} pairs  ({skip} skipped)")

    # ── val ──────────────────────────────────────────────────────
    val_rows    = load_rows(cfg["val_jsonl"])
    covered_val = coverage_mask(np.load(cfg["m_val"]))
    print(f"  val rows loaded:   {len(val_rows)}"
          f"  |  covered: {covered_val.sum()}/{len(covered_val)}")

    if mode == "snorkel":
        y_hat_val = np.load(cfg["y_hat_val"])
        val_pairs, skip = convert_snorkel(val_rows, y_hat_val, covered_val)
        acc, cor, tot = label_accuracy(val_rows, y_hat_val, covered_val)
        print(f"  Y_hat vs GT (val):   {acc:.4f}  ({cor}/{tot})")
    else:
        val_pairs, skip = convert_gt(val_rows, covered_val)

    print(f"  val:   {len(val_pairs)} pairs  ({skip} skipped)")

    # fuse train + val into the final training split
    fused_train = train_pairs + val_pairs
    print(f"  fused train: {len(train_pairs)} + {len(val_pairs)} = {len(fused_train)} pairs")
    save_jsonl(fused_train, os.path.join(out_dir, "train.jsonl"))
    # val.jsonl kept separately so the training script has an eval split to monitor
    save_jsonl(val_pairs,   os.path.join(out_dir, "val.jsonl"))

    # ── test (always GT labels, coverage from M_test) ────────────
    test_rows    = load_rows(cfg["test_jsonl"])
    covered_test = coverage_mask(np.load(cfg["m_test"]))
    test_pairs, skip = convert_gt(test_rows, covered_test)
    print(f"  test:  {len(test_pairs)} pairs  ({skip} skipped)"
          f"  |  covered: {covered_test.sum()}/{len(covered_test)}")
    save_jsonl(test_pairs, os.path.join(out_dir, "test.jsonl"))


def main():
    parser = argparse.ArgumentParser(
        description="Build RM training data for judgelm and prometheus.")
    parser.add_argument(
        "--datasets", nargs="+", default=["judgelm", "prometheus"],
        choices=["judgelm", "prometheus"],
        help="Datasets to process (default: both).")
    parser.add_argument(
        "--modes", nargs="+", default=["snorkel", "gt"],
        choices=["snorkel", "gt"],
        help="Label sources to generate (default: both).")
    args = parser.parse_args()

    for name in args.datasets:
        for mode in args.modes:
            process(name, DATASETS[name], mode)

    labeling_cost_summary(args.datasets)

    print("\nOutput directories:")
    for name in args.datasets:
        for mode in args.modes:
            suffix = "_gt" if mode == "gt" else ""
            print(f"  rm_data_{name}{suffix}/")


if __name__ == "__main__":
    main()
