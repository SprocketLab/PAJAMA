"""
PAJAMA — Unified Judge Program Generator
========================================
Generate 80 judge programs (10 heuristics × 8 variants) for any PAJAMA
dataset.  Few-shot examples are loaded from the validation split
of the PAJAMA Hugging Face dataset:

    https://huggingface.co/datasets/sprocket-lab/PAJAMA

Supported datasets (--dataset):
    judgelm, pandalm, multipref, prometheus, preference_700K

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    # Private repo: hf auth login  (or export HF_TOKEN=...)

    python generate_judging_programs.py --dataset judgelm
    python generate_judging_programs.py --dataset pandalm
    python generate_judging_programs.py --dataset multipref
    python generate_judging_programs.py --dataset prometheus
    python generate_judging_programs.py --dataset preference_700K
"""

import argparse
import os
import re
import json
import random
import difflib
import anthropic

try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'datasets'. Install with:\n"
        "  pip install datasets huggingface_hub"
    ) from exc

# ── Configuration ──────────────────────────────────────────────────────────
MODEL = "claude-opus-4-6"
USE_THINKING = False
PROGRAMS_PER_HEURISTIC = 8          # 10 × 8 = 80
TOTAL_PROGRAMS = 80
NUM_FEW_SHOT_EXAMPLES = 10          # 2% of val set
SEED = 42
MAX_RETRIES_PER_SLOT = 5            # retries per program slot
MAX_GLOBAL_ATTEMPTS = 200           # safety valve for the whole run
SIMILARITY_THRESHOLD = 0.60         # reject if code similarity > this (Brian's value)

HF_REPO_ID = os.environ.get("PAJAMA_HF_REPO", "sprocket-lab/PAJAMA")
HF_SPLIT = "validation"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CLI name -> HF config + output paths.  All PAJAMA val rows share:
#   query, response1, response2, verdict  (+ score1/score2 when has_scores)
DATASET_CONFIGS = {
    "judgelm": {
        "hf_config": "judgelm",
        "has_scores": True,
        "output_dir": "judge_programs_judgelm",
        "manifest": "program_manifest_judgelm.json",
    },
    "pandalm": {
        "hf_config": "pandalm",
        "has_scores": False,
        "output_dir": "judge_programs_pandalm",
        "manifest": "program_manifest_pandalm.json",
    },
    "multipref": {
        "hf_config": "multipref",
        "has_scores": True,
        "output_dir": "judge_programs_multipref",
        "manifest": "program_manifest_multipref.json",
    },
    "prometheus": {
        "hf_config": "prometheus",
        "has_scores": True,
        "output_dir": "judge_programs_prometheus",
        "manifest": "program_manifest_prometheus.json",
    },
    "preference_700K": {
        "hf_config": "preference_700K",
        "has_scores": True,
        "output_dir": "judge_programs_preference_700K",
        "manifest": "program_manifest_preference_700K.json",
    },
}
# ───────────────────────────────────────────────────────────────────────────


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


# ── Similarity Engine (adapted from Brian's ROUGE-L approach) ─────────────

def clean_code_for_similarity(code: str) -> str:
    """Strip comments, normalize whitespace — same idea as Brian's clean_code."""
    lines = []
    for line in code.split("\n"):
        stripped = line.split("#")[0].rstrip()
        if stripped.strip():
            lines.append(stripped)
    return " ".join(lines)


def code_similarity(code_a: str, code_b: str) -> float:
    """Compute similarity ratio between two code strings.

    Uses difflib.SequenceMatcher (built-in, no rouge_score dependency).
    Comparable to ROUGE-L for our purposes.
    """
    clean_a = clean_code_for_similarity(code_a)
    clean_b = clean_code_for_similarity(code_b)
    return difflib.SequenceMatcher(None, clean_a, clean_b).ratio()


def is_too_similar(new_code: str, existing_codes: list, threshold: float) -> bool:
    """Check if new_code is too similar to any existing program."""
    for existing in existing_codes:
        if code_similarity(new_code, existing) > threshold:
            return True
    return False


# ── Few-shot Examples (from PAJAMA HF validation split) ───────────────────

def resolve_dataset_name(name: str) -> str:
    """Validate CLI dataset name against DATASET_CONFIGS."""
    key = name.strip()
    if key not in DATASET_CONFIGS:
        valid = sorted(DATASET_CONFIGS)
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {', '.join(valid)}")
    return key


def _verdict_to_label(verdict) -> str:
    if verdict in (1, "1"):
        return "Response A is better"
    if verdict in (2, "2"):
        return "Response B is better"
    raise ValueError(f"Unsupported verdict for few-shot prompt: {verdict!r}")


def load_validation_rows(dataset_key: str, repo_id: str = HF_REPO_ID) -> list[dict]:
    """Load the PAJAMA validation split for one config from Hugging Face."""
    cfg = DATASET_CONFIGS[dataset_key]
    print(f"Loading HF dataset: {repo_id}  config={cfg['hf_config']}  split={HF_SPLIT}")
    ds = load_dataset(repo_id, cfg["hf_config"], split=HF_SPLIT)
    rows = [dict(row) for row in ds]
    print(f"  Loaded {len(rows)} validation examples")
    return rows


def row_to_few_shot_example(row: dict, has_scores: bool) -> dict:
    """Convert one PAJAMA row to a few-shot example dict."""
    example = {
        "query": str(row["query"])[:300],
        "response_a": str(row["response1"])[:400],
        "response_b": str(row["response2"])[:400],
        "verdict": _verdict_to_label(row["verdict"]),
    }
    if has_scores:
        example["score_a"] = float(row["score1"])
        example["score_b"] = float(row["score2"])
    return example


def load_few_shot_examples(dataset_key: str, n=5, seed=42, repo_id: str = HF_REPO_ID):
    """Sample few-shot examples from the PAJAMA HF validation split."""
    cfg = DATASET_CONFIGS[dataset_key]
    rows = load_validation_rows(dataset_key, repo_id=repo_id)

    random.seed(seed)
    samples = random.sample(rows, min(n, len(rows)))
    examples = []
    for row in samples:
        examples.append(row_to_few_shot_example(row, has_scores=cfg["has_scores"]))
    return examples


def format_few_shot_examples(examples):
    parts = []
    for i, ex in enumerate(examples, 1):
        if "score_a" in ex and "score_b" in ex:
            a_label = f"Response A (score={ex['score_a']}):"
            b_label = f"Response B (score={ex['score_b']}):"
            gap_line = f" (gap = {abs(ex['score_a'] - ex['score_b'])})"
        else:
            a_label = "Response A:"
            b_label = "Response B:"
            gap_line = ""
        parts.append(
            f"--- Example {i} ---\n"
            f"Query: {ex['query']}\n\n"
            f"{a_label}\n{ex['response_a']}\n\n"
            f"{b_label}\n{ex['response_b']}\n\n"
            f"Ground-truth verdict: {ex['verdict']}{gap_line}\n"
        )
    return "\n".join(parts)


# ── Prompt ────────────────────────────────────────────────────────────────

def build_prompt(heuristic, few_shot_text, variant_index, existing_approaches=None):
    """Build generation prompt. Optionally include descriptions of existing
    variants so Claude knows what NOT to repeat."""
    avoid_block = ""
    if existing_approaches:
        descs = "\n".join(f"  - Variant {i+1}: {d}" for i, d in enumerate(existing_approaches))
        avoid_block = (
            f"\nThe following approaches have ALREADY been implemented for this heuristic. "
            f"You MUST use a substantially DIFFERENT algorithm:\n{descs}\n"
        )

    return f"""You are an expert Python developer and AI evaluation researcher.

Your task: Write a Python function that evaluates the quality of an LLM-generated response
to a given query. The function should return a numeric score where HIGHER values indicate
BETTER quality.

EVALUATION STRATEGY — focus STRICTLY on this dimension:
{heuristic['name']}: {heuristic['description']}
{avoid_block}
Here are some real examples from our dataset so you can understand what real queries and
responses look like, and what "good" vs "bad" answers look like in practice:

{few_shot_text}

IMPORTANT REQUIREMENTS:
- Output ONLY valid, executable Python code inside ```python ... ``` blocks. No explanation.
- The function signature must be exactly: def judging_function(query, response):
- Return a single numeric score (int or float). Higher = better quality.
- Use standard Python string/math operations for speed. You may use common libraries
  (re, math, collections, string, statistics) but NO heavy ML/NLP libraries.
- This is variant {variant_index}/{PROGRAMS_PER_HEURISTIC} — use a MEANINGFULLY DIFFERENT
  algorithm, features, and scoring formula than other variants.
- Include comprehensive error handling (try/except) so the function never crashes.
- The function must handle edge cases (empty strings, very short/long inputs).
- Return scores in a reasonable numeric range (e.g., 0-10 or 0-100).
- Make the function DISCRIMINATIVE: it should produce clearly different scores for
  high-quality vs low-quality responses.

```python
def judging_function(query, response):
    # Your implementation here
```"""


# ── Code Extraction & Validation ─────────────────────────────────────────

def extract_code(llm_output):
    matches = re.findall(r'```python\n(.*?)\n```', llm_output, re.DOTALL)
    if matches:
        for m in matches:
            if "def judging_function" in m:
                return m
        return matches[-1]
    return llm_output.replace("```python", "").replace("```", "").strip()


def validate_code(code_str):
    """Validate: syntax, execution, non-constant output, score range."""
    try:
        compiled = compile(code_str, "<generated>", "exec")
        ns = {}
        exec(compiled, ns)

        if "judging_function" not in ns:
            return False, "Missing 'judging_function'"

        fn = ns["judging_function"]

        test_cases = [
            ("What is the capital of France?",
             "Paris is the capital of France, located in the Ile-de-France region. "
             "It is known for the Eiffel Tower, Louvre Museum, and rich cultural history."),
            ("What is the capital of France?",
             "I don't know."),
            ("What is the capital of France?", ""),
            ("Explain quantum computing in detail",
             "Quantum computing uses qubits that can exist in superposition states, "
             "enabling parallel computation. Unlike classical bits (0 or 1), qubits "
             "leverage quantum entanglement and interference for exponential speedups "
             "in specific problem classes like factoring and optimization."),
            ("Explain quantum computing in detail",
             "It's complicated."),
            ("How do I make pasta?",
             "1. Boil water with salt. 2. Add pasta, cook 8-10 min. "
             "3. Drain. 4. Toss with sauce. Serves 2-4."),
        ]

        scores = []
        for q, r in test_cases:
            s = fn(q, r)
            if s is None:
                return False, "Returned None"
            scores.append(float(s))

        if len(set(scores)) <= 1:
            return False, f"Constant output ({scores[0]})"

        if max(scores) - min(scores) < 0.05:
            return False, f"Near-constant range ({min(scores):.3f}-{max(scores):.3f})"

        return True, "Valid"

    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Runtime error: {e}"


def summarize_approach(code_str):
    """Extract a brief description of the approach from code structure.

    Used to tell Claude what approaches already exist so it avoids repeating them.
    """
    keywords = []
    code_lower = code_str.lower()

    feature_markers = [
        ("flesch", "Flesch readability"),
        ("syllable", "syllable counting"),
        ("sentence_len", "sentence length"),
        ("avg_word_len", "word length"),
        ("type.token", "type-token ratio"),
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
        ("regex", "regex pattern matching"),
        ("entropy", "entropy/information"),
        ("concrete", "concreteness"),
    ]
    for marker, label in feature_markers:
        if marker in code_lower:
            keywords.append(label)

    if not keywords:
        keywords = ["general heuristic"]
    return ", ".join(keywords[:4])


# ── Incremental State Management (from Brian's approach) ─────────────────

def load_existing_state(output_dir, manifest_path):
    """Load existing programs and manifest for resume capability."""
    manifest = []
    codes = {}

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        for entry in manifest:
            if entry["status"] == "success":
                fpath = os.path.join(output_dir, entry["filename"])
                if os.path.exists(fpath):
                    with open(fpath, "r") as f:
                        codes[entry["program_id"]] = f.read()

    return manifest, codes


def save_state(manifest, manifest_path):
    """Save manifest incrementally (crash-safe)."""
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ── CLI & Main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate 80 PAJAMA judge programs for one dataset."
    )
    p.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset: judgelm, pandalm, multipref, prometheus, preference_700K"
        ),
    )
    p.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face dataset repo (default: {HF_REPO_ID})",
    )
    p.add_argument(
        "--num-few-shot",
        type=int,
        default=NUM_FEW_SHOT_EXAMPLES,
        help="Number of few-shot examples sampled from the HF validation split",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_key = resolve_dataset_name(args.dataset)
    cfg = DATASET_CONFIGS[dataset_key]

    output_dir = os.path.join(SCRIPT_DIR, cfg["output_dir"])
    manifest_path = os.path.join(SCRIPT_DIR, cfg["manifest"])

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Export it before running:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Dataset:     {dataset_key}  (HF config: {cfg['hf_config']})")
    print(f"HF repo:     {args.repo_id}")
    print(f"Output dir:  {output_dir}")
    print(f"Manifest:    {manifest_path}")
    print(f"Few-shot:    {args.num_few_shot} examples from validation split")
    print(f"Has scores:  {cfg['has_scores']}")
    print()

    print(f"Loading {args.num_few_shot} few-shot examples from Hugging Face ...")
    examples = load_few_shot_examples(
        dataset_key,
        n=args.num_few_shot,
        seed=args.seed,
        repo_id=args.repo_id,
    )
    few_shot_text = format_few_shot_examples(examples)
    print(f"  Prepared {len(examples)} few-shot examples")

    # ── Resume from existing state ────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    manifest, existing_codes = load_existing_state(output_dir, manifest_path)

    completed_ids = {e["program_id"] for e in manifest if e["status"] == "success"}
    all_code_texts = list(existing_codes.values())

    # Per-heuristic approach tracking (for the "avoid repeating" prompt feature)
    heuristic_approaches = {h_id: [] for h_id in HEURISTICS}
    for entry in manifest:
        if entry["status"] == "success" and entry["program_id"] in existing_codes:
            h_id = entry["heuristic_id"]
            approach_desc = summarize_approach(existing_codes[entry["program_id"]])
            heuristic_approaches[h_id].append(approach_desc)

    if completed_ids:
        print(f"  Resuming: {len(completed_ids)}/{TOTAL_PROGRAMS} already done")

    # ── Build program plan ────────────────────────────────────────────────
    random.seed(SEED)
    program_plan = []
    for h_id in HEURISTICS:
        for variant in range(1, PROGRAMS_PER_HEURISTIC + 1):
            program_plan.append({
                "heuristic_id": h_id,
                "heuristic_name": HEURISTICS[h_id]["name"],
                "variant": variant,
            })

    assert len(program_plan) == TOTAL_PROGRAMS

    # ── Generate programs ─────────────────────────────────────────────────
    global_attempts = 0
    success_count = len(completed_ids)
    similarity_rejects = 0
    validation_rejects = 0

    for i, plan in enumerate(program_plan):
        program_id = i + 1

        if program_id in completed_ids:
            continue

        heuristic = HEURISTICS[plan["heuristic_id"]]
        h_id = plan["heuristic_id"]
        existing_approaches = heuristic_approaches.get(h_id, [])

        print(f"\n[{program_id:2d}/{TOTAL_PROGRAMS}] {plan['heuristic_name']} (v{plan['variant']})")

        best_code = None
        best_status = "failed"
        slot_attempts = 0

        while slot_attempts < MAX_RETRIES_PER_SLOT and global_attempts < MAX_GLOBAL_ATTEMPTS:
            slot_attempts += 1
            global_attempts += 1

            try:
                prompt = build_prompt(
                    heuristic, few_shot_text, plan["variant"],
                    existing_approaches=existing_approaches if existing_approaches else None
                )

                api_kwargs = dict(
                    model=MODEL,
                    max_tokens=5000,
                    messages=[{"role": "user", "content": prompt}],
                )
                if USE_THINKING:
                    api_kwargs["thinking"] = {"type": "adaptive"}
                else:
                    api_kwargs["temperature"] = 0.713

                raw = client.messages.create(**api_kwargs)

                text_block = next(
                    (b for b in raw.content if b.type == "text"), None
                )
                if text_block is None:
                    validation_rejects += 1
                    print(f"  attempt {slot_attempts}: NO TEXT BLOCK in response")
                    continue
                code = extract_code(text_block.text)

                # Validation check
                is_valid, msg = validate_code(code)
                if not is_valid:
                    validation_rejects += 1
                    print(f"  attempt {slot_attempts}: INVALID ({msg})")
                    continue

                # Similarity check (Brian's key innovation)
                if is_too_similar(code, all_code_texts, SIMILARITY_THRESHOLD):
                    similarity_rejects += 1
                    print(f"  attempt {slot_attempts}: TOO SIMILAR (>{SIMILARITY_THRESHOLD:.0%})")
                    continue

                best_code = code
                best_status = "success"
                break

            except Exception as e:
                print(f"  attempt {slot_attempts}: API ERROR ({e})")
                best_status = f"error: {e}"

        # Save result
        filename = f"judge_{program_id}.py"
        filepath = os.path.join(output_dir, filename)

        if best_code is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(best_code)

            all_code_texts.append(best_code)
            approach_desc = summarize_approach(best_code)
            heuristic_approaches[h_id].append(approach_desc)

            success_count += 1
            print(f"  -> {filename} OK (approach: {approach_desc})")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    f"# FAILED after {slot_attempts} attempts: {best_status}\n"
                    f"def judging_function(query, response):\n"
                    f"    return len(response)\n"
                )
            print(f"  -> {filename} FAILED ({best_status})")

        entry = {
            "program_id": program_id,
            "filename": filename,
            "dataset": dataset_key,
            "hf_config": cfg["hf_config"],
            "hf_repo": args.repo_id,
            "heuristic_id": h_id,
            "heuristic_name": plan["heuristic_name"],
            "variant": plan["variant"],
            "status": best_status,
        }
        if best_code is not None:
            entry["approach_summary"] = summarize_approach(best_code)

        manifest = [e for e in manifest if e.get("program_id") != program_id]
        manifest.append(entry)
        manifest.sort(key=lambda x: x["program_id"])

        save_state(manifest, manifest_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Succeeded:           {success_count}/{TOTAL_PROGRAMS}")
    print(f"  Similarity rejects:  {similarity_rejects}")
    print(f"  Validation rejects:  {validation_rejects}")
    print(f"  Total API calls:     {global_attempts + len(completed_ids)}")
    print(f"  Programs:            {output_dir}/")
    print(f"  Manifest:            {manifest_path}")
    print(f"{'=' * 60}")

    # Per-heuristic breakdown
    print(f"\nPer-heuristic breakdown:")
    for h_id, h in HEURISTICS.items():
        h_programs = [e for e in manifest
                      if e["heuristic_id"] == h_id and e["status"] == "success"]
        print(f"  {h['name']:<45s}: {len(h_programs)}/{PROGRAMS_PER_HEURISTIC}")


if __name__ == "__main__":
    main()