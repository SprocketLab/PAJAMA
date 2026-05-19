"""
PAJAMA — Judge Program Generator v4
=====================================
Combines the best of Brian's original approach with our curated heuristics:

  From Brian's code:
    - Code similarity dedup (rejects programs too similar to existing ones)
    - Incremental save / resume capability (crash-safe)
    - Pool-based rejection loop (keeps trying until target reached)
    - Rich structured metadata

  From our v3 pipeline analysis:
    - 10 curated heuristics (dropped noisy ones, added 2 new)
    - validate_code() with constant-output & crash detection + retry
    - Few-shot examples from validation set
    - Discriminative prompt design
    - Anthropic Claude API

Heuristic Selection (see more_heuristics_generate_programs.py for full rationale):
  1. Relevance to Query         — proven #1 area
  2. Language Quality            — proven #1 performer
  3. Completeness / Coverage     — strong performer
  4. Factual Accuracy            — core signal
  5. Logical Coherence           — argument structure
  6. Clarity and Conciseness     — anti-bloat
  7. Reasoning Transparency      — step-by-step (user's #17)
  8. Epistemic Calibration       — confidence calibration (#16)
  9. Structural Organization     — NEW: formatting & layout
 10. Evidence Density            — NEW: concrete vs vague

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    cd ~/pajama/pajama
    conda run -n snorkel_env python generate_judging_programs_v4.py
"""

import os
import re
import json
import random
import difflib
import anthropic

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "judgelm_val_500.jsonl"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "judge_programs")
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "program_manifest.json")
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


# ── Few-shot Examples ─────────────────────────────────────────────────────

def load_few_shot_examples(val_path, n=5, seed=42):
    random.seed(seed)
    with open(val_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    samples = random.sample(lines, min(n, len(lines)))
    examples = []
    for line in samples:
        data = json.loads(line)
        s1, s2 = data["score"]
        if s1 > s2:
            verdict = "Response A is better"
        elif s2 > s1:
            verdict = "Response B is better"
        else:
            verdict = "Tie"
        examples.append({
            "query": data["question_body"][:300],
            "response_a": data["answer1_body"][:400],
            "response_b": data["answer2_body"][:400],
            "score_a": s1, "score_b": s2,
            "verdict": verdict,
        })
    return examples


def format_few_shot_examples(examples):
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"--- Example {i} ---\n"
            f"Query: {ex['query']}\n\n"
            f"Response A (score={ex['score_a']}):\n{ex['response_a']}\n\n"
            f"Response B (score={ex['score_b']}):\n{ex['response_b']}\n\n"
            f"Ground-truth verdict: {ex['verdict']} "
            f"(gap = {abs(ex['score_a'] - ex['score_b'])})\n"
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Export it before running:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading {NUM_FEW_SHOT_EXAMPLES} few-shot examples ...")
    examples = load_few_shot_examples(VAL_DATA_PATH, n=NUM_FEW_SHOT_EXAMPLES, seed=SEED)
    few_shot_text = format_few_shot_examples(examples)
    print(f"  Loaded {len(examples)} examples")

    # ── Resume from existing state ────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest, existing_codes = load_existing_state(OUTPUT_DIR, MANIFEST_PATH)

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
        filepath = os.path.join(OUTPUT_DIR, filename)

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

        save_state(manifest, MANIFEST_PATH)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Succeeded:           {success_count}/{TOTAL_PROGRAMS}")
    print(f"  Similarity rejects:  {similarity_rejects}")
    print(f"  Validation rejects:  {validation_rejects}")
    print(f"  Total API calls:     {global_attempts + len(completed_ids)}")
    print(f"  Programs:            {OUTPUT_DIR}/")
    print(f"  Manifest:            {MANIFEST_PATH}")
    print(f"{'=' * 60}")

    # Per-heuristic breakdown
    print(f"\nPer-heuristic breakdown:")
    for h_id, h in HEURISTICS.items():
        h_programs = [e for e in manifest
                      if e["heuristic_id"] == h_id and e["status"] == "success"]
        print(f"  {h['name']:<45s}: {len(h_programs)}/{PROGRAMS_PER_HEURISTIC}")


if __name__ == "__main__":
    main()