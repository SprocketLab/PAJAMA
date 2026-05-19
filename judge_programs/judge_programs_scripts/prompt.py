"""
PAJAMA — Judge Program Generation Prompt
==========================================
Complete prompt used to generate judging programs via LLM.

The prompt instructs the model to write a Python scoring function for a
specific quality heuristic. It includes:
  1. Task description and role assignment
  2. One of 10 quality-evaluation heuristics (the "dimension")
  3. An optional "avoid" block listing previously generated approaches
  4. Few-shot examples drawn from the JudgeLM validation set
  5. Output format and constraint requirements
"""

# ── Quality-Evaluation Heuristics ─────────────────────────────────────────
# Each heuristic defines a single dimension along which LLM responses are
# scored.  The generator produces PROGRAMS_PER_HEURISTIC distinct programs
# per heuristic (default 8), yielding 80 programs total.

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


# ── Few-Shot Example Format ───────────────────────────────────────────────
# 10 examples (2% of validation set) are sampled from JudgeLM-val-500 and
# formatted as below.  Each example shows a query, two responses with their
# ground-truth scores, and the verdict.

FEW_SHOT_TEMPLATE = """\
--- Example {index} ---
Query: {query}

Response A (score={score_a}):
{response_a}

Response B (score={score_b}):
{response_b}

Ground-truth verdict: {verdict} (gap = {gap})
"""


# ── Main Generation Prompt ────────────────────────────────────────────────
# {heuristic_name}        — e.g. "Relevance to the Query"
# {heuristic_description} — the full description from HEURISTICS[id]
# {avoid_block}           — (optional) list of already-generated approaches
# {few_shot_text}         — concatenated FEW_SHOT_TEMPLATE instances
# {variant_index}         — current variant number (1..8)
# {programs_per_heuristic} — total variants per heuristic (8)

GENERATION_PROMPT = """\
You are an expert Python developer and AI evaluation researcher.

Your task: Write a Python function that evaluates the quality of an LLM-generated response
to a given query. The function should return a numeric score where HIGHER values indicate
BETTER quality.

EVALUATION STRATEGY — focus STRICTLY on this dimension:
{heuristic_name}: {heuristic_description}
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
- This is variant {variant_index}/{programs_per_heuristic} — use a MEANINGFULLY DIFFERENT
  algorithm, features, and scoring formula than other variants.
- Include comprehensive error handling (try/except) so the function never crashes.
- The function must handle edge cases (empty strings, very short/long inputs).
- Return scores in a reasonable numeric range (e.g., 0-10 or 0-100).
- Make the function DISCRIMINATIVE: it should produce clearly different scores for
  high-quality vs low-quality responses.

```python
def judging_function(query, response):
    # Your implementation here
```\
"""


# ── Avoid-Repetition Block ────────────────────────────────────────────────
# Injected into the prompt when prior variants already exist for the same
# heuristic, so the LLM is instructed to use a different algorithm.

AVOID_BLOCK_TEMPLATE = """\

The following approaches have ALREADY been implemented for this heuristic. \
You MUST use a substantially DIFFERENT algorithm:
{existing_approaches}
"""


# ── Helper: Build the full prompt string ──────────────────────────────────

PROGRAMS_PER_HEURISTIC = 8


def build_prompt(heuristic, few_shot_text, variant_index, existing_approaches=None):
    """Assemble the complete prompt for one program generation call.

    Parameters
    ----------
    heuristic : dict
        {"name": str, "description": str} from HEURISTICS.
    few_shot_text : str
        Pre-formatted few-shot examples string.
    variant_index : int
        Which variant (1-based) within this heuristic.
    existing_approaches : list[str] or None
        Short summaries of approaches already generated for this heuristic.

    Returns
    -------
    str
        The fully rendered prompt.
    """
    avoid_block = ""
    if existing_approaches:
        descs = "\n".join(f"  - Variant {i+1}: {d}" for i, d in enumerate(existing_approaches))
        avoid_block = AVOID_BLOCK_TEMPLATE.format(existing_approaches=descs)

    return GENERATION_PROMPT.format(
        heuristic_name=heuristic["name"],
        heuristic_description=heuristic["description"],
        avoid_block=avoid_block,
        few_shot_text=few_shot_text,
        variant_index=variant_index,
        programs_per_heuristic=PROGRAMS_PER_HEURISTIC,
    )


# ── Preview ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    placeholder_few_shot = (
        "--- Example 1 ---\n"
        "Query: What is the capital of France?\n\n"
        "Response A (score=9):\n"
        "Paris is the capital of France ...\n\n"
        "Response B (score=3):\n"
        "I don't know.\n\n"
        "Ground-truth verdict: Response A is better (gap = 6)\n"
    )

    print("=" * 70)
    print("EXAMPLE PROMPT (Heuristic 1, Variant 1, no prior approaches)")
    print("=" * 70)
    print(build_prompt(HEURISTICS[1], placeholder_few_shot, variant_index=1))
    print()

    print("=" * 70)
    print("EXAMPLE PROMPT (Heuristic 3, Variant 4, with prior approaches)")
    print("=" * 70)
    print(build_prompt(
        HEURISTICS[3], placeholder_few_shot, variant_index=4,
        existing_approaches=[
            "word overlap, Jaccard similarity",
            "sentence length, paragraph analysis",
            "n-gram coverage, TF-IDF",
        ],
    ))
