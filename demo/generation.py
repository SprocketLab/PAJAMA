"""
Claude-based program generation + per-program chat helpers.

Two flows:
  - generate_program(heuristic, variant, few_shot_text, prompt_template, ...)
      One-shot batch generation, used when the user clicks "Generate 80 programs".
  - chat_regenerate(program_code, chat_history, ...)
      Conversational edit of a single program; returns new code + assistant message.

Both reuse Claude Opus 4.6 with the same prompt scaffold from the main pajama
pipeline so the demo stays faithful to the production generator.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

try:
    import anthropic
except ImportError:  # the demo can still run in mock-only mode without anthropic installed
    anthropic = None  # type: ignore


MODEL = "claude-opus-4-6"

# 10 heuristics, same wording as the main PAJAMA generator.
HEURISTICS: dict[int, dict[str, str]] = {
    1: {
        "name": "Relevance to the Query",
        "description": (
            "Evaluate how semantically relevant the response is to the question asked. "
            "Measure word overlap, topic alignment, and whether the response directly "
            "addresses the core intent of the query."
        ),
    },
    2: {
        "name": "Language Quality and Readability",
        "description": (
            "Evaluate language quality and readability: grammar, spelling, punctuation, "
            "sentence variety, vocabulary richness, readability scores."
        ),
    },
    3: {
        "name": "Completeness and Coverage",
        "description": (
            "Evaluate completeness and thoroughness: are all aspects/sub-questions "
            "addressed, are edge cases covered, is there sufficient depth?"
        ),
    },
    4: {
        "name": "Factual Accuracy Indicators",
        "description": (
            "Evaluate indicators of factual reliability: citations, specific names/dates/"
            "numbers, appropriate hedging; penalize hallucination red-flags."
        ),
    },
    5: {
        "name": "Logical Coherence and Argument Structure",
        "description": (
            "Evaluate logical flow: well-structured arguments, smooth transitions, "
            "no internal contradictions or non-sequiturs."
        ),
    },
    6: {
        "name": "Clarity and Conciseness",
        "description": (
            "Evaluate clarity and conciseness: clear and efficient communication, "
            "no filler, no redundancy."
        ),
    },
    7: {
        "name": "Reasoning Transparency and Step-wise Formulation",
        "description": (
            "Evaluate how transparently the response shows its reasoning process: "
            "step-by-step breakdowns, visible intermediate conclusions."
        ),
    },
    8: {
        "name": "Epistemic Calibration and Uncertainty Communication",
        "description": (
            "Evaluate how the response communicates confidence and uncertainty: "
            "appropriate hedging, no overconfidence on speculative claims."
        ),
    },
    9: {
        "name": "Structural Organization and Formatting",
        "description": (
            "Evaluate structural organization and formatting: lists, headers, "
            "paragraphs, logical grouping, effective whitespace use."
        ),
    },
    10: {
        "name": "Evidence Density and Specificity",
        "description": (
            "Evaluate density of concrete evidence and specific details: examples, "
            "data points, named entities, precise numbers, actionable details."
        ),
    },
}


DEFAULT_PROMPT_TEMPLATE = """You are an expert Python developer and AI evaluation researcher.

Your task: Write a Python function that evaluates the quality of an LLM-generated response to a given query. The function should return a numeric score where HIGHER values indicate BETTER quality.

EVALUATION STRATEGY — focus STRICTLY on this dimension:
{heuristic_name}: {heuristic_description}
{avoid_block}
Here are some real examples from the user's dataset so you understand what realistic queries and responses look like:

{few_shot_text}

IMPORTANT REQUIREMENTS:
- Output ONLY valid, executable Python inside ```python ... ``` blocks. No prose.
- Function signature must be exactly: def judging_function(query, response):
- Return a single numeric score (int or float). Higher = better quality.
- Use standard Python only (re, math, collections, string, statistics). No heavy ML libs.
- This is variant {variant}/8 for this heuristic — use a MEANINGFULLY DIFFERENT algorithm than other variants for the same heuristic.
- Include comprehensive try/except so the function never crashes on edge cases.
- Return scores in a reasonable range (e.g., 0–10 or 0–100), and make the function DISCRIMINATIVE.

```python
def judging_function(query, response):
    # Your implementation here
```"""


# ── Few-shot formatting ──────────────────────────────────────────────────


def format_few_shot_from_rows(rows: list[dict], n: int = 5, max_q: int = 300, max_r: int = 400) -> str:
    parts: list[str] = []
    for i, row in enumerate(rows[:n], 1):
        q = str(row.get("query", ""))[:max_q]
        a = str(row.get("response1", ""))[:max_r]
        b = str(row.get("response2", ""))[:max_r]
        verdict = row.get("verdict")
        if verdict in (1, "1"):
            v_str = "Ground-truth verdict: Response A is better"
        elif verdict in (2, "2"):
            v_str = "Ground-truth verdict: Response B is better"
        else:
            v_str = "(no ground-truth verdict — unlabeled pair)"
        parts.append(
            f"--- Example {i} ---\n"
            f"Query: {q}\n\n"
            f"Response A:\n{a}\n\n"
            f"Response B:\n{b}\n\n"
            f"{v_str}\n"
        )
    return "\n".join(parts)


# ── Code extraction ──────────────────────────────────────────────────────


def extract_python_code(text: str) -> str:
    matches = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    if matches:
        for m in matches:
            if "def judging_function" in m:
                return m
        return matches[-1]
    return text.replace("```python", "").replace("```", "").strip()


def summarize_approach(code: str) -> str:
    """Mirror of the main PAJAMA generator's keyword-tag summary."""
    code_lower = code.lower()
    markers = [
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
        ("trigram", "char trigram similarity"),
        ("repetition", "repetition penalty"),
    ]
    tags = [label for marker, label in markers if marker in code_lower]
    if not tags:
        tags = ["general heuristic"]
    return ", ".join(tags[:4])


# ── Anthropic helpers ────────────────────────────────────────────────────


def _client(api_key: str | None = None):
    if anthropic is None:
        raise RuntimeError(
            "anthropic SDK not installed. `pip install anthropic` to enable live mode."
        )
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it or pass it via the sidebar."
        )
    return anthropic.Anthropic(api_key=key)


@dataclass
class GenerationOutcome:
    code: str
    raw_text: str
    approach_summary: str
    error: str | None = None


def build_generation_prompt(
    heuristic_id: int,
    variant: int,
    few_shot_text: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    existing_approaches: list[str] | None = None,
    heuristics: dict[int, dict[str, str]] | None = None,
) -> str:
    table = heuristics if heuristics is not None else HEURISTICS
    h = table[heuristic_id]
    avoid_block = ""
    if existing_approaches:
        descs = "\n".join(f"  - Variant {i+1}: {d}" for i, d in enumerate(existing_approaches))
        avoid_block = (
            "\nThe following approaches have ALREADY been implemented for this heuristic. "
            "You MUST use a substantially DIFFERENT algorithm:\n"
            f"{descs}\n"
        )
    return prompt_template.format(
        heuristic_name=h["name"],
        heuristic_description=h["description"],
        avoid_block=avoid_block,
        variant=variant,
        few_shot_text=few_shot_text,
    )


def generate_program(
    heuristic_id: int,
    variant: int,
    few_shot_text: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    existing_approaches: list[str] | None = None,
    api_key: str | None = None,
    temperature: float = 0.713,
    max_tokens: int = 5000,
    heuristics: dict[int, dict[str, str]] | None = None,
) -> GenerationOutcome:
    client = _client(api_key)
    prompt = build_generation_prompt(
        heuristic_id=heuristic_id,
        variant=variant,
        few_shot_text=few_shot_text,
        prompt_template=prompt_template,
        existing_approaches=existing_approaches,
        heuristics=heuristics,
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in msg.content if b.type == "text"), "")
    code = extract_python_code(text)
    return GenerationOutcome(code=code, raw_text=text, approach_summary=summarize_approach(code))


# ── Per-program chat ─────────────────────────────────────────────────────


CHAT_SYSTEM_PROMPT = """You are an expert Python developer helping the user iteratively refine a single PAJAMA judge program.

A judge program is a pure-Python function:

    def judging_function(query, response):
        ...
        return score  # higher = better quality

Rules you MUST follow when proposing new code:
- Output the FULL updated program inside ```python ... ``` (no diff format).
- Keep the exact signature: def judging_function(query, response).
- Pure Python only (re, math, collections, string, statistics). No external deps.
- Wrap risky logic in try/except so the function never raises.
- Make the change minimal and explainable in 1–2 sentences before the code block.

Always begin your reply with a brief 1-2 sentence explanation of what you changed and why,
then provide the full updated function in a single ```python``` block.
"""


def chat_regenerate(
    current_code: str,
    user_message: str,
    history: list[dict] | None = None,
    api_key: str | None = None,
    max_tokens: int = 4000,
) -> tuple[str, str]:
    """Send one user turn to Claude with the current program + chat history.

    Returns (assistant_reply_text, extracted_code_or_empty).
    """
    client = _client(api_key)
    history = history or []

    seed_user = (
        "Here is the current judge program. Treat it as the source of truth for our edits:\n\n"
        f"```python\n{current_code}\n```"
    )

    messages = [{"role": "user", "content": seed_user}]
    messages.append(
        {
            "role": "assistant",
            "content": "Got it — I have the current program in mind. What would you like me to change?",
        }
    )
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=CHAT_SYSTEM_PROMPT,
        messages=messages,
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    code = extract_python_code(text) if "```python" in text else ""
    return text, code


# ── Global prompt-template chat ──────────────────────────────────────────


PROMPT_CHAT_SYSTEM = """You help the user iteratively refine the PROMPT TEMPLATE used to generate PAJAMA judge programs.

The template is a string with five required placeholders:
  {heuristic_name}, {heuristic_description}, {avoid_block}, {variant}, {few_shot_text}

When the user asks for changes, output the FULL revised template inside a ```text``` block.
Briefly (1-2 sentences) explain the change before the code block.
Never drop any of the five placeholders.
"""


def chat_refine_prompt(
    current_template: str,
    user_message: str,
    history: list[dict] | None = None,
    api_key: str | None = None,
    max_tokens: int = 3000,
) -> tuple[str, str]:
    """Returns (assistant_reply_text, extracted_template_or_empty)."""
    client = _client(api_key)
    history = history or []

    seed = (
        "Here is the CURRENT prompt template (treat as ground truth for our edits):\n\n"
        f"```text\n{current_template}\n```"
    )
    messages = [
        {"role": "user", "content": seed},
        {
            "role": "assistant",
            "content": "Got the current template. What change would you like?",
        },
    ]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=PROMPT_CHAT_SYSTEM,
        messages=messages,
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    m = re.search(r"```text\n(.*?)\n```", text, re.DOTALL)
    template = m.group(1) if m else ""
    return text, template
