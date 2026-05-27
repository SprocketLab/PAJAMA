"""
PAJAMA Workflow — Streamlit demo.

Run with:
    streamlit run app.py

Top workflow (no sidebar):
  1. Start    → Mock / Live mode + load data
  2. Prompt   → generation template (+ optional Claude chat in Live mode)
  3. Programs → 80 judges as rows (Edit / Chat / Delete)
  4. Results  → aggregation + labeled output
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline as P  # noqa: E402
import generation as G  # noqa: E402


# ── Paths ────────────────────────────────────────────────────────────────
PAJAMA_ROOT = THIS_DIR.parent
MOCK_JUDGES_DIR = PAJAMA_ROOT / "synthesized_programmatic_judges" / "judgelm"
MOCK_MANIFEST = PAJAMA_ROOT / "synthesized_programmatic_judges" / "manifest_judgelm.json"
MOCK_SAMPLE = THIS_DIR / "examples" / "judgelm_sample_60.jsonl"

# Cached scores + gold labels copied from pajama_workflow/judgelm_outputs.
# The demo re-runs the production pipeline (run.py logic) on these arrays.
DEMO_OUTPUTS = THIS_DIR / "outputs"
DEMO_VAL_S1 = DEMO_OUTPUTS / "val_s1.npy"
DEMO_VAL_S2 = DEMO_OUTPUTS / "val_s2.npy"
DEMO_TEST_S1 = DEMO_OUTPUTS / "test_s1.npy"
DEMO_TEST_S2 = DEMO_OUTPUTS / "test_s2.npy"
DEMO_PIPELINE_SUMMARY = DEMO_OUTPUTS / "pipeline_summary.json"

MOCK_SAMPLE_SIZE = 60

# ── Branding ────────────────────────────────────────────────────────────
# Pajamas icon by Freepik (https://www.flaticon.com/free-icon/pajamas_2892174),
# bundled locally so the demo doesn't depend on a remote CDN at runtime.
ICON_PATH = THIS_DIR / "assets" / "pajamas_icon.png"


def _icon_data_uri() -> str:
    """Return the pajamas icon as a base64 data URI for inline HTML use.

    Falls back to an empty string if the icon file is missing so the rest
    of the app still renders.
    """
    if not ICON_PATH.exists():
        return ""
    try:
        return "data:image/png;base64," + base64.b64encode(ICON_PATH.read_bytes()).decode()
    except Exception:
        return ""


ICON_DATA_URI = _icon_data_uri()


# ── Page setup ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAJAMA Workflow",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ── Session state ───────────────────────────────────────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("mode", "mock")
    ss.setdefault("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
    ss.setdefault("rows", None)  # list[dict] of (query, response1, response2[, verdict])
    ss.setdefault("rows_source", "")  # "mock", "uploaded", etc.
    ss.setdefault("orig_indices", None)  # for mock mode: indices into cached test arrays
    ss.setdefault("programs", None)  # list[Program]
    ss.setdefault("aggregation", None)  # AggregationResult (live mode)
    ss.setdefault("production_result", None)  # P.ProductionPipelineResult (mock mode)
    ss.setdefault("production_threshold", None)  # threshold_max last run
    ss.setdefault("prompt_template", G.DEFAULT_PROMPT_TEMPLATE)
    ss.setdefault("prompt_chat", [])  # list[{role, content}]
    ss.setdefault("program_chats", {})  # {program_id: list[{role, content}]}
    ss.setdefault("editing_program_id", None)
    ss.setdefault("abstain_band", 0.14)
    ss.setdefault("workflow_step", "setup")
    ss.setdefault("program_search", "")


init_state()

WORKFLOW_STEPS = [
    ("Load Dataset", "setup"),
    ("Synthesis Prompt", "prompt"),
    ("Generated Programs", "programs"),
    ("Verdict Aggregation", "results"),
]

_APPLE_CSS = """
<style>
    /* Hide sidebar — everything lives in the main canvas */
    section[data-testid="stSidebar"], section[data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    section.main .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.75rem !important;
        max-width: 46rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        overflow: visible !important;
    }
    section.main [data-testid="stVerticalBlock"],
    section.main [data-testid="stMarkdownContainer"],
    section.main [data-testid="stMarkdown"] {
        overflow: visible !important;
    }
    div[data-testid="stMarkdownContainer"]:has(.pajama-hero),
    div[data-testid="stMarkdown"]:has(.pajama-hero),
    .pajama-hero-outer {
        background: transparent !important;
        overflow: visible !important;
        padding-top: 2px;
    }
    .pajama-hero {
        background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 55%, #f9fafb 100%);
        border: 1px solid #e8ecf4;
        border-radius: 20px !important;
        padding: 1.25rem 1.5rem;
        margin: 0.25rem 0 1rem 0;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .pajama-hero h1 { font-size: 1.65rem; font-weight: 650; letter-spacing: -0.02em; margin: 0; }
    .pajama-hero p { color: #6b7280; margin: 0.35rem 0 0 0; font-size: 0.95rem; }
    .step-rail {
        display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.75rem 0 1.25rem 0;
    }
    .step-pill {
        padding: 0.45rem 1rem; border-radius: 999px; font-size: 0.88rem; font-weight: 500;
        border: 1px solid #e5e7eb; background: #fff; color: #6b7280;
    }
    .step-pill.active {
        background: #007aff; border-color: #007aff; color: #fff;
        box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25);
    }
    .card-panel {
        background: #fff; border: 1px solid #e8ecf4; border-radius: 16px;
        padding: 1.25rem 1.35rem; margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .card-panel h3 { margin: 0 0 0.35rem 0; font-size: 1.05rem; font-weight: 600; }
    .card-panel .sub { color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem; }
    .prog-table-head {
        font-size: 0.78rem; font-weight: 600; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.04em;
        padding: 0.35rem 0 0.5rem 0; border-bottom: 1px solid #eef0f4;
    }
    .prog-row {
        padding: 0.4rem 0; border-bottom: 1px solid #f3f4f6;
        align-items: center;
    }
    div[data-testid="stDialog"] {
        border-radius: 18px !important;
        box-shadow: 0 24px 48px rgba(15, 23, 42, 0.18) !important;
    }
</style>
"""


def _inject_app_css() -> None:
    st.markdown(_APPLE_CSS, unsafe_allow_html=True)


def _render_app_header() -> None:
    icon_html = (
        f'<img src="{ICON_DATA_URI}" width="32" height="32" style="border-radius:8px;" />'
        if ICON_DATA_URI
        else ""
    )
    st.markdown(
        f"""
        <div class="pajama-hero-outer">
          <div class="pajama-hero">
            <div style="display:flex;align-items:center;gap:12px;">
              {icon_html}
              <div>
                <h1>PAJAMA Workflow</h1>
                <p>Synthesized Programmatic Judges for Scalable Preference Evaluation.</p>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_step_nav() -> None:
    step = st.session_state.workflow_step
    cols = st.columns(len(WORKFLOW_STEPS))
    for i, (label, key) in enumerate(WORKFLOW_STEPS):
        with cols[i]:
            if st.button(
                label,
                key=f"nav_{key}",
                width='stretch',
                type="primary" if key == step else "secondary",
            ):
                st.session_state.workflow_step = key
                st.rerun()

    idx = next(i for i, (_, k) in enumerate(WORKFLOW_STEPS) if k == step)
    has_back = idx > 0
    has_continue = idx < len(WORKFLOW_STEPS) - 1

    if has_back and has_continue:
        nav_l, nav_r = st.columns(2)
        with nav_l:
            if st.button("← Back", width='stretch'):
                st.session_state.workflow_step = WORKFLOW_STEPS[idx - 1][1]
                st.rerun()
        with nav_r:
            if st.button("Continue →", width='stretch'):
                st.session_state.workflow_step = WORKFLOW_STEPS[idx + 1][1]
                st.rerun()
    elif has_back:
        if st.button("← Back", width='stretch'):
            st.session_state.workflow_step = WORKFLOW_STEPS[idx - 1][1]
            st.rerun()
    elif has_continue:
        if st.button("Continue →", width='stretch'):
            st.session_state.workflow_step = WORKFLOW_STEPS[idx + 1][1]
            st.rerun()


# ── Helpers ─────────────────────────────────────────────────────────────


def _load_mock_programs():
    if not MOCK_JUDGES_DIR.exists():
        st.error(f"Mock judges directory not found: {MOCK_JUDGES_DIR}")
        return None
    manifest = MOCK_MANIFEST if MOCK_MANIFEST.exists() else None
    return P.load_programs_from_dir(str(MOCK_JUDGES_DIR), str(manifest) if manifest else None)


def _load_mock_sample():
    if not MOCK_SAMPLE.exists():
        st.error(f"Mock sample missing: {MOCK_SAMPLE}")
        return None, None
    rows = P.load_jsonl(str(MOCK_SAMPLE))[:MOCK_SAMPLE_SIZE]
    orig = [int(r.get("_orig_test_index", -1)) for r in rows]
    return rows, orig


def _bootstrap_mock():
    rows, orig = _load_mock_sample()
    if rows is None:
        return
    progs = _load_mock_programs()
    if progs is None:
        return
    st.session_state.rows = rows
    st.session_state.orig_indices = orig
    st.session_state.rows_source = f"mock: judgelm sample ({len(rows)} samples)"
    st.session_state.programs = progs
    st.session_state.aggregation = None
    st.session_state.production_result = None
    st.session_state.production_threshold = None


def _selected_programs():
    if st.session_state.programs is None:
        return []
    return [p for p in st.session_state.programs if p.selected and p.fn is not None]


def _run_mock_production_pipeline(
    threshold_max: float,
    progress_callback=None,
) -> P.ProductionPipelineResult | None:
    """Execute production pipeline, persist summary under demo/outputs/."""
    arrays = P.load_demo_val_arrays(DEMO_OUTPUTS)
    if arrays is None:
        return None
    val_s1, val_s2, y_val = arrays
    result = P.run_from_cached_scores(
        val_s1,
        val_s2,
        y_val,
        threshold_max=float(threshold_max),
        progress_callback=progress_callback,
    )
    P.save_summary(result.summary, DEMO_PIPELINE_SUMMARY)
    st.session_state.production_result = result
    st.session_state.production_threshold = float(threshold_max)
    st.session_state.aggregation = None
    return result


def _pipeline_threshold_stale() -> bool:
    """True when the slider value differs from the last pipeline run."""
    pipe = st.session_state.production_result
    if pipe is None:
        return False
    last = st.session_state.production_threshold
    if last is None:
        return True
    return abs(float(last) - float(st.session_state.abstain_band)) > 1e-9


def _mock_pipeline_ready() -> bool:
    return P.load_demo_val_arrays(DEMO_OUTPUTS) is not None



def _score_with_programs(rows, programs, progress_text="Scoring"):
    """Score (rows × programs). Uses cached arrays in mock mode when possible."""
    use_cache = (
        st.session_state.mode == "mock"
        and st.session_state.orig_indices is not None
        and all(i >= 0 for i in st.session_state.orig_indices)
        and DEMO_TEST_S1.exists()
        and DEMO_TEST_S2.exists()
        and len(programs) == 80
        and all(p.program_id == i + 1 for i, p in enumerate(programs))
        and not any(p.dirty for p in programs)
    )
    if use_cache:
        idx = np.asarray(st.session_state.orig_indices, dtype=int)
        s1_full = np.load(DEMO_TEST_S1)
        s2_full = np.load(DEMO_TEST_S2)
        # cached arrays are aligned: column j -> judge_(j+1). Pick selected columns.
        cols = [p.program_id - 1 for p in programs]
        s1 = s1_full[idx][:, cols]
        s2 = s2_full[idx][:, cols]
        return s1, s2

    prog_bar = st.progress(0.0, text=progress_text)

    def cb(i, n):
        prog_bar.progress(i / n, text=f"{progress_text} {i}/{n}")

    s1, s2 = P.collect_raw_scores(rows, programs, progress_cb=cb)
    prog_bar.empty()
    return s1, s2


# ── Dialogs (Edit / Chat pop-ups) ─────────────────────────────────────────


@st.dialog("Edit judge program", width="large")
def _program_edit_dialog(program_id: int) -> None:
    progs = st.session_state.programs or []
    p = next((x for x in progs if x.program_id == program_id), None)
    if p is None:
        st.warning("Program not found.")
        return

    h_meta = G.HEURISTICS.get(p.heuristic_id or 0, {})
    st.caption(
        f"{p.display_name} · {h_meta.get('name', '—')} · variant {p.variant or '?'}"
    )
    new_code = st.text_area("Python code", value=p.code, height=360, label_visibility="collapsed")
    s1, s2, s3 = st.columns(3)
    with s1:
        if st.button("Save", type="primary", width='stretch'):
            if new_code != p.code:
                p.dirty = True
            p.code = new_code
            try:
                p.fn = P.compile_program(new_code)
                p.status = "success"
                p.approach_summary = G.summarize_approach(new_code)
                st.session_state.aggregation = None
                st.session_state.production_result = None
                st.success("Saved.")
            except Exception as e:
                p.fn = None
                p.status = f"compile_error: {e}"
                st.error(str(e))
    with s2:
        if st.button("Recompile", width='stretch'):
            try:
                p.fn = P.compile_program(p.code)
                p.status = "success"
                st.success("OK")
            except Exception as e:
                p.fn = None
                st.error(str(e))
    with s3:
        if st.button("Close", width='stretch'):
            st.rerun()


@st.dialog("Chat with Claude", width="large")
def _program_chat_dialog(program_id: int) -> None:
    progs = st.session_state.programs or []
    p = next((x for x in progs if x.program_id == program_id), None)
    if p is None:
        return

    if st.session_state.mode != "live" or not st.session_state.api_key:
        st.info("Switch to **Live** mode on the Start step and add your API key to use chat.")
        if st.button("Close"):
            st.rerun()
        return

    st.session_state.program_chats.setdefault(program_id, [])
    history = st.session_state.program_chats[program_id]

    with st.container(height=320, border=False):
        for turn in history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

    user_msg = st.chat_input("Ask Claude to improve this judge…")
    if user_msg:
        history.append({"role": "user", "content": user_msg})
        try:
            with st.spinner("Thinking…"):
                reply, new_code = G.chat_regenerate(
                    current_code=p.code,
                    user_message=user_msg,
                    history=history[:-1],
                    api_key=st.session_state.api_key,
                )
            history.append({"role": "assistant", "content": reply})
            if new_code:
                st.session_state[f"_pending_code_{program_id}"] = new_code
        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()

    pending = st.session_state.get(f"_pending_code_{program_id}")
    if pending:
        st.code(pending[:2500] + ("…" if len(pending) > 2500 else ""), language="python")
        a, b = st.columns(2)
        with a:
            if st.button("Use this code", type="primary", width='stretch'):
                try:
                    p.code = pending
                    p.fn = P.compile_program(pending)
                    p.status = "success"
                    p.dirty = True
                    p.approach_summary = G.summarize_approach(pending)
                    st.session_state.pop(f"_pending_code_{program_id}", None)
                    st.session_state.aggregation = None
                    st.success("Updated.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with b:
            if st.button("Discard", width='stretch'):
                st.session_state.pop(f"_pending_code_{program_id}", None)
                st.rerun()


def _render_step_setup() -> None:
    st.markdown(
        """
        <div class="card-panel">
          <h3>How do you want to run PAJAMA?</h3>
          <p class="sub">You can pick a mode, load/upload your data, then tap <strong>Continue</strong> to move on.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Mode",
        options=["mock", "live"],
        format_func=lambda x: (
            "Demo — try instantly with an example dataset (using JudgeLM samples)."
            if x == "mock"
            else "Live — upload your own data and calling Claude API for new programs."
        ),
        index=0 if st.session_state.mode == "mock" else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    if mode == "mock":
        st.markdown(
            '<div class="card-panel"><h3>Load an Example Dataset</h3>'
            '<p class="sub">60 sample pairs + 80 pre-built judges (no API key needed).</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Load an Example Dataset", type="primary", width='stretch'):
            _bootstrap_mock()
            st.success("Ready — 60 samples and 80 programmatic judges loaded.")
            st.rerun()
    else:
        st.markdown(
            '<div class="card-panel"><h3>Submit Your API key</h3>'
            '<p class="sub">Generate and edit your own programmatic judges.</p></div>',
            unsafe_allow_html=True,
        )
        st.session_state.api_key = st.text_input(
            "Anthropic API key",
            value=st.session_state.api_key,
            type="password",
            label_visibility="collapsed",
        )
        uploaded = st.file_uploader(
            "Upload JSONL (query, response1, response2)",
            type=["jsonl", "json"],
            label_visibility="collapsed",
        )
        if uploaded is not None and st.button("Use this file", width='stretch'):
            text = uploaded.read().decode("utf-8")
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            st.session_state.rows = rows
            st.session_state.orig_indices = None
            st.session_state.rows_source = f"uploaded: {uploaded.name} ({len(rows)} samples)"
            st.session_state.aggregation = None
            st.session_state.production_result = None
            st.success(f"Loaded {len(rows)} samples.")
            st.rerun()

    if st.session_state.rows is not None:
        n = len(st.session_state.rows)
        st.success(f"Dataset ready — {n} pairwise samples.")
        with st.expander("Preview data"):
            preview = pd.DataFrame(
                [
                    {
                        "query": str(r.get("query", "")),
                        "response1": str(r.get("response1", "")),
                        "response2": str(r.get("response2", "")),
                    }
                    for r in st.session_state.rows[:30]
                ]
            )
            st.dataframe(preview, width='stretch', hide_index=True)


def _render_step_prompt() -> None:
    st.markdown(
        """
        <div class="card-panel">
          <h3>Our Prompt Template</h3>
          <p class="sub">This is the instruction sent to Claude Opus 4.6 to generate 80 programmatic judges. You can modify it by changing the code block below. Otherwise, press Continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    template = st.text_area(
        "Template",
        value=st.session_state.prompt_template,
        height=380,
        label_visibility="collapsed",
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save template", type="primary", width='stretch'):
            required = {
                "{heuristic_name}",
                "{heuristic_description}",
                "{avoid_block}",
                "{variant}",
                "{few_shot_text}",
            }
            missing = [p for p in required if p not in template]
            if missing:
                st.error(f"Missing placeholders: {missing}")
            else:
                st.session_state.prompt_template = template
                st.success("Saved.")
    with b2:
        if st.button("Reset to default", width='stretch'):
            st.session_state.prompt_template = G.DEFAULT_PROMPT_TEMPLATE
            st.rerun()

    if st.session_state.mode == "live" and st.session_state.api_key:
        st.markdown("##### Refine template with Claude")
        for turn in st.session_state.prompt_chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
        user_q = st.chat_input("Ask Claude to improve the template…")
        if user_q:
            st.session_state.prompt_chat.append({"role": "user", "content": user_q})
            try:
                with st.spinner("Thinking…"):
                    reply, new_template = G.chat_refine_prompt(
                        current_template=st.session_state.prompt_template,
                        user_message=user_q,
                        history=st.session_state.prompt_chat[:-1],
                        api_key=st.session_state.api_key,
                    )
                st.session_state.prompt_chat.append({"role": "assistant", "content": reply})
                if new_template:
                    st.session_state["_pending_prompt"] = new_template
            except Exception as e:
                st.session_state.prompt_chat.append({"role": "assistant", "content": str(e)})
            st.rerun()
        pending = st.session_state.get("_pending_prompt")
        if pending:
            if st.button("Use proposed template", type="primary"):
                st.session_state.prompt_template = pending
                st.session_state.pop("_pending_prompt", None)
                st.rerun()


def _render_step_programs() -> None:
    progs = st.session_state.programs

    st.markdown(
        """
        <div class="card-panel">
          <h3>Your 80 judges</h3>
          <p class="sub">Each row represents a programmatic judge. You can customize them by using Edit, Chat, or Delete — one judge at a time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tb1, tb2 = st.columns(2)
    with tb1:
        gen_ok = (
            st.session_state.mode == "live"
            and st.session_state.api_key
            and st.session_state.rows is not None
        )
        if st.button("Generate 80 judges", disabled=not gen_ok, width='stretch'):
            _run_full_generation()
            st.rerun()
    with tb2:
        if st.button("Reload (demo)", disabled=st.session_state.mode != "mock", width='stretch'):
            st.session_state.programs = _load_mock_programs()
            st.session_state.production_result = None
            st.rerun()

    if not progs:
        st.info("Load sample data on the **Start** step, or generate judges in Live mode.")
        return

    st.session_state.program_search = st.text_input(
        "Search judges",
        value=st.session_state.program_search,
        placeholder="select a judge by name, e.g., judge_12",
        label_visibility="collapsed",
    )
    q = st.session_state.program_search.strip().lower()
    filtered = [
        p for p in sorted(progs, key=lambda x: x.program_id)
        if not q or q in p.display_name.lower() or q in str(p.program_id)
    ]

    with st.container(height=480, border=True):
        h0, h1, h2, h3 = st.columns([2.5, 1, 1, 1])
        with h0:
            st.caption("**judge_id**")
        with h1:
            st.caption("**Edit**")
        with h2:
            st.caption("**Chat**")
        with h3:
            st.caption("**Delete**")

        for p in filtered:
            ok = p.fn is not None
            c0, c1, c2, c3 = st.columns([2.5, 1, 1, 1])
            with c0:
                dot = "●" if ok else "○"
                color = "#34c759" if ok else "#ff3b30"
                st.markdown(
                    f'<span style="color:{color};font-size:0.65rem;">{dot}</span> '
                    f"**{p.display_name}**",
                    unsafe_allow_html=True,
                )
            with c1:
                if st.button("Edit", key=f"edit_{p.program_id}", width='stretch'):
                    _program_edit_dialog(p.program_id)
            with c2:
                if st.button("Chat", key=f"chat_{p.program_id}", width='stretch'):
                    _program_chat_dialog(p.program_id)
            with c3:
                if st.button("Delete", key=f"del_{p.program_id}", width='stretch'):
                    st.session_state.programs = [
                        x for x in st.session_state.programs if x.program_id != p.program_id
                    ]
                    st.session_state.aggregation = None
                    st.rerun()


def _run_full_generation() -> None:
    """Generate 80 programs end-to-end via Claude (live mode)."""
    rows = st.session_state.rows
    if not rows:
        st.error("Upload data first.")
        return
    few_shot_text = G.format_few_shot_from_rows(rows, n=min(10, len(rows)))
    progs: list[P.Program] = []
    h_approaches: dict[int, list[str]] = {h: [] for h in G.HEURISTICS}
    pid = 0
    plan = [(h, v) for h in G.HEURISTICS for v in range(1, 9)]
    bar = st.progress(0.0, text="Generating judges…")
    for k, (h_id, variant) in enumerate(plan):
        pid += 1
        try:
            outcome = G.generate_program(
                heuristic_id=h_id,
                variant=variant,
                few_shot_text=few_shot_text,
                prompt_template=st.session_state.prompt_template,
                existing_approaches=h_approaches[h_id] or None,
                api_key=st.session_state.api_key,
            )
            try:
                fn = P.compile_program(outcome.code)
                status = "success"
            except Exception as e:
                fn = None
                status = f"compile_error: {e}"
            progs.append(
                P.Program(
                    program_id=pid,
                    filename=f"judge_{pid}.py",
                    code=outcome.code,
                    heuristic_id=h_id,
                    heuristic_name=G.HEURISTICS[h_id]["name"],
                    variant=variant,
                    approach_summary=outcome.approach_summary,
                    status=status,
                    fn=fn,
                )
            )
            h_approaches[h_id].append(outcome.approach_summary)
        except Exception as e:
            progs.append(
                P.Program(
                    program_id=pid,
                    filename=f"judge_{pid}.py",
                    code=f"# generation failed: {e}\ndef judging_function(query, response):\n    return len(response)\n",
                    heuristic_id=h_id,
                    heuristic_name=G.HEURISTICS[h_id]["name"],
                    variant=variant,
                    approach_summary="general heuristic",
                    status=f"error: {e}",
                    fn=None,
                )
            )
        bar.progress((k + 1) / len(plan), text=f"Generating… {k + 1}/{len(plan)}")
    bar.empty()
    st.session_state.programs = progs
    st.session_state.aggregation = None
    st.success(f"Generated {len(progs)} judges.")


def _render_step_results() -> None:
    rows = st.session_state.rows
    pipe = st.session_state.production_result
    agg = st.session_state.aggregation

    st.markdown(
        """
        <div class="card-panel">
          <h3>Aggregate &amp; download predictions</h3>
          <p class="sub">Set how strict judges should be, run the weak supervision pipeline, then review accuracy and predicted preferences.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.abstain_band = st.slider(
        "Abstain threshold",
        min_value=0.0,
        max_value=0.30,
        value=float(st.session_state.abstain_band),
        step=0.01,
        help="Higher = more cautious judges. Default 0.14 for the demo.",
    )

    if st.session_state.mode == "mock":
        c1, c2 = st.columns(2)
        with c1:
            run_pipeline = st.button(
                "Run pipeline",
                type="primary",
                disabled=not _mock_pipeline_ready(),
                width='stretch',
            )
        with c2:
            if st.button("Clear results", disabled=pipe is None, width='stretch'):
                st.session_state.production_result = None
                st.session_state.production_threshold = None
                st.rerun()

        progress_slot = st.empty()
        if run_pipeline:
            try:
                progress_bar = progress_slot.progress(0.0, text="Starting…")

                def _on_progress(frac: float, msg: str) -> None:
                    progress_bar.progress(frac, text=msg)

                t0 = time.time()
                _run_mock_production_pipeline(
                    float(st.session_state.abstain_band),
                    progress_callback=_on_progress,
                )
                progress_bar.progress(1.0, text="Done.")
                st.toast(f"Finished in {time.time()-t0:.1f}s")
                st.rerun()
            except Exception as e:
                progress_slot.error(str(e))

        if _pipeline_threshold_stale() and pipe is not None:
            st.warning("Threshold changed — click **Run pipeline** to refresh.")

        pipe = st.session_state.production_result
        if pipe is None:
            st.info("Click **Run pipeline** above to see results.")
            return

        # Compute predictions on the displayed test samples first so the
        # Coverage metric below is consistent with the abstained column in the table.
        labeled: list[dict] = rows or []
        hard: np.ndarray | None = None
        soft: np.ndarray | None = None
        if (
            rows is not None
            and st.session_state.orig_indices is not None
            and DEMO_TEST_S1.exists()
            and DEMO_TEST_S2.exists()
        ):
            idx = np.asarray(st.session_state.orig_indices, dtype=int)
            s1_test = np.load(DEMO_TEST_S1)[idx]
            s2_test = np.load(DEMO_TEST_S2)[idx]
            hard, soft = P.predict_on_scores(s1_test, s2_test, pipe)
            labeled = P.attach_predictions_to_rows(rows, hard, soft)

        test_coverage = float((hard != -1).mean()) if hard is not None else None

        # Accuracy on non-abstained test rows that have a ground-truth verdict.
        test_accuracy: float | None = None
        if hard is not None and rows is not None:
            verdicts = [r.get("verdict") for r in rows]
            valid = [
                i for i, (h, v) in enumerate(zip(hard, verdicts))
                if h != -1 and str(v) in ("1", "2")
            ]
            if valid:
                y_true = np.array([int(verdicts[i]) - 1 for i in valid])  # {1,2} → {0,1}
                y_pred = hard[valid]
                test_accuracy = float((y_true == y_pred).mean())

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Agreement with ground truth",
            f"{100*test_accuracy:.1f}%" if test_accuracy is not None else "—",
        )
        m2.metric(
            "Coverage",
            f"{100*test_coverage:.1f}%" if test_coverage is not None else "—",
        )
        m3.metric("# of Selected Judges", pipe.summary.get("best_k", "—"))

        st.markdown("##### Selected judges")
        df = pd.DataFrame(
            {
                "program": [f"judge_{pid}" for pid in pipe.selected_program_ids],
                "aggregation weight": pipe.weights.astype(float),
                "coverage": pipe.coverage.astype(float),
                "conflict": pipe.conflicts.astype(float),
            }
        )
        st.dataframe(df, width='stretch', hide_index=True)

    else:
        sel = _selected_programs()
        if st.button(
            "Run aggregation",
            type="primary",
            disabled=not (sel and rows),
            width='stretch',
        ):
            s1, s2 = _score_with_programs(rows, sel)
            with st.spinner("Aggregating…"):
                st.session_state.aggregation = P.aggregate(
                    s1=s1,
                    s2=s2,
                    program_ids=[p.program_id for p in sel],
                    abstain_band=float(st.session_state.abstain_band),
                )
            st.session_state.production_result = None
            st.rerun()

        agg = st.session_state.aggregation
        if agg is None:
            if rows is None:
                st.info("Load data on the **Start** step first.")
            else:
                st.info("Click **Run aggregation** above.")
            return

        st.metric("Coverage", f"{100*(agg.hard != -1).mean():.1f}%")
        labeled = P.label_jsonl_export(rows, agg)

    st.markdown("---")
    st.markdown("##### Your Predicted Preferences (Annotated by Programs)")

    if not labeled:
        st.info("Load data on the **Start** step first.")
        return

    out = pd.DataFrame(
        [
            {
                "index": i,
                "query": str(r.get("query", "")),
                "abstained": r.get("pajama_abstained", "—"),
                "prediction": r.get("pajama_predicted_verdict", "—"),
                "ground truth": r.get("verdict", "—"),
                "P(response_2 wins)": (
                    f"{r['pajama_response2_prob']:.2f}"
                    if r.get("pajama_response2_prob") is not None
                    else "—"
                ),
            }
            for i, r in enumerate(labeled)
        ]
    )
    st.dataframe(out, width='stretch', hide_index=True)
    buf = io.StringIO()
    for r in labeled:
        buf.write(json.dumps(r) + "\n")
    st.download_button(
        "Download labeled JSONL",
        data=buf.getvalue(),
        file_name="pajama_labeled.jsonl",
        width='stretch',
    )


# ── Main layout ─────────────────────────────────────────────────────────
_inject_app_css()
_render_app_header()
_render_step_nav()

_step = st.session_state.workflow_step
if _step == "setup":
    _render_step_setup()
elif _step == "prompt":
    _render_step_prompt()
elif _step == "programs":
    _render_step_programs()
else:
    _render_step_results()
