"""
PAJAMA Labeling Studio — Streamlit demo.

Run with:
    streamlit run app.py

Workflow:
  1. Sidebar  → choose Mock or Live mode, upload (or load sample) JSONL.
  2. Programs → view / edit / remove / select any of the 80 generated programs.
                Per-program chat to regenerate one program with Claude.
  3. Prompt   → view the prompt template used for generation; chat with Claude to refine it.
  4. Aggregate→ run the val-free Snorkel pipeline on the selected programs,
                see per-program weight / coverage / conflict.
  5. Results  → preview labeled rows and download labeled JSONL.
"""

from __future__ import annotations

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
PAJAMA_ROOT = Path(os.path.expanduser("~/pajama"))
MOCK_JUDGES_DIR = PAJAMA_ROOT / "judge_programs" / "judge_programs_judgelm"
MOCK_MANIFEST = (
    PAJAMA_ROOT
    / "judge_programs"
    / "judge_programs_scripts"
    / "program_manifest_judgelm.json"
)
MOCK_SAMPLE = THIS_DIR / "examples" / "judgelm_sample_60.jsonl"
MOCK_CACHED_S1 = (
    PAJAMA_ROOT / "snorkel_label_model_pipeline" / "pipeline_outputs_judgelm" / "test_s1.npy"
)
MOCK_CACHED_S2 = (
    PAJAMA_ROOT / "snorkel_label_model_pipeline" / "pipeline_outputs_judgelm" / "test_s2.npy"
)


# ── Page setup ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAJAMA Labeling Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
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
    ss.setdefault("aggregation", None)  # AggregationResult
    ss.setdefault("prompt_template", G.DEFAULT_PROMPT_TEMPLATE)
    ss.setdefault("prompt_chat", [])  # list[{role, content}]
    ss.setdefault("program_chats", {})  # {program_id: list[{role, content}]}
    ss.setdefault("editing_program_id", None)
    ss.setdefault("abstain_band", 0.02)


init_state()


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
    rows = P.load_jsonl(str(MOCK_SAMPLE))
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
    st.session_state.rows_source = f"mock: judgelm sample ({len(rows)} rows)"
    st.session_state.programs = progs
    st.session_state.aggregation = None


def _selected_programs():
    if st.session_state.programs is None:
        return []
    return [p for p in st.session_state.programs if p.selected and p.fn is not None]


def _heuristic_label(p: P.Program) -> str:
    if p.heuristic_name:
        return f"H{p.heuristic_id}: {p.heuristic_name}"
    return "—"


def _format_program_card(p: P.Program) -> str:
    h = _heuristic_label(p)
    summary = p.approach_summary or G.summarize_approach(p.code)
    return f"**{p.display_name}** &nbsp;·&nbsp; {h} &nbsp;·&nbsp; *variant {p.variant or '?'}*  \n`{summary}`"


def _score_with_programs(rows, programs, progress_text="Scoring"):
    """Score (rows × programs). Uses cached arrays in mock mode when possible."""
    use_cache = (
        st.session_state.mode == "mock"
        and st.session_state.orig_indices is not None
        and all(i >= 0 for i in st.session_state.orig_indices)
        and MOCK_CACHED_S1.exists()
        and MOCK_CACHED_S2.exists()
        and len(programs) == 80
        and all(p.program_id == i + 1 for i, p in enumerate(programs))
        and not any(p.dirty for p in programs)
    )
    if use_cache:
        idx = np.asarray(st.session_state.orig_indices, dtype=int)
        s1_full = np.load(MOCK_CACHED_S1)
        s2_full = np.load(MOCK_CACHED_S2)
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


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 PAJAMA Labeling Studio")
    st.caption("Val-free weak supervision for pairwise preference data.")

    st.subheader("1 · Mode")
    mode = st.radio(
        "Mode",
        options=["mock", "live"],
        format_func=lambda x: "Mock (judgelm sample + cached scores)"
        if x == "mock"
        else "Live (call Claude + score uploaded data)",
        index=0 if st.session_state.mode == "mock" else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    if mode == "live":
        api_key_input = st.text_input(
            "ANTHROPIC_API_KEY",
            value=st.session_state.api_key,
            type="password",
            help="Used for program generation and per-program chat.",
        )
        st.session_state.api_key = api_key_input
        if not api_key_input:
            st.warning("Set the API key to enable generation / chat.")

    st.subheader("2 · Dataset")
    if mode == "mock":
        if st.button("Load judgelm sample (60 rows)", use_container_width=True):
            _bootstrap_mock()
            st.success("Mock dataset + 80 cached programs loaded.")
    else:
        uploaded = st.file_uploader(
            "Upload JSONL with `query`, `response1`, `response2` "
            "(optional `verdict` for few-shot)",
            type=["jsonl", "json"],
        )
        if uploaded is not None and st.button("Use uploaded data", use_container_width=True):
            text = uploaded.read().decode("utf-8")
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            st.session_state.rows = rows
            st.session_state.orig_indices = None
            st.session_state.rows_source = f"uploaded: {uploaded.name} ({len(rows)} rows)"
            st.session_state.aggregation = None
            st.success(f"Loaded {len(rows)} rows.")

    if st.session_state.rows is not None:
        st.caption(f"📁 {st.session_state.rows_source}")

    st.subheader("3 · Programs")
    if mode == "mock":
        if st.button("Reload mock programs", use_container_width=True):
            st.session_state.programs = _load_mock_programs()
            st.session_state.aggregation = None
    else:
        st.caption(
            "Click **Generate 80 programs** on the Programs tab once your dataset is loaded."
        )

    st.subheader("4 · Aggregation")
    st.session_state.abstain_band = st.slider(
        "Abstain band (|diff| ≤ band → abstain)",
        min_value=0.00,
        max_value=0.20,
        value=float(st.session_state.abstain_band),
        step=0.01,
        help="Width of the no-confidence band around 0 used to decide abstention.",
    )

    st.markdown("---")
    n_progs = len(st.session_state.programs) if st.session_state.programs else 0
    n_sel = len(_selected_programs())
    n_rows = len(st.session_state.rows) if st.session_state.rows else 0
    st.caption(f"**State:** {n_rows} rows · {n_sel}/{n_progs} programs selected")


# ── Main ────────────────────────────────────────────────────────────────
st.title("PAJAMA Labeling Studio")
st.caption(
    "Generate programmatic judges → aggregate with Snorkel → download labeled preference data."
)

tab_data, tab_programs, tab_prompt, tab_agg, tab_results = st.tabs(
    ["📥 Data", "🧮 Programs", "💬 Prompt", "📊 Aggregation", "📤 Results"]
)


# ── Tab: Data ───────────────────────────────────────────────────────────
with tab_data:
    if st.session_state.rows is None:
        st.info("Load a dataset from the sidebar to get started.")
    else:
        rows = st.session_state.rows
        st.markdown(f"### {len(rows)} pairwise comparisons")
        st.caption(st.session_state.rows_source)

        preview_df = pd.DataFrame(
            [
                {
                    "i": i,
                    "query": str(r.get("query", ""))[:140],
                    "response1": str(r.get("response1", ""))[:140],
                    "response2": str(r.get("response2", ""))[:140],
                    "verdict": r.get("verdict", "—"),
                }
                for i, r in enumerate(rows[:50])
            ]
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        with st.expander("Schema check"):
            keys = sorted({k for r in rows[:10] for k in r.keys()})
            required = {"query", "response1", "response2"}
            missing = required - set(keys)
            if missing:
                st.error(f"Missing required fields: {missing}")
            else:
                st.success("All required fields present.")
            st.write({"fields_seen": keys})


# ── Tab: Programs ───────────────────────────────────────────────────────
def render_programs_tab():
    progs = st.session_state.programs

    col1, col2, col3, col4 = st.columns([1.3, 1, 1, 1])
    with col1:
        gen_disabled = (
            st.session_state.mode != "live"
            or not st.session_state.api_key
            or st.session_state.rows is None
        )
        if st.button(
            "✨ Generate 80 programs (live)",
            disabled=gen_disabled,
            use_container_width=True,
            help="Calls Claude Opus 4.6 for all 80 slots. Requires API key + uploaded data.",
        ):
            _run_full_generation()

    with col2:
        if st.button("Select all", disabled=progs is None, use_container_width=True):
            for p in progs:
                p.selected = True
            st.session_state.aggregation = None

    with col3:
        if st.button("Deselect all", disabled=progs is None, use_container_width=True):
            for p in progs:
                p.selected = False
            st.session_state.aggregation = None

    with col4:
        if st.button(
            "Validate all (recompile)",
            disabled=progs is None,
            use_container_width=True,
        ):
            ok, fail = 0, 0
            for p in progs:
                try:
                    p.fn = P.compile_program(p.code)
                    p.status = "success"
                    ok += 1
                except Exception as e:
                    p.fn = None
                    p.status = f"compile_error: {e}"
                    fail += 1
            st.toast(f"Recompiled — {ok} ok, {fail} broken.")

    if progs is None:
        st.info("Load programs from the sidebar (mock mode) or generate them (live mode).")
        return

    # Per-heuristic grid layout
    by_heuristic: dict[int, list[P.Program]] = {}
    for p in progs:
        by_heuristic.setdefault(p.heuristic_id or 0, []).append(p)

    heuristic_filter = st.multiselect(
        "Filter by heuristic",
        options=sorted(by_heuristic),
        default=sorted(by_heuristic),
        format_func=lambda h: (
            f"H{h}: {G.HEURISTICS.get(h, {}).get('name', 'unknown')}"
            if h in G.HEURISTICS
            else f"H{h}"
        ),
    )

    for h_id in sorted(heuristic_filter):
        group = by_heuristic[h_id]
        h_meta = G.HEURISTICS.get(h_id, {})
        st.markdown(
            f"#### H{h_id} · {h_meta.get('name', 'unknown')}  "
            f"<span style='color:#888'>({len(group)} programs)</span>",
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        for idx, p in enumerate(group):
            with cols[idx % 4]:
                _render_program_card(p)
        st.markdown("---")

    # Per-program editor (rendered below the grid)
    if st.session_state.editing_program_id is not None:
        _render_program_editor(st.session_state.editing_program_id)


def _render_program_card(p: P.Program):
    summary = p.approach_summary or G.summarize_approach(p.code)
    status_emoji = "🟢" if p.fn is not None else "🔴"
    with st.container(border=True):
        top1, top2 = st.columns([3, 1])
        with top1:
            st.markdown(f"**{status_emoji} {p.display_name}** · *v{p.variant or '?'}*")
            st.caption(summary)
        with top2:
            new_sel = st.checkbox(
                "use",
                value=p.selected,
                key=f"sel_{p.program_id}",
                label_visibility="collapsed",
            )
            if new_sel != p.selected:
                p.selected = new_sel
                st.session_state.aggregation = None

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("View / Edit", key=f"edit_{p.program_id}", use_container_width=True):
                st.session_state.editing_program_id = p.program_id
        with b2:
            if st.button("💬 Chat", key=f"chat_{p.program_id}", use_container_width=True):
                st.session_state.editing_program_id = p.program_id
                st.session_state[f"_open_chat_{p.program_id}"] = True
        with b3:
            if st.button("🗑️", key=f"del_{p.program_id}", use_container_width=True):
                st.session_state.programs = [
                    x for x in st.session_state.programs if x.program_id != p.program_id
                ]
                st.session_state.aggregation = None
                st.toast(f"Removed {p.display_name}")
                st.rerun()


def _render_program_editor(program_id: int):
    progs = st.session_state.programs
    p = next((x for x in progs if x.program_id == program_id), None)
    if p is None:
        st.session_state.editing_program_id = None
        return

    st.markdown(f"## ✏️ Editing `{p.display_name}`")
    h_meta = G.HEURISTICS.get(p.heuristic_id or 0, {})
    st.caption(
        f"**Heuristic:** H{p.heuristic_id} · {h_meta.get('name', 'unknown')} · "
        f"variant {p.variant or '?'}"
    )
    st.caption(f"**Approach tags:** `{p.approach_summary or G.summarize_approach(p.code)}`")

    col_code, col_chat = st.columns([1.2, 1])

    with col_code:
        new_code = st.text_area(
            "Program source",
            value=p.code,
            height=500,
            key=f"code_{p.program_id}",
        )
        save, recompile, close = st.columns(3)
        with save:
            if st.button("💾 Save", key=f"save_{p.program_id}", use_container_width=True):
                if new_code != p.code:
                    p.dirty = True
                p.code = new_code
                try:
                    p.fn = P.compile_program(new_code)
                    p.status = "success"
                    p.approach_summary = G.summarize_approach(new_code)
                    st.session_state.aggregation = None
                    st.success("Saved.")
                except Exception as e:
                    p.fn = None
                    p.status = f"compile_error: {e}"
                    st.error(f"Compile error: {e}")
        with recompile:
            if st.button("🔄 Recompile", key=f"recompile_{p.program_id}", use_container_width=True):
                try:
                    p.fn = P.compile_program(p.code)
                    p.status = "success"
                    st.success("Compiled.")
                except Exception as e:
                    p.fn = None
                    p.status = f"compile_error: {e}"
                    st.error(f"Compile error: {e}")
        with close:
            if st.button("Close editor", key=f"close_{p.program_id}", use_container_width=True):
                st.session_state.editing_program_id = None
                st.rerun()

    with col_chat:
        st.markdown("##### 💬 Chat to refine this program")
        if st.session_state.mode != "live" or not st.session_state.api_key:
            st.info("Set ANTHROPIC_API_KEY and switch to Live mode to chat.")
        chat_key = f"chat_{p.program_id}"
        st.session_state.program_chats.setdefault(p.program_id, [])
        history = st.session_state.program_chats[p.program_id]

        with st.container(height=380, border=True):
            for turn in history:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

        user_msg = st.chat_input(
            "Ask Claude to refine this program (e.g. 'add bigram overlap')",
            key=f"chatin_{p.program_id}",
            disabled=(st.session_state.mode != "live" or not st.session_state.api_key),
        )
        if user_msg:
            history.append({"role": "user", "content": user_msg})
            try:
                with st.spinner("Thinking..."):
                    reply, new_code = G.chat_regenerate(
                        current_code=p.code,
                        user_message=user_msg,
                        history=history[:-1],
                        api_key=st.session_state.api_key,
                    )
                history.append({"role": "assistant", "content": reply})
                if new_code:
                    st.session_state[f"_pending_code_{p.program_id}"] = new_code
            except Exception as e:
                history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
            st.rerun()

        pending = st.session_state.get(f"_pending_code_{p.program_id}")
        if pending:
            st.markdown("**Claude proposed a new version:**")
            st.code(pending[:2000] + ("..." if len(pending) > 2000 else ""), language="python")
            ca, cb = st.columns(2)
            with ca:
                if st.button(
                    "✅ Accept proposed code",
                    key=f"accept_{p.program_id}",
                    use_container_width=True,
                ):
                    try:
                        new_fn = P.compile_program(pending)
                        p.code = pending
                        p.fn = new_fn
                        p.status = "success"
                        p.dirty = True
                        p.approach_summary = G.summarize_approach(pending)
                        st.session_state.pop(f"_pending_code_{p.program_id}", None)
                        st.session_state.aggregation = None
                        st.success("Accepted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Compile error in proposed code: {e}")
            with cb:
                if st.button("✖️ Discard", key=f"discard_{p.program_id}", use_container_width=True):
                    st.session_state.pop(f"_pending_code_{p.program_id}", None)
                    st.rerun()


def _run_full_generation():
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
    bar = st.progress(0.0, text="Generating programs…")
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
        bar.progress((k + 1) / len(plan), text=f"Generating programs… {k+1}/{len(plan)}")
    bar.empty()
    st.session_state.programs = progs
    st.session_state.aggregation = None
    st.success(f"Generated {len(progs)} programs.")


with tab_programs:
    render_programs_tab()


# ── Tab: Prompt ─────────────────────────────────────────────────────────
with tab_prompt:
    st.markdown("### Global generation prompt template")
    st.caption(
        "This template is used for all 80 generation calls. "
        "Required placeholders: `{heuristic_name}`, `{heuristic_description}`, "
        "`{avoid_block}`, `{variant}`, `{few_shot_text}`."
    )
    template = st.text_area(
        "Template",
        value=st.session_state.prompt_template,
        height=450,
        key="prompt_template_box",
    )
    save_p, reset_p = st.columns(2)
    with save_p:
        if st.button("💾 Save template", use_container_width=True):
            required = {
                "{heuristic_name}",
                "{heuristic_description}",
                "{avoid_block}",
                "{variant}",
                "{few_shot_text}",
            }
            missing = [p for p in required if p not in template]
            if missing:
                st.error(f"Template missing required placeholders: {missing}")
            else:
                st.session_state.prompt_template = template
                st.success("Saved.")
    with reset_p:
        if st.button("↺ Reset to default", use_container_width=True):
            st.session_state.prompt_template = G.DEFAULT_PROMPT_TEMPLATE
            st.rerun()

    st.markdown("### 💬 Chat with Claude to refine the prompt")
    if st.session_state.mode != "live" or not st.session_state.api_key:
        st.info("Live mode + API key required to use the prompt-refinement chat.")
    with st.container(height=300, border=True):
        for turn in st.session_state.prompt_chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
    user_q = st.chat_input(
        "Ask Claude to refine the prompt template",
        disabled=(st.session_state.mode != "live" or not st.session_state.api_key),
    )
    if user_q:
        st.session_state.prompt_chat.append({"role": "user", "content": user_q})
        try:
            with st.spinner("Thinking..."):
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
            st.session_state.prompt_chat.append(
                {"role": "assistant", "content": f"⚠️ Error: {e}"}
            )
        st.rerun()
    pending = st.session_state.get("_pending_prompt")
    if pending:
        st.markdown("**Claude proposed a new template:**")
        st.code(pending, language="text")
        ca, cb = st.columns(2)
        with ca:
            if st.button("✅ Use this template", use_container_width=True):
                st.session_state.prompt_template = pending
                st.session_state.pop("_pending_prompt", None)
                st.rerun()
        with cb:
            if st.button("✖️ Discard proposal", use_container_width=True):
                st.session_state.pop("_pending_prompt", None)
                st.rerun()


# ── Tab: Aggregation ────────────────────────────────────────────────────
with tab_agg:
    progs = _selected_programs()
    rows = st.session_state.rows

    st.markdown("### Val-free Snorkel aggregation")
    st.caption(
        "Per-program robust normalization → diff → vote with abstain band → "
        "fit LabelModel directly on all selected programs."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        can_run = bool(progs) and bool(rows)
        if st.button(
            "▶ Run aggregation",
            disabled=not can_run,
            use_container_width=True,
            type="primary",
        ):
            t0 = time.time()
            s1, s2 = _score_with_programs(rows, progs, progress_text="Scoring rows")
            with st.spinner("Fitting Snorkel LabelModel..."):
                result = P.aggregate(
                    s1=s1,
                    s2=s2,
                    program_ids=[p.program_id for p in progs],
                    abstain_band=float(st.session_state.abstain_band),
                )
            st.session_state.aggregation = result
            st.toast(f"Aggregated in {time.time()-t0:.1f}s")
    with c2:
        if st.button("Clear", disabled=st.session_state.aggregation is None, use_container_width=True):
            st.session_state.aggregation = None
            st.rerun()
    with c3:
        if not can_run:
            st.caption("Need at least one selected program and a loaded dataset.")

    result = st.session_state.aggregation
    if result is not None:
        # ── Summary KPIs
        n, m = result.M.shape
        n_cov = int((result.hard != -1).sum())
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", n)
        k2.metric("Programs (selected)", m)
        k3.metric("Coverage", f"{100*n_cov/n:.1f}%")
        k4.metric("Avg per-program coverage", f"{100*result.coverage.mean():.1f}%")

        # ── Per-program metrics dataframe ── selected programs only
        prog_lookup = {p.program_id: p for p in progs}
        df = pd.DataFrame(
            {
                "program": [f"judge_{pid}" for pid in result.program_ids],
                "heuristic": [
                    f"H{prog_lookup[pid].heuristic_id}: {prog_lookup[pid].heuristic_name}"
                    if prog_lookup[pid].heuristic_id
                    else "—"
                    for pid in result.program_ids
                ],
                "variant": [prog_lookup[pid].variant for pid in result.program_ids],
                "logic_tags": [
                    prog_lookup[pid].approach_summary
                    or G.summarize_approach(prog_lookup[pid].code)
                    for pid in result.program_ids
                ],
                "weight": result.weights.astype(float),
                "coverage": result.coverage.astype(float),
                "conflict": result.conflicts.astype(float),
            }
        )
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)

        st.markdown("### Per-program weights · coverage · conflict")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "weight": st.column_config.ProgressColumn(
                    "weight (learned accuracy)",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "coverage": st.column_config.ProgressColumn(
                    "coverage", format="%.2f", min_value=0.0, max_value=1.0
                ),
                "conflict": st.column_config.ProgressColumn(
                    "conflict", format="%.2f", min_value=0.0, max_value=1.0
                ),
            },
        )

        # ── Side-by-side bar charts
        st.markdown("### Bar view")
        col_w, col_c, col_x = st.columns(3)
        with col_w:
            st.caption("Snorkel learned weight (≈ per-program accuracy)")
            st.bar_chart(df.set_index("program")["weight"], height=420)
        with col_c:
            st.caption("Coverage (fraction non-abstain)")
            st.bar_chart(df.set_index("program")["coverage"], height=420)
        with col_x:
            st.caption("Conflict rate (LFAnalysis)")
            st.bar_chart(df.set_index("program")["conflict"], height=420)


# ── Tab: Results ────────────────────────────────────────────────────────
with tab_results:
    result = st.session_state.aggregation
    rows = st.session_state.rows
    if result is None or rows is None:
        st.info("Run aggregation first.")
    else:
        st.markdown("### Predicted labels")
        labeled = P.label_jsonl_export(rows, result)
        df = pd.DataFrame(
            [
                {
                    "i": i,
                    "query": str(r.get("query", ""))[:120],
                    "predicted_verdict": r["pajama_predicted_verdict"],
                    "P(response2 wins)": (
                        f"{r['pajama_response2_prob']:.3f}"
                        if r["pajama_response2_prob"] is not None
                        else "—"
                    ),
                    "abstained": r["pajama_abstained"],
                    "gold_verdict": r.get("verdict", "—"),
                }
                for i, r in enumerate(labeled)
            ]
        )

        # KPI: agreement with gold where gold is available
        gold = [r.get("verdict") for r in labeled]
        gold_known = [(i, g) for i, g in enumerate(gold) if g in (1, "1", 2, 2.0, "2")]
        if gold_known:
            covered = [
                (g, labeled[i]["pajama_predicted_verdict"])
                for i, g in gold_known
                if not labeled[i]["pajama_abstained"]
            ]
            if covered:
                agree = sum(int(int(g) == int(p)) for g, p in covered)
                st.metric(
                    "Agreement with gold (on covered rows)",
                    f"{100*agree/len(covered):.1f}%",
                    help=f"{agree}/{len(covered)} covered rows agree with gold verdict.",
                )

        st.dataframe(df, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        for r in labeled:
            buf.write(json.dumps(r) + "\n")
        st.download_button(
            "⬇️ Download labeled JSONL",
            data=buf.getvalue(),
            file_name="pajama_labeled.jsonl",
            mime="application/jsonl",
            use_container_width=True,
        )
