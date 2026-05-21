# PAJAMA Labeling Studio (Streamlit demo)

A self-contained Streamlit UI for the **val-free** PAJAMA pipeline:

```
JSONL (query, response1, response2)
        │
        ▼
80 program-judges  ── Claude Opus 4.6
        │
        ▼
per-program normalize → vote (with abstain band)
        │
        ▼
Snorkel LabelModel  ── learns per-program weights
        │
        ▼
labeled JSONL (predicted_verdict, response2_prob)
```

No validation set required. No threshold tuning, no top-k selection — all 80 (or fewer, if the user deselects some) programs feed directly into Snorkel.

## What you can do in the UI

- **Data** — preview the loaded pairwise dataset; schema-check.
- **Programs** — see all 80 generated programs grouped by heuristic, with each program's "logic tags" (e.g. `Flesch readability, syllable counting`). You can:
  - select / deselect any program (excludes it from aggregation)
  - view & edit code in-place
  - delete a program entirely
  - open a per-program **chat** with Claude to iteratively rewrite that one program
- **Prompt** — edit the global prompt template used for all 80 generations, or chat with Claude to refine it. Required placeholders are enforced.
- **Aggregation** — run the val-free Snorkel pipeline on the selected programs. See:
  - Snorkel's learned per-program weight (≈ accuracy)
  - coverage (fraction of rows where the program didn't abstain)
  - conflict rate (from `LFAnalysis`)
  - bar charts side-by-side
- **Results** — preview predicted verdicts and download the labeled JSONL. If the input has gold `verdict`, also shows agreement.

## Two modes

| Mode | Data | Programs | Aggregation |
|------|------|----------|-------------|
| **Mock** (default) | Bundled 60-row sample from `sprocket-lab/PAJAMA · judgelm/test` | Existing 80 from `~/pajama/judge_programs/judge_programs_judgelm/` | Uses cached `test_s1.npy` / `test_s2.npy` from the snorkel pipeline so the demo runs in **<1s** |
| **Live** | User uploads any JSONL with `query`, `response1`, `response2` (optional `verdict`) | Generated fresh by Claude Opus 4.6 (10 heuristics × 8 variants) | Scores every row with every selected program, then fits Snorkel |

Per-program chat and prompt-template chat call Claude in both modes when an `ANTHROPIC_API_KEY` is provided.

## Setup

```bash
cd ~/pajama/labeling_ui_demo
pip install -r requirements.txt

# Optional for live mode and chat:
export ANTHROPIC_API_KEY="sk-ant-..."

streamlit run app.py
```

## Files

```
labeling_ui_demo/
├── app.py          # Streamlit UI (sidebar + 5 tabs)
├── pipeline.py     # Val-free Snorkel aggregator + program loader
├── generation.py   # Claude program generation + chat helpers
├── examples/
│   └── judgelm_sample_60.jsonl   # bundled sample for mock mode
├── requirements.txt
└── README.md
```

## Relationship to the rest of `~/pajama`

- Mock mode reads from `~/pajama/judge_programs/judge_programs_judgelm/` and reuses the cached score arrays in `~/pajama/snorkel_label_model_pipeline/pipeline_outputs_judgelm/` — nothing else.
- Live mode uses the same prompt scaffolding as `judge_programs/judge_programs_scripts/generate_judging_programs.py` but is decoupled (the demo doesn't write into your existing program directories).
