# PAJAMA Studio (A Streamlit Demo)

This is a self-contained Streamlit app using programmatic judges to evaluate pairwise preference.

**Workflow:** Sythesis Prompt → Synthesized Programs → Aggregation

| Mode | Data | Judges | Aggregation |
|------|------|--------|-------------|
| **Demo** (an example) | 60-row sample from bundled JSONL | 80 pre-built judges under `synthesized_programmatic_judges/judgelm/` | Production pipeline on cached val scores (`demo/outputs/*.npy`) |
| **Live** (can be your dataset) | Upload JSONL (`query`, `response1`, `response2`) | Generate 80 via Claude | Using Snorkel's Label Model on selected judges |

## Quick Start

```bash
# Optional for live mode / chat:
export ANTHROPIC_API_KEY="sk-ant-..."

streamlit run app.py
```

## Layout

```
demo/
├── app.py                  # Streamlit UI
├── pipeline.py             # Programs, live aggregation, mock production pipeline
├── generation.py           # Claude generation + chat helpers
├── assets/pajamas_icon.png
├── examples/judgelm_sample_60.jsonl
├── outputs/                # Cached .npy scores (see outputs/README.md)
├── .streamlit/config.toml
└── requirements.txt
```

Mock mode reads judges from `../synthesized_programmatic_judges/judgelm/` at the repo root.
