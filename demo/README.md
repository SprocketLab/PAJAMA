# PAJAMA Workflow (Streamlit demo)

A self-contained Streamlit app for programmatic pairwise preference labeling.

**Workflow:** Start → Prompt → Programs → Results

| Mode | Data | Judges | Aggregation |
|------|------|--------|-------------|
| **Demo** (default) | 50-row sample from bundled JSONL | 80 pre-built judges under `synthesized_programmatic_judges/judgelm/` | Production pipeline on cached val scores (`demo/outputs/*.npy`) |
| **Live** | Upload JSONL (`query`, `response1`, `response2`) | Generate 80 via Claude | Val-free Snorkel on selected judges |

## Setup

```bash
cd demo
pip install -r requirements.txt

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
