# LLM Judge Evaluation

Runs a local model as a pairwise preference judge via a vLLM server. Supports 5 datasets across val/test splits.

## Quick Start

**Step 1 — Start the vLLM server** (e.g. in a tmux session)

```bash
bash vllm_server.sh --model Qwen/Qwen2.5-7B-Instruct --served-model-name judge-model
```

**Step 2 — Run the judge** (separate terminal)

```bash
python llm_judge_server.py --model judge-model
```

`--model` must match `--served-model-name` used when starting the server.

Results are saved to `./<model-tag>/` by default.

## Output

Each dataset/split produces two files in the output directory:

- `<dataset>_<split>_<tag>_outputs.json` — per-sample predictions and raw model output
- `<dataset>_<split>_<tag>_summary.json` — metrics: accuracy, precision, recall, F1, coverage
- `all_summaries_<tag>.json` — combined summary across all datasets and splits

Metrics are computed only on parseable outputs (`A` or `B`). `strict_accuracy` counts parse failures as wrong answers.