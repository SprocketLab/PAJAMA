# Reward Model Distillation

Fine-tune Qwen2.5 as a pairwise reward model on PAJAMA preference data.

## Quick Start

Run `converting_data.py` first to prepare the data (see [Data](#data) section below), then train:

```bash
# This is distilling from programmatic judges
python reward_model_training.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_set_path rm_data_judgelm \
    --output_path ./models/qwen_rm_judgelm

# This is distilling from proprietary models (e.g., GPT-4)
python reward_model_training.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_set_path rm_data_judgelm_gpt4 \
    --output_path ./models/qwen_rm_judgelm_gpt4
```

## Dataset

Run `converting_data.py` to produce the training directories:

| Mode | Label source | Output dir |
| --- | --- | --- |
| `snorkel` | Program-based predictions | `rm_data_judgelm/`, `rm_data_prometheus/` |
| `gt` | GPT-4 ground-truth scores | `rm_data_judgelm_gpt4/`, `rm_data_prometheus_gpt4/` |

Each directory contains `train.jsonl`, `val.jsonl`, `test.jsonl` in `chosen`/`rejected` chat format.