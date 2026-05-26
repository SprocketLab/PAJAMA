# Generating Programmatic Judges

Synthesizes 80 judge programs (using 10 rubrics × 8 variants) per dataset. Each program is a Python function `judging_function(query, response) -> float` where higher scores indicate better quality.

## Quick start

```bash
# this is a reminder
export ANTHROPIC_API_KEY="sk-ant-..."
hf auth login   # or: export HF_TOKEN="hf_..."

python generate_programs.py --dataset judgelm
python generate_programs.py --dataset pandalm
python generate_programs.py --dataset multipref
python generate_programs.py --dataset prometheus
python generate_programs.py --dataset preference_700K

# if you want to feed 20 examples into the prompt
python generate_programs.py --dataset prometheus --num-few-shot 20
```

## Employed Rubrics

| # | Evaluation Topic |
|----|------|
| 1 | Relevance to the Query |
| 2 | Language Quality and Readability |
| 3 | Completeness and Coverage |
| 4 | Factual Accuracy Indicators |
| 5 | Logical Coherence and Argument Structure |
| 6 | Clarity and Conciseness |
| 7 | Reasoning Transparency and Step-wise Formulation |
| 8 | Epistemic Calibration and Uncertainty Communication |
| 9 | Structural Organization and Formatting |
| 10 | Evidence Density and Specificity |

Generated programs `judge_1`–`judge_8` is using evaluation rubric #1 (variants 1–8), `judge_9`–`judge_16` cover rubric #2, and so on.
