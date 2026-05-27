# Demo pipeline caches (judgelm)

Copied from `pajama_workflow/judgelm_outputs/` so the Streamlit demo can re-run the
**production** Snorkel pipeline (`run.py` logic) without re-scoring or calling HF.

| File | Description |
|------|-------------|
| `val_s1.npy`, `val_s2.npy` | Raw judge scores, 500 × 80 (validation split) |
| `val_gold.npy` | Gold labels (0 = response1 wins, 1 = response2 wins) |
| `test_s1.npy`, `test_s2.npy` | Raw judge scores, 5000 × 80 (test split; used for the 50-row preview) |
| `pipeline_summary.json` | Written by the demo when you click **Run pipeline** (optional; recreated each run) |

To refresh score caches after a new `run.py` execution:

```bash
cp ../pajama_workflow/judgelm_outputs/{val_s1,val_s2,test_s1,test_s2}.npy .
```

Regenerate `val_gold.npy` from Hugging Face if needed:

```python
from datasets import load_dataset
import numpy as np
ds = load_dataset("sprocket-lab/PAJAMA", "judgelm", split="validation")
v = np.array(ds["verdict"])
np.save("val_gold.npy", np.where(v == 1, 0, 1).astype(np.int64))
```
