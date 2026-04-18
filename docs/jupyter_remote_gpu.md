# Remote JupyterLab GPU Workflow

This repository now supports a notebook-friendly GNN workflow for the
DocMatcher line-matching stage.

Important:
- `docmatcher@inv3d` without extra flags uses the released baseline LightGlue checkpoint.
- The GNN matcher is an opt-in path.
- For real GNN results, fine-tune the matcher first and then pass that custom checkpoint back into full inference.

## 1. Environment Setup

Run these cells in JupyterLab.

```python
import sys

!{sys.executable} -m pip install --upgrade pip
```

If PyTorch is not already installed in your remote environment, install a wheel
matching your CUDA runtime. Example for CUDA 11.8:

```python
import sys

!{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install the project dependencies:

```python
import sys

!{sys.executable} -m pip install -r requirements.txt
```

## 2. Baseline Sanity Check

This verifies the standard released DocMatcher pipeline:

```python
import sys

!{sys.executable} inference.py --model docmatcher@inv3d --dataset example --gpu 0
```

## 3. Fine-Tune the GNN Matcher

This step initializes the GNN matcher from the released DocMatcher LightGlue
checkpoint and fine-tunes the graph-specific layers together with the existing
matcher weights.

```python
import sys

!{sys.executable} train.py \
    --model-part lightglue \
    --gpu 0 \
    --enable-gnn \
    --graph-k-neighbors 5 \
    --experiment-name inv3d_former2_glue1_gnn
```

Optional:
- Add `--graph-sparse-attention` to enforce neighbor-only self-attention.
- Change `--graph-k-neighbors` to try `3`, `5`, or `8`.
- Use `--lightglue-learning-rate 5e-5` if you want a smaller fine-tuning step.
- Use `--lightglue-init-checkpoint /path/to/checkpoint.ckpt` to start from a custom matcher checkpoint.

## 4. Locate the Best Checkpoint

```python
from pathlib import Path

checkpoint_dir = Path("models/training/lightglue/inv3d_former2_glue1_gnn")
checkpoints = sorted(checkpoint_dir.rglob("*.ckpt"))
checkpoints[-5:]
```

Typical useful checkpoints:
- `last.ckpt`
- the best validation checkpoint saved by Lightning in the same run directory

## 5. Run Full DocMatcher with the GNN Matcher

Replace `CKPT` with the checkpoint you want to use.

```python
import sys
from pathlib import Path

CKPT = sorted(Path("models/training/lightglue/inv3d_former2_glue1_gnn").rglob("last.ckpt"))[-1]

!{sys.executable} inference.py \
    --model docmatcher@inv3d \
    --dataset example \
    --gpu 0 \
    --enable-gnn-line-matcher \
    --graph-k-neighbors 5 \
    --lightglue-checkpoint {CKPT}
```

For real evaluation:

```python
import sys
from pathlib import Path

CKPT = sorted(Path("models/training/lightglue/inv3d_former2_glue1_gnn").rglob("last.ckpt"))[-1]

!{sys.executable} inference.py \
    --model docmatcher@inv3d \
    --dataset inv3d_real \
    --gpu 0 \
    --enable-gnn-line-matcher \
    --graph-k-neighbors 5 \
    --lightglue-checkpoint {CKPT}
```

## 6. Evaluate the Output

```python
import sys

!{sys.executable} eval.py --run example-docmatcher@inv3d
```

Or for the real dataset:

```python
import sys

!{sys.executable} eval.py --run inv3d_real-docmatcher@inv3d
```

## 7. Notes

- If you run `--enable-gnn-line-matcher` without a fine-tuned GNN checkpoint, the
  code will warn you. That run is only useful as an architecture sanity check.
- The graph bias now affects only graph-connected pairs in dense mode, so
  `--graph-k-neighbors` matters even without sparse attention.
- The graph gate starts at `0.0` for compatibility with the released baseline
  checkpoint and becomes active during fine-tuning.
