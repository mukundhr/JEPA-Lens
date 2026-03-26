# JEPA-Lens
Probing predictive representations to understand what models find predictable.

JEPA-Lens is a from-scratch Image JEPA project on CIFAR-10.  
Instead of reconstructing pixels, it predicts masked patch representations and uses prediction error as a signal to study what the model finds easy or difficult to predict.

---

## What this project asks

JEPA-Lens asks:

- Which regions are predictable from context?
- Which regions remain difficult to predict?

This reframes understanding as predictability.

---

## Core idea

- Low prediction error → region is easy to infer from context  
- High prediction error → region is difficult or uncertain  

These per-patch errors form an "understanding map" without using labels.

---

## What this project finds

Prediction error is not a simple signal.

Through causal and controlled tests, we find:

- It is not random noise  
- It is not an edge detector  
- It is not purely semantic  

Instead, it is a mixed signal:

- A stable structural component that persists under geometric transforms  
- A fine-grained component sensitive to texture and local detail  

This suggests that predictability captures structure correlated with semantics, but remains entangled with low-level information.

---

## Why JEPA

JEPA predicts representations instead of pixels.

This encourages learning:

- shape and layout  
- object parts  
- contextual relationships  

rather than texture reconstruction.

---

## How it works

### Architecture

```text
Image (32x32)
    |
PatchEmbed → 8x8 grid (64 patches of 4x4 pixels)
    |
Mask sampling:
    - 50% context → online encoder
    - 4 target sets (25% each) → target encoder (EMA)
    |
Online encoder → context representations
Target encoder → target representations
    |
Predictor → predicts target representations from context
    |
Loss = MSE in representation space

```

## Why two encoders are used
- Online encoder: updated by backprop every step.
- Target encoder: updated slowly by EMA.

This avoids collapse and stabilizes self-supervised training.

## variants
- Baseline v2: depth 6, multi-crop masking
- Noise-robust: noisy input for online encoder
- Structure-focused: edge-based input
- High-mask: 75% masked

Variants are fine-tuned from the baseline.

## How to read the outputs
- Linear probe accuracy:
  - Quick sanity check for semantic quality of learned features.
  - Random chance is 10% on CIFAR-10.
- t-SNE:
  - If embeddings are one blob, representations are weak.
  - If class clusters emerge, representations carry structure.
- Understanding maps:
  - Green/cool regions: predictable from context.
  - Red/warm regions: uncertain or complex.
  - Useful signal is often in *differences between variants*.

## Setup
```bash
git clone https://github.com/mukundhr/JEPA-Lens.git
cd JEPA-Lens

pip install torch torchvision
pip install -r requirements.txt
```

Optional GPU check:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick start
Train baseline v2:
```bash
python train.py
```

This saves:
- `checkpoints/baseline_v2.pth`
- `outputs/loss_curve_v2.png`

and auto-runs:
- `evaluate.py --checkpoint checkpoints/baseline_v2.pth`

CIFAR-10 downloads automatically on first run.

## Evaluate checkpoints
```bash
python evaluate.py --checkpoint checkpoints/baseline_v2.pth
python evaluate.py --checkpoint checkpoints/noise_robust.pth
python evaluate.py --checkpoint checkpoints/structure_focused.pth
python evaluate.py --checkpoint checkpoints/high_mask.pth
```

Typical outputs:
- `outputs/tsne_<variant>.png`
- `outputs/understanding_map_<variant>.png`

## Train variants
`train_variants.py` expects `checkpoints/baseline_v2.pth` to exist and fine-tunes variant checkpoints.

```bash
python train_variants.py
```

Variant checkpoints:
- `checkpoints/noise_robust.pth`
- `checkpoints/structure_focused.pth`
- `checkpoints/high_mask.pth`

## Additional visualizations
```bash
python understanding.py --checkpoint checkpoints/baseline_v2.pth
python visuals.py --checkpoint checkpoints/baseline_v2.pth
```

`understanding.py` compares:
- Per-patch masking (fast, blocky)
- Sliding-window masking (smoother, slower)

`visuals.py` shows:
- Original image
- What context the encoder actually saw
- Error over hidden patches

## Validation tests
These scripts move beyond visual inspection and test whether error maps are meaningful.

### 1) Signal nature test
Checks whether prediction-error maps are stable under blur, weakly tied to raw edges, and distinct from an edge-focused model.

```bash
python signal_nature_test.py --baseline checkpoints/baseline_v2.pth --structure checkpoints/structure_focused.pth
```

Main outputs:
- `outputs/signal_nature_test.png`
- `outputs/signal_nature_qualitative.png`

### 2) Consistency test
Checks if understanding maps remain consistent under input transformations (horizontal flip and random crop).

```bash
python consistency_test.py
```

Main output:
- `outputs/transform_consistency.png`

### 3) Causal test
Tests the core claim directly: if high-error patches are semantically important, ablating them should hurt linear-probe accuracy more than ablating low-error or random patches.

```bash
python causal_test.py --checkpoint checkpoints/baseline_v2.pth
python causal_test.py --checkpoint checkpoints/baseline_v2.pth --k 8 16 24
```

Main output:
- `outputs/causal_test_<variant>.png`

## Dashboard
Generate a single-file HTML report (no server required):
```bash
python dashboard.py
```

Output:
- `dashboard.html`

The dashboard includes:
- Overview and pipeline summary
- Interactive understanding-map viewer
- t-SNE and masking visualizations
- Validation experiments
- Training curves and artifact-status cards

## Typical workflow
1. Train baseline
2. Train variants
3. Evaluate checkpoints
4. Visualize maps
5. Run validation experiments
6. Generate dashboard

## What success looks like
- Training loss trends down without instability.
- Linear probe is far above 10% random chance.
- t-SNE shows meaningful grouping.
- Understanding maps consistently highlight object-relevant regions more than flat background.
- Variant maps shift in plausible ways given the training change.

## Scope and limitations
- This is a compact CIFAR-10 research prototype, not a production I-JEPA implementation.
- Understanding maps are useful diagnostics, not guaranteed causal explanations.
- Spatial resolution is patch-limited (4x4 patch units on 32x32 images).

## License
MIT
