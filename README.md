# JEPA-Lens
Mapping and steering model understanding via predictive representations.

JEPA-Lens is a from-scratch Image JEPA project on CIFAR-10.  
Instead of reconstructing pixels, it predicts masked patch representations and uses prediction error as an "understanding map" signal.

## What this project is trying to answer
Most model-interpretability tools ask: "Which pixels influenced a class decision?"

JEPA-Lens asks a different question:
- Which parts of an image are easy for the model to *predict from context*?
- Which parts remain hard or uncertain?

That distinction gives a map of what the model finds structurally simple vs semantically complex.

## Core idea
- Low prediction error: region is easy and predictable.
- High prediction error: region is complex or uncertain.
- No labels are required to produce these heatmaps.

## Why JEPA (not pixel reconstruction)
In JEPA, the model does not try to regenerate missing pixels.  
It predicts latent representations of hidden regions from visible context.

This pushes learning toward higher-level structure (shape, layout, object parts), rather than low-level texture matching.

## How it works

### The architecture

```text
Image (32x32)
    |
PatchEmbed - cuts image into 8x8 grid = 64 patches of 4x4 pixels each
    |
Mask sampling (baseline v2):
    - One context set: 50% of patches (32) -> online Encoder
    - Four target sets: 25% each (16 each) -> target Encoder (EMA copy)
    - Target sets are sampled independently (can overlap)
    |
Online Encoder runs once on context -> context representations
Target Encoder runs for each target set -> target representations
    |
Predictor -> from context representations + target positions,
             predicts each target set
    |
Loss = mean MSE across target sets:
       MSE(predicted target repr, target-encoder repr)
       <- representation space, NOT pixel space
```

For variant fine-tuning in `train_variants.py`, masking is single-split per step (except `high_mask`, which uses 25% context / 75% target).

## Why two encoders are used
- Online encoder: updated by backprop every step.
- Target encoder: updated slowly by EMA.

This avoids collapse and stabilizes self-supervised training.

## Baseline vs variants
- Baseline v2 (`train.py`): depth-6 encoder, multi-crop target sampling, 60 epochs.
- Noise-robust: online encoder sees noisy input, target path sees clean input.
- Structure-focused: edge-based input to emphasize shape boundaries.
- High-mask: hides 75% of patches to force stronger global reasoning.

Variants are fine-tuned from `baseline_v2` in `train_variants.py`.

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

## Repository layout
- `train.py`: train baseline v2 (depth 6, multi-crop masking, 60 epochs), saves `checkpoints/baseline_v2.pth`.
- `train_variants.py`: fine-tune additional variants from `baseline_v2`.
- `evaluate.py`: linear probe, t-SNE, and understanding-map outputs for a checkpoint.
- `understanding.py`: side-by-side per-patch vs sliding-window understanding maps.
- `visuals.py`: masking visualization (what encoder saw vs hidden patches).
- `signal_nature_test.py`: tests whether the understanding signal is semantic vs texture/edge driven.
- `consistency_test.py`: tests transformation consistency (flip and crop stability).
- `causal_test.py`: ablation-style causal test (high-error vs low-error vs random patch removal).
- `dashboard.py`: generates a self-contained HTML dashboard at `jepa_lens_dashboard.html`.

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
- `jepa_lens_dashboard.html`

## Typical workflow
1. `python train.py`
2. `python train_variants.py`
3. Evaluate each checkpoint with `evaluate.py`
4. Generate extra diagnostics with `understanding.py` and `visuals.py`
5. Run `signal_nature_test.py`, `consistency_test.py`, and `causal_test.py` for deeper validation
6. Build presentation artifact with `dashboard.py`

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
