# JEPA-Lens 🔍
### Mapping and Steering Model Understanding via Predictive Representations

> *"A system that reveals what parts of an image a model truly understands — by measuring how well it can predict them without seeing them."*

---

## What is this?

JEPA-Lens is a from-scratch implementation of **Image JEPA (Joint Embedding Predictive Architecture)** — Yann LeCun's proposed alternative to generative models like LLMs and diffusion models.

The core idea: instead of predicting raw pixels, predict **abstract representations** of missing image regions. The model never reconstructs what a patch looks like — it only predicts what it *means*. This forces the encoder to learn semantically meaningful structure rather than pixel-level texture.

JEPA-Lens extends this by using the **prediction error as a signal of understanding**:
- Low error → model finds this region predictable (background, uniform texture)
- High error → model finds this region complex (object faces, boundaries, fine structure)

This produces an **Understanding Map** — a heatmap showing what the model considers structured vs trivial, without any labels or gradients.

---

## Why is this different from saliency maps?

| | Saliency Maps | JEPA-Lens Understanding Maps |
|---|---|---|
| Signal source | Gradients w.r.t. a label | Prediction error in representation space |
| Requires labels | Yes | No |
| Requires gradients | Yes | No |
| What it measures | What affects the output | What the model finds unpredictable |

---

## How it works

### The architecture

```
Image (32×32)
    ↓
PatchEmbed — cuts image into 8×8 grid = 64 patches of 4×4 pixels each
    ↓
Random mask — splits 64 patches into:
    • Context patches (50%) → fed to online Encoder
    • Target patches  (50%) → fed to target Encoder (EMA copy)
    ↓
Online Encoder → context representations
Target Encoder → target representations (ground truth)
    ↓
Predictor → given context representations + target positions,
            predict what the target representations should be
    ↓
Loss = MSE(predicted representations, true representations)
       ← in representation space, NOT pixel space
```

### The two-encoder trick (anti-collapse)

If one encoder produced both context and target representations, it would collapse — mapping everything to the same vector makes prediction trivially easy. Instead:

- **Online encoder** — trained via backprop, updates every step
- **Target encoder** — updated only via EMA (slowly drifts toward online encoder)

The target encoder is always slightly ahead of the online encoder, forcing genuine learning rather than shortcut solutions.

### Multi-crop masking (v2)

Instead of one target region per image per step, v2 samples **4 target regions** from the same context. The encoder gets 4× the gradient signal without seeing 4× more data — forcing representations useful for predicting any region, not just one lucky mask.

---

## Setup

```bash
# Clone
git clone https://github.com/mukundhr/JEPA-Lens.git
cd JEPA-Lens

# Install dependencies (CUDA version for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Training

```bash
# Baseline (30 epochs, depth 4) — ~30-45 mins on RTX 3060
python train.py

# Improved (60 epochs, depth 6, multi-crop) — ~60-75 mins on RTX 3060
python train_v2.py
```

CIFAR-10 (~170MB) downloads automatically on first run.

After training, `evaluate.py` runs automatically and produces:
- `outputs/loss_curve.png` — training curve
- `outputs/tsne_baseline.png` — t-SNE of encoder representations
- `outputs/understanding_map_baseline.png` — per-patch and sliding window heatmaps

---

## Evaluation

```bash
# Run manually on any checkpoint
python evaluate.py --checkpoint checkpoints/baseline.pth
python evaluate.py --checkpoint checkpoints/baseline_v2.pth
```

**What to look for:**
- Loss curve should fall steadily without plateauing early
- t-SNE should show loose class clustering (not a uniform blob)
- Linear probe accuracy >> 10% (random chance) confirms semantic structure

---

## Understanding Maps

```bash
# Per-patch and sliding window comparison
python understanding_map.py --checkpoint checkpoints/baseline.pth

# Masking visualization — what encoder saw vs what was hidden
python visualize_masking.py --checkpoint checkpoints/baseline.pth
```

---

## Results (baseline)

| Metric | Value |
|---|---|
| Linear probe accuracy (2k train) | 41.5% |
| Linear probe accuracy (50k train) | 52.1% |
| Random baseline | 10.0% |
| Training time (RTX 3060) | ~35 mins |
| Parameters | 2M |
| Dataset | CIFAR-10 (50k images) |

Understanding maps show consistent semantic structure — object regions (faces, bodies, boundaries) show higher prediction error than background regions, emerging purely from the self-supervised objective with no labels.

---

## Key insight

The encoder doesn't know what's important upfront. It discovers importance through the training objective — only features that help predict missing patches survive. Sky and background never help predict a dog's face. The dog's posture and shape always do. This distinction emerges from physics and semantics, not from human annotation.

> *Low prediction error = the model finds this region trivially predictable.*
> *High prediction error = the model finds this region genuinely complex.*
> *The gap between the two is where understanding lives.*

---

## License
MIT License