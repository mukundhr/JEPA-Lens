"""
visualize_masking.py — Show what JEPA sees vs what it has to predict
=====================================================================
For each image, shows 3 things side by side:
  1. Original image
  2. What the encoder SAW (context patches visible, hidden patches grayed out)
  3. Prediction error map (how hard was each hidden patch to predict?)

This answers: "what was hidden, and how well did JEPA handle it?"

Run:
  python visualize_masking.py --checkpoint checkpoints/baseline.pth
"""

import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import Encoder, Predictor, NUM_PATCHES, EMBED_DIM, sample_masks

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/baseline.pth")
parser.add_argument("--n_images",   default=6, type=int)
parser.add_argument("--seed",       default=42, type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CIFAR_CLASSES = ['plane','car','bird','cat','deer',
                 'dog','frog','horse','ship','truck']
GRID       = 8     # 8x8 patch grid
PATCH_SIZE = 4     # 4x4 pixels per patch
os.makedirs("outputs", exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
variant = ckpt.get("config", {}).get("variant", "baseline")

encoder   = Encoder(depth=4).to(DEVICE)
predictor = Predictor(depth=2).to(DEVICE)
encoder.load_state_dict(ckpt["encoder"])
predictor.load_state_dict(ckpt["predictor"])
encoder.eval()
predictor.eval()

# ── DATA ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
loader    = DataLoader(test_data, batch_size=args.n_images, shuffle=True, num_workers=0)
imgs, labels = next(iter(loader))
imgs = imgs.to(DEVICE)

# Denormalize for display
mean = torch.tensor([0.4914, 0.4822, 0.4465])
std  = torch.tensor([0.2023, 0.1994, 0.2010])
imgs_display = (imgs.cpu() * std.view(3,1,1) + mean.view(3,1,1)).clamp(0,1)


# ── HELPER: apply mask to image for visualization ─────────────────────────────
def apply_mask_to_image(img_np, hidden_indices, patch_size=4, grid=8):
    """
    img_np        : (H, W, 3) numpy array, values in [0,1]
    hidden_indices: list of patch indices that were HIDDEN from encoder
    Returns a copy with hidden patches grayed out and outlined in red
    """
    masked = img_np.copy()
    for idx in hidden_indices:
        row = idx // grid
        col = idx  % grid
        r0, r1 = row * patch_size, (row+1) * patch_size
        c0, c1 = col * patch_size, (col+1) * patch_size
        masked[r0:r1, c0:c1] = 0.5   # gray out
    return masked


def draw_patch_borders(ax, indices, color, patch_size=4, grid=8, lw=1.5):
    """Draw colored borders around specified patches on an axes."""
    for idx in indices:
        row = idx // grid
        col = idx  % grid
        r0 = row * patch_size
        c0 = col * patch_size
        rect = mpatches.Rectangle(
            (c0 - 0.5, r0 - 0.5), patch_size, patch_size,
            linewidth=lw, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)


# ── GENERATE ONE FIXED MASK (same for all images so comparison is fair) ───────
ctx_idx, tgt_idx = sample_masks(context_ratio=0.5, mask_ratio=0.5)


# ── COMPUTE PREDICTION ERROR PER HIDDEN PATCH ─────────────────────────────────
def patch_errors(img_batch, ctx_idx, tgt_idx):
    """Returns (B, n_tgt) error per hidden patch."""
    with torch.no_grad():
        ctx  = encoder(img_batch, patch_indices=ctx_idx)
        tgt  = encoder(img_batch, patch_indices=tgt_idx)
        tgt  = F.layer_norm(tgt, [EMBED_DIM])
        pred = predictor(ctx, ctx_idx, tgt_idx)
        err  = ((pred - tgt) ** 2).mean(dim=-1)   # (B, n_tgt)
    return err.cpu().numpy()

errors = patch_errors(imgs, ctx_idx, tgt_idx)   # (B, n_tgt)


# ── PLOT ──────────────────────────────────────────────────────────────────────
N   = args.n_images
fig, axes = plt.subplots(3, N, figsize=(N * 2.8, 8))
fig.patch.set_facecolor('#1a1a2e')

col_titles = [CIFAR_CLASSES[labels[i]] for i in range(N)]
row_labels  = ["Original", "What encoder saw\n(gray = hidden)", "Prediction error\non hidden patches"]

for i in range(N):
    img_np = imgs_display[i].permute(1, 2, 0).numpy()

    # ── ROW 0: Original ──────────────────────────────────────────────────────
    ax = axes[0, i]
    ax.imshow(img_np, interpolation='nearest')
    ax.set_title(col_titles[i], color='white', fontsize=10, pad=4)
    ax.axis('off')
    # Draw context patches in blue, hidden in red
    draw_patch_borders(ax, ctx_idx, color='#4fc3f7', lw=1.0)
    draw_patch_borders(ax, tgt_idx, color='#ef5350', lw=1.0)

    # ── ROW 1: Masked image ───────────────────────────────────────────────────
    ax = axes[1, i]
    masked_np = apply_mask_to_image(img_np, tgt_idx)
    ax.imshow(masked_np, interpolation='nearest')
    ax.axis('off')
    draw_patch_borders(ax, tgt_idx, color='#ef5350', lw=1.5)

    # ── ROW 2: Error map on hidden patches only ───────────────────────────────
    ax = axes[2, i]
    # Start with a dark background
    error_canvas = np.zeros((32, 32))

    # Fill in errors only for hidden (target) patches
    err_vals = errors[i]   # (n_tgt,)
    # Normalize across all hidden patches of this image
    e_min, e_max = err_vals.min(), err_vals.max() + 1e-8

    for j, patch_idx in enumerate(tgt_idx):
        row = patch_idx // GRID
        col = patch_idx  % GRID
        r0, r1 = row * PATCH_SIZE, (row+1) * PATCH_SIZE
        c0, c1 = col * PATCH_SIZE, (col+1) * PATCH_SIZE
        normalized_err = (err_vals[j] - e_min) / (e_max - e_min)
        error_canvas[r0:r1, c0:c1] = normalized_err

    # Show original image faintly as background
    ax.imshow(img_np, alpha=0.25, interpolation='nearest')
    # Overlay error on hidden patches
    im = ax.imshow(np.ma.masked_equal(error_canvas, 0),
                   cmap='RdYlGn_r', vmin=0, vmax=1,
                   alpha=0.85, interpolation='nearest')
    ax.axis('off')

    # Add colorbar on last column only
    if i == N - 1:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=7)
        cbar.set_label('prediction\nerror', color='white', fontsize=7)


# ── ROW LABELS ────────────────────────────────────────────────────────────────
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].set_ylabel(label, color='white', fontsize=9,
                                 rotation=0, labelpad=70, va='center')

# ── LEGEND ────────────────────────────────────────────────────────────────────
blue_patch = mpatches.Patch(color='#4fc3f7', label='Context (encoder saw this)')
red_patch  = mpatches.Patch(color='#ef5350', label='Target (hidden from encoder)')
fig.legend(handles=[blue_patch, red_patch],
           loc='lower center', ncol=2,
           fontsize=9, framealpha=0.3,
           labelcolor='white',
           facecolor='#2d2d44',
           edgecolor='white',
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f"JEPA-Lens: Masking Visualization — {variant.upper()}\n"
    "Row 1: original with patch grid  |  "
    "Row 2: what encoder saw  |  "
    "Row 3: prediction error on hidden patches",
    color='white', fontsize=10, y=0.98
)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])

out_path = f"outputs/masking_viz_{variant}.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✓ Saved → {out_path}")
print("\nWhat you're seeing:")
print("  Blue borders  = context patches (encoder saw these)")
print("  Red borders   = target patches  (hidden from encoder)")
print("  Row 3 colors  = red means hard to predict, green means easy")
print("  Faint image   = original shown as reference behind error map")