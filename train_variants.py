"""
train_variants.py — Train all 3 JEPA-Lens variants sequentially
================================================================
Starts from baseline_v2.pth weights and fine-tunes each variant for 20 epochs.
Total time: ~75 mins on RTX 3060.

VARIANT 1: Noise-robust
  Encoder input = image + Gaussian noise
  Target encoder input = clean image
  Hypothesis: error maps become more uniform, model forced to reason globally
  not just locally from nearby pixels

VARIANT 2: Structure-focused
  Encoder input = Canny edge-detected image (3-channel grayscale edges)
  Target encoder input = same edge image
  Hypothesis: error shifts toward shape boundaries, away from texture/color

VARIANT 3: High-mask
  Mask ratio: 0.5 → 0.75 (model sees only 25% of patches, predicts 75%)
  Hypothesis: model learns more global, holistic representations

Run:
  python train_variants.py
"""

import os, copy, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2

from models import Encoder, Predictor, ema_update, sample_masks, EMBED_DIM, NUM_PATCHES


# ── SHARED CONFIG ─────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 256
FINETUNE_EPOCHS= 20
LR             = 5e-5       # lower LR for fine-tuning
EMA_DECAY      = 0.996
CONTEXT_RATIO  = 0.5
BASE_CHECKPOINT= "checkpoints/baseline_v2.pth"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs",     exist_ok=True)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


# ── CANNY EDGE TRANSFORM ──────────────────────────────────────────────────────
class CannyEdgeTransform:
    """
    Converts a normalized tensor image to Canny edge map (3-channel).
    Applied to encoder input only in structure-focused variant.
    """
    def __call__(self, tensor):
        # Denormalize → numpy uint8
        mean = torch.tensor(CIFAR_MEAN).view(3,1,1)
        std  = torch.tensor(CIFAR_STD).view(3,1,1)
        img  = (tensor * std + mean).clamp(0, 1)
        img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)

        # Canny on grayscale
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges  = cv2.Canny(gray, threshold1=50, threshold2=150)  # (H, W) uint8

        # Convert back to 3-channel normalized tensor
        edges_3ch = np.stack([edges, edges, edges], axis=2).astype(np.float32) / 255.0
        edge_tensor = torch.tensor(edges_3ch).permute(2,0,1)  # (3, H, W)

        # Normalize same as CIFAR
        edge_tensor = transforms.functional.normalize(edge_tensor,
                                                       mean=list(CIFAR_MEAN),
                                                       std=list(CIFAR_STD))
        return edge_tensor


def load_base_model():
    """Load baseline_v2 encoder + predictor as starting point."""
    ckpt      = torch.load(BASE_CHECKPOINT, map_location=DEVICE, weights_only=False)
    encoder   = Encoder(depth=6).to(DEVICE)
    predictor = Predictor(depth=3).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    target_encoder = copy.deepcopy(encoder).to(DEVICE)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    return encoder, target_encoder, predictor


def make_optimizer(encoder, predictor):
    return torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=0.05
    )


def save_loss_curve(loss_history, variant_name, epochs):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs+1), loss_history, "b-o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (representation space)")
    ax.set_title(f"JEPA-Lens {variant_name} — Fine-tuning Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"outputs/loss_curve_{variant_name}.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT 1: NOISE-ROBUST
# ─────────────────────────────────────────────────────────────────────────────
def train_noise_robust():
    print("\n" + "="*60)
    print("  VARIANT 1: Noise-Robust JEPA")
    print("  Encoder sees noisy input, target sees clean input")
    print("="*60 + "\n")

    NOISE_STD  = 0.15
    CHECKPOINT = "checkpoints/noise_robust.pth"
    MASK_RATIO = 0.5

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_data   = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)

    encoder, target_encoder, predictor = load_base_model()
    optimizer = make_optimizer(encoder, predictor)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    loss_history = []

    for epoch in range(FINETUNE_EPOCHS):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)

            # Noisy input for online encoder
            noise      = torch.randn_like(imgs) * NOISE_STD
            imgs_noisy = imgs + noise

            ctx_idx, tgt_idx = sample_masks(context_ratio=CONTEXT_RATIO,
                                            mask_ratio=MASK_RATIO)

            # Online encoder sees NOISY image
            ctx_repr = encoder(imgs_noisy, patch_indices=ctx_idx)

            # Target encoder sees CLEAN image
            with torch.no_grad():
                tgt_repr = target_encoder(imgs, patch_indices=tgt_idx)
                tgt_repr = F.layer_norm(tgt_repr, [EMBED_DIM])

            pred_repr = predictor(ctx_repr, ctx_idx, tgt_idx)
            loss      = F.mse_loss(pred_repr, tgt_repr)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            ema_update(encoder, target_encoder, EMA_DECAY)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()
        print(f"  Epoch [{epoch+1:02d}/{FINETUNE_EPOCHS}]  loss: {avg_loss:.4f}")

    torch.save({
        "encoder":        encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor":      predictor.state_dict(),
        "loss_history":   loss_history,
        "config": {
            "variant":    "noise_robust",
            "depth":      6,
            "pred_depth": 3,
            "noise_std":  NOISE_STD,
            "mask_ratio": MASK_RATIO,
        }
    }, CHECKPOINT)
    save_loss_curve(loss_history, "noise_robust", FINETUNE_EPOCHS)
    print(f"\n  ✓ Saved → {CHECKPOINT}")


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT 2: STRUCTURE-FOCUSED
# ─────────────────────────────────────────────────────────────────────────────
def train_structure_focused():
    print("\n" + "="*60)
    print("  VARIANT 2: Structure-Focused JEPA")
    print("  Encoder sees Canny edge-detected images")
    print("="*60 + "\n")

    CHECKPOINT = "checkpoints/structure_focused.pth"
    MASK_RATIO = 0.5

    # Base transform (no edge detection yet — applied per-batch)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    canny = CannyEdgeTransform()

    train_data   = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)

    encoder, target_encoder, predictor = load_base_model()
    optimizer = make_optimizer(encoder, predictor)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    loss_history = []

    for epoch in range(FINETUNE_EPOCHS):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)

            # Apply Canny edge detection to each image in batch
            imgs_edge = torch.stack([
                canny(imgs[i].cpu()).to(DEVICE) for i in range(imgs.shape[0])
            ])

            ctx_idx, tgt_idx = sample_masks(context_ratio=CONTEXT_RATIO,
                                            mask_ratio=MASK_RATIO)

            # Both encoder and target see edge images
            ctx_repr = encoder(imgs_edge, patch_indices=ctx_idx)

            with torch.no_grad():
                tgt_repr = target_encoder(imgs_edge, patch_indices=tgt_idx)
                tgt_repr = F.layer_norm(tgt_repr, [EMBED_DIM])

            pred_repr = predictor(ctx_repr, ctx_idx, tgt_idx)
            loss      = F.mse_loss(pred_repr, tgt_repr)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            ema_update(encoder, target_encoder, EMA_DECAY)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()
        print(f"  Epoch [{epoch+1:02d}/{FINETUNE_EPOCHS}]  loss: {avg_loss:.4f}")

    torch.save({
        "encoder":        encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor":      predictor.state_dict(),
        "loss_history":   loss_history,
        "config": {
            "variant":    "structure_focused",
            "depth":      6,
            "pred_depth": 3,
            "mask_ratio": MASK_RATIO,
        }
    }, CHECKPOINT)
    save_loss_curve(loss_history, "structure_focused", FINETUNE_EPOCHS)
    print(f"\n  ✓ Saved → {CHECKPOINT}")


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT 3: HIGH-MASK
# ─────────────────────────────────────────────────────────────────────────────
def train_high_mask():
    print("\n" + "="*60)
    print("  VARIANT 3: High-Mask JEPA")
    print("  Mask ratio: 0.5 → 0.75 (encoder sees only 25% of patches)")
    print("="*60 + "\n")

    MASK_RATIO = 0.75
    CHECKPOINT = "checkpoints/high_mask.pth"

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_data   = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)

    encoder, target_encoder, predictor = load_base_model()
    optimizer = make_optimizer(encoder, predictor)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    loss_history = []

    for epoch in range(FINETUNE_EPOCHS):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)

            # Context = 25%, Target = 75%
            ctx_idx, tgt_idx = sample_masks(
                context_ratio=1.0 - MASK_RATIO,   # 0.25
                mask_ratio=MASK_RATIO              # 0.75
            )

            ctx_repr = encoder(imgs, patch_indices=ctx_idx)

            with torch.no_grad():
                tgt_repr = target_encoder(imgs, patch_indices=tgt_idx)
                tgt_repr = F.layer_norm(tgt_repr, [EMBED_DIM])

            pred_repr = predictor(ctx_repr, ctx_idx, tgt_idx)
            loss      = F.mse_loss(pred_repr, tgt_repr)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            ema_update(encoder, target_encoder, EMA_DECAY)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()
        print(f"  Epoch [{epoch+1:02d}/{FINETUNE_EPOCHS}]  loss: {avg_loss:.4f}")

    torch.save({
        "encoder":        encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor":      predictor.state_dict(),
        "loss_history":   loss_history,
        "config": {
            "variant":    "high_mask",
            "depth":      6,
            "pred_depth": 3,
            "mask_ratio": MASK_RATIO,
        }
    }, CHECKPOINT)
    save_loss_curve(loss_history, "high_mask", FINETUNE_EPOCHS)
    print(f"\n  ✓ Saved → {CHECKPOINT}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Base checkpoint: {BASE_CHECKPOINT}")
    print(f"Fine-tuning epochs per variant: {FINETUNE_EPOCHS}")
    print(f"Estimated time: ~75 mins on RTX 3060\n")

    # Install opencv if not present
    try:
        import cv2
    except ImportError:
        print("Installing opencv-python...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
        import cv2

    # train_noise_robust()    
    train_structure_focused()
    train_high_mask()

    print("\n" + "="*60)
    print("  ALL VARIANTS TRAINED")
    print("="*60)
    print("\nCheckpoints saved:")
    print("  checkpoints/noise_robust.pth")
    print("  checkpoints/structure_focused.pth")
    print("  checkpoints/high_mask.pth")
    print("\nNext step: run evaluate.py on each variant")
    print("  python evaluate.py --checkpoint checkpoints/noise_robust.pth")
    print("  python evaluate.py --checkpoint checkpoints/structure_focused.pth")
    print("  python evaluate.py --checkpoint checkpoints/high_mask.pth")