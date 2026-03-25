"""
train.py — Train JEPA-Lens baseline, then auto-run evaluate.py
============================================================
What happens:
  1. Train baseline JEPA on CIFAR-10 (standard masking, clean images)
  2. Save checkpoint to checkpoints/baseline.pth
  3. Auto-run evaluate.py — linear probe + t-SNE + understanding maps

Run:
  python train.py
"""

import os, copy, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import Encoder, Predictor, ema_update, sample_masks, EMBED_DIM


if __name__ == '__main__':

    # ── CONFIG ────────────────────────────────────────────────────────────────
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE    = 128
    EPOCHS        = 30
    LR            = 1.5e-4
    EMA_DECAY     = 0.996
    MASK_RATIO    = 0.5
    CONTEXT_RATIO = 0.5
    CHECKPOINT    = "checkpoints/baseline.pth"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs",     exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}")
    print(f"Checkpoint will be saved to: {CHECKPOINT}\n")

    # ── DATA ──────────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data   = datasets.CIFAR10('./data', train=True, download=True,
                                    transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)

    # ── MODEL ─────────────────────────────────────────────────────────────────
    encoder        = Encoder(depth=4).to(DEVICE)
    target_encoder = copy.deepcopy(encoder).to(DEVICE)
    predictor      = Predictor(depth=2).to(DEVICE)

    for p in target_encoder.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = (sum(p.numel() for p in encoder.parameters()) +
                    sum(p.numel() for p in predictor.parameters()))
    print(f"Trainable parameters: {total_params:,}\n")

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────
    loss_history = []

    for epoch in range(EPOCHS):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)

            ctx_idx, tgt_idx = sample_masks(context_ratio=CONTEXT_RATIO,
                                            mask_ratio=MASK_RATIO)

            # Context encoder (online — gets gradients)
            ctx_repr = encoder(imgs, patch_indices=ctx_idx)

            # Target encoder (EMA — no gradients)
            with torch.no_grad():
                tgt_repr = target_encoder(imgs, patch_indices=tgt_idx)
                tgt_repr = F.layer_norm(tgt_repr, [EMBED_DIM])

            # Predictor: guess target representations from context
            pred_repr = predictor(ctx_repr, ctx_idx, tgt_idx)

            # Loss: L2 in representation space — NOT pixel space
            loss = F.mse_loss(pred_repr, tgt_repr)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0
            )
            optimizer.step()
            ema_update(encoder, target_encoder, EMA_DECAY)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  loss: {avg_loss:.4f}  "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

    # ── SAVE CHECKPOINT ───────────────────────────────────────────────────────
    torch.save({
        "encoder":        encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor":      predictor.state_dict(),
        "loss_history":   loss_history,
        "config": {
            "variant":       "baseline",
            "mask_ratio":    MASK_RATIO,
            "context_ratio": CONTEXT_RATIO,
            "epochs":        EPOCHS,
        }
    }, CHECKPOINT)
    print(f"\n✓ Checkpoint saved → {CHECKPOINT}")

    # ── LOSS CURVE ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS+1), loss_history, "b-o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (representation space)")
    ax.set_title("JEPA-Lens Baseline — Training Loss\n"
                 "Falling = encoder learning better representations")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/loss_curve.png", dpi=150)
    plt.close()
    print("✓ Loss curve saved → outputs/loss_curve.png")

    # ── AUTO-RUN EVALUATE ─────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("Training complete. Auto-running evaluate.py...")
    print("─"*50 + "\n")
    subprocess.run([sys.executable, "evaluate.py", "--checkpoint", CHECKPOINT],
                   check=True)