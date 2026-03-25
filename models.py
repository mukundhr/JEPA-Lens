"""
models.py — Shared architecture for JEPA-Lens
All 4 models (baseline + 3 variants) use these same classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 32
PATCH_SIZE  = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 64
EMBED_DIM   = 192
NUM_HEADS   = 4


# ── PATCH EMBEDDING ───────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """
    Splits image into non-overlapping patches and linearly projects each.
    (B, 3, 32, 32) → (B, 64, EMBED_DIM)
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.proj(x)                      # (B, D, 8, 8)
        x = x.flatten(2).transpose(1, 2)      # (B, 64, D)
        return x


# ── TRANSFORMER BLOCK ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True,
                                           dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n  = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x  = x + a
        x  = x + self.mlp(self.norm2(x))
        return x


# ── ENCODER ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Online encoder — the thing that learns "what matters."
    Can process all patches or a selected subset (context patches).
    """
    def __init__(self, depth=4):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed   = nn.Parameter(
            torch.randn(1, NUM_PATCHES, EMBED_DIM) * 0.02
        )
        self.blocks = nn.Sequential(
            *[TransformerBlock(EMBED_DIM, NUM_HEADS) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(EMBED_DIM)

    def forward(self, x, patch_indices=None):
        """
        x             : (B, 3, 32, 32)
        patch_indices : list of ints — which patches to encode (None = all)
        returns       : (B, n_patches, EMBED_DIM)
        """
        tokens = self.patch_embed(x) + self.pos_embed   # (B, 64, D)
        if patch_indices is not None:
            tokens = tokens[:, patch_indices, :]
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens


# ── PREDICTOR ─────────────────────────────────────────────────────────────────
class Predictor(nn.Module):
    """
    Given context representations, predicts representations at TARGET positions.
    Narrower than encoder — it only needs to bridge the gap.
    """
    def __init__(self, depth=2):
        super().__init__()
        pred_dim = EMBED_DIM // 2
        self.input_proj  = nn.Linear(EMBED_DIM, pred_dim)
        self.pos_embed   = nn.Parameter(
            torch.randn(1, NUM_PATCHES, pred_dim) * 0.02
        )
        self.blocks = nn.Sequential(
            *[TransformerBlock(pred_dim, NUM_HEADS // 2) for _ in range(depth)]
        )
        self.norm        = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, EMBED_DIM)

    def forward(self, context_repr, context_indices, target_indices):
        """
        context_repr    : (B, n_ctx, EMBED_DIM)
        context_indices : list of ints — positional origin of context tokens
        target_indices  : list of ints — positions we need to predict
        returns         : (B, n_tgt, EMBED_DIM) — predicted target representations
        """
        B   = context_repr.shape[0]
        x   = self.input_proj(context_repr)                           # (B, n_ctx, pred_dim)
        x   = x + self.pos_embed[:, context_indices, :]               # add ctx positions
        tgt = self.pos_embed[:, target_indices, :].expand(B, -1, -1)  # target position queries
        x   = torch.cat([x, tgt], dim=1)                              # (B, n_ctx+n_tgt, pred_dim)
        x   = self.blocks(x)
        x   = self.norm(x)
        return self.output_proj(x[:, len(context_indices):, :])       # (B, n_tgt, EMBED_DIM)


# ── EMA UPDATE ────────────────────────────────────────────────────────────────
@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, decay: float):
    """
    Slowly update target encoder toward online encoder.
    This is the anti-collapse mechanism — target stays stable
    while online encoder chases a moving but coherent target.
    """
    for op, tp in zip(online.parameters(), target.parameters()):
        tp.data = decay * tp.data + (1.0 - decay) * op.data


# ── MASKING ───────────────────────────────────────────────────────────────────
import numpy as np

def sample_masks(num_patches=NUM_PATCHES, context_ratio=0.5, mask_ratio=0.5):
    """Returns (context_indices, target_indices) — disjoint, randomly sampled."""
    perm    = np.random.permutation(num_patches)
    n_ctx   = int(num_patches * context_ratio)
    n_tgt   = int(num_patches * mask_ratio)
    ctx_idx = sorted(perm[:n_ctx].tolist())
    tgt_idx = sorted(perm[n_ctx:n_ctx + n_tgt].tolist())
    return ctx_idx, tgt_idx