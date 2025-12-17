#!/usr/bin/env python3
"""
GPT Pretraining Script (FlashAttention v2, Crash-Proof)

Purpose:
--------
This script performs large-scale autoregressive pretraining of a GPT-style
Transformer using a high-performance, memory-efficient training stack.

It is designed to be resilient, restartable, and stable over long training runs,
even on imperfect datasets or unstable environments.

Key Features:
-------------
- FlashAttention v2 for fast and memory-efficient causal self-attention
- Gradient checkpointing to reduce activation memory
- BF16 mixed-precision training for speed and numerical stability
- Automatic sequence-length shrinking when datasets are small
- Safe binary data loading using NumPy (WSL / mounted disk compatible)
- Robust checkpointing with automatic resume support

Training Objective:
-------------------
- Standard causal language modeling (next-token prediction)
- Inputs and targets are token streams offset by one position
- Cross-entropy loss with optional ignore_index support

Model Architecture:
-------------------
- Decoder-only GPT Transformer
- Learned token embeddings and positional embeddings
- Pre-LayerNorm Transformer blocks
- FlashAttention2-based multi-head self-attention
- Feed-forward networks with GELU activation
- Final projection head tied to vocabulary size

Data Format:
------------
- Training and validation data are flat binary token streams (.bin)
- No explicit sequence boundaries are stored
- Training loop dynamically slices fixed-length windows
- Metadata (vocab size, tokenizer path) is loaded from meta.pkl

Stability and Safety Mechanisms:
--------------------------------
- Validates binary file existence and non-emptiness
- Prevents training crashes on small datasets by shrinking seq_len
- Uses gradient clipping to avoid exploding gradients
- Periodic validation to track generalization
- Time-based auto-saving to prevent progress loss
- Supports seamless resume from the latest checkpoint

Checkpointing Strategy:
-----------------------
- latest.pt : always overwritten, used for resume
- best.pt   : saved when validation loss improves
- Checkpoints include model, optimizer, scaler, step, config, and best loss

Precision and Performance:
--------------------------
- BF16 autocast for forward pass
- GradScaler for safe mixed-precision backward pass
- FlashAttention v2 reduces quadratic memory overhead
- Checkpointed blocks trade compute for memory efficiency

Assumptions and Constraints:
----------------------------
- Tokenizer vocabulary matches vocab_size in meta.pkl
- FlashAttention v2 is installed and CUDA is available for best performance
- Binary data files were generated with the same tokenizer
- This script is intended for long-running pretraining jobs

Important Warning:
------------------
This file defines the authoritative pretraining architecture.
Any architectural change here must be mirrored exactly in downstream
SFT and inference code, or checkpoints will become incompatible.

This script does not experiment.
It endures.
"""


import math
import time
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# FlashAttention v2
from flash_attn.flash_attn_interface import flash_attn_func


# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
class GPTConfig:
    def __init__(self, vocab_size, n_layer=12, n_head=12, d_model=768, seq_len=1024, dropout=0.0):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout


# -----------------------------------------------------
# FlashAttention2 Self-Attention
# -----------------------------------------------------
class FlashSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout

        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        # Compute Q, K, V
        qkv = self.in_proj(x)
        q, k, v = qkv.split(D, dim=2)

        # ---- FLASH ATTENTION v2 FORMAT ----
        # Expected input shape: (B, H, T, head_dim)

        def to_4d(t):
            return (
                t.view(B, T, self.n_head, self.head_dim)
                 .permute(0, 2, 1, 3)      # (B, H, T, head_dim)
                 .contiguous()
            )

        q = to_4d(q)
        k = to_4d(k)
        v = to_4d(v)

        # ---- FLASH ATTENTION v2 CALL ----
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout,
            causal=True
        )  # Output: (B, H, T, head_dim)

        # Convert back to (B, T, D)
        out = (
            out.permute(0, 2, 1, 3)   # (B, T, H, head_dim)
               .contiguous()
               .view(B, T, D)
        )

        return self.out_proj(out)


# -----------------------------------------------------
# BLOCK
# -----------------------------------------------------
class FlashBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.attn = FlashSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout)
        hidden = 4 * cfg.d_model
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
        )

    def forward(self, x):
        def inner(y):
            y = y + self.attn(self.ln1(y))
            y = y + self.ffn(self.ln2(y))
            return y

        # inside FlashBlock.forward
        return torch.utils.checkpoint.checkpoint(inner, x, use_reentrant=False)



# -----------------------------------------------------
# MODEL
# -----------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([FlashBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
        )
        return logits, loss


# -----------------------------------------------------
# DATA UTILITIES
# -----------------------------------------------------
def load_bin(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{path} is empty!")

    # ---- FIX: use NumPy fromfile (works on /mnt/d) ----
    print(f"Loading {path} using numpy.fromfile ...")
    try:
        data = np.fromfile(path, dtype=np.int32)
    except Exception as e:
        raise RuntimeError(f"numpy.fromfile failed: {e}")

    if data.size == 0:
        raise ValueError(f"NumPy loaded 0 elements from {path}")

    # convert to torch tensor (still CPU)
    return torch.from_numpy(data)


def get_batch(data, seq_len, batch_size):
    if len(data) <= seq_len + 1:
        seq_len = max(2, len(data) - 2)
        print(f"⚠️ Auto-shrinking seq_len → {seq_len}")

    max_start = len(data) - seq_len - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])

    # ensure integer token indices are int64 (torch.long) for cross_entropy
    return x.long(), y.long()



# -----------------------------------------------------
# TRAIN LOOP
# -----------------------------------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device}")

    data_dir = Path("../data/pretrain")
    meta = pickle.load(open(data_dir/"meta.pkl", "rb"))
    vocab_size = int(meta["vocab_size"])
    print(f"✔ vocab_size = {vocab_size}")

    train_data = load_bin(data_dir/"train.bin")
    val_data   = load_bin(data_dir/"val.bin")

    print(f"✔ train tokens: {len(train_data):,}")
    print(f"✔ val tokens:   {len(val_data):,}")
    print("Sample:", train_data[:10].tolist())

    # Model build
    cfg = GPTConfig(vocab_size=vocab_size, seq_len=128)
    model = GPT(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    batch_size = 8
    max_steps = 200_000

    ckpt_dir = Path("../checkpoints/pretrained")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest = ckpt_dir/"latest.pt"
    best_ckpt = ckpt_dir/"best.pt"

    # ------------------------
    # RESUME IF EXISTS
    # ------------------------
    start = 0
    best_loss = float("inf")

    if latest.exists():
        print("🔄 Resuming from latest...")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start = ckpt["step"]
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"➡️ Resumed @ step {start} | best_loss = {best_loss:.4f}")

    last_save = time.time()
    model.train()

    # -----------------------------------------------------
    # helper: evaluate val loss
    # -----------------------------------------------------
    @torch.no_grad()
    def get_val_loss():
        model.eval()
        xb, yb = get_batch(val_data, cfg.seq_len, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        model.train()
        return loss.item()

    # -----------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------
    for step in range(start, max_steps):
        xb, yb = get_batch(train_data, cfg.seq_len, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"step {step} | train loss {loss.item():.4f}")

        # -----------------------------------------------------
        # VALIDATION + CHECKPOINTING
        # -----------------------------------------------------
        if step % 1000 == 0 and step > start:
            val_loss = get_val_loss()
            print(f"🔍 Validation loss @ step {step}: {val_loss:.4f}")

            # ---- Save latest ----
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": cfg,
                "best_loss": best_loss,
            }, latest)
            print("💾 Saved latest checkpoint")

            # ---- Save best ----
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "config": cfg,
                    "best_loss": best_loss,
                }, best_ckpt)
                print(f"🏆 Saved BEST checkpoint (loss {best_loss:.4f})")

        # -----------------------------------------------------
        # TIME-BASED CHECKPOINT
        # -----------------------------------------------------
        if time.time() - last_save > 900:  # 15 minutes
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": cfg,
                "best_loss": best_loss,
            }, latest)
            print("🕒 Auto-saved latest checkpoint")
            last_save = time.time()

    print("🎉 Training complete!")
if __name__ == "__main__":
    train()