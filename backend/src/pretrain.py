#!/usr/bin/env python3
"""
Optimized + Crash-Proof GPT Pretraining Script
- FlashAttention
- Gradient Checkpointing
- BF16 mixed precision
- Auto-save every N steps
- Auto-save every X minutes
- Resume from latest checkpoint
"""

import math
import time
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn.modules.mha import MHA
from flash_attn.ops.fused_dense import FusedDense


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
# MODEL
# -----------------------------------------------------
class FlashBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.attn = MHA(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_head,
            dropout=cfg.dropout,
            causal=True,
        )

        self.ff = nn.Sequential(
            FusedDense(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            FusedDense(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        def checkpoint_forward(x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

        return torch.utils.checkpoint.checkpoint(checkpoint_forward, x)


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([FlashBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss


# -----------------------------------------------------
# DATA
# -----------------------------------------------------
def get_batch(data, seq_len, batch_size):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


# -----------------------------------------------------
# TRAIN (CRASH-PROOF)
# -----------------------------------------------------
def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device}")

    # Load vocab + dataset
    meta = pickle.load(open("../data/pretrain/meta.pkl", "rb"))
    vocab_size = meta["vocab_size"]

    train_data = torch.from_file("../data/pretrain/train.bin", dtype=torch.int32).to(device)
    val_data = torch.from_file("../data/pretrain/val.bin", dtype=torch.int32).to(device)

    # Build model
    cfg = GPTConfig(vocab_size=vocab_size)
    model = GPT(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    batch_size = 8  # manageable for 4070 SUPER
    max_steps = 100_000  # 5–7 epochs

    ckpt_dir = Path("../checkpoints/pretrained")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pt"

    # -----------------------------------------------------
    # RESUME SUPPORT
    # -----------------------------------------------------
    start_step = 0
    if latest_ckpt.exists():
        print("🔄 Resuming from latest checkpoint...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"➡️  Resumed at step {start_step}")

    last_save_time = time.time()

    model.train()
    for step in range(start_step, max_steps):

        xb, yb = get_batch(train_data, cfg.seq_len, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

        # -----------------------------------------------------
        # SAVE EVERY 1000 STEPS
        # -----------------------------------------------------
        if step % 1000 == 0 and step > start_step:
            save_path = ckpt_dir / f"step_{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": cfg,
            }, save_path)

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": cfg,
            }, latest_ckpt)

            print(f"💾 Saved checkpoint: {save_path}")
            print(f"💾 Updated latest checkpoint.")

        # -----------------------------------------------------
        # TIME-BASED CHECKPOINT (every 15 minutes)
        # -----------------------------------------------------
        if time.time() - last_save_time > 900:  # 900 sec = 15 mins
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": cfg,
            }, latest_ckpt)

            print("🕒 Auto-saved latest checkpoint (15 min interval).")
            last_save_time = time.time()

    print("🎉 Training DONE!")


if __name__ == "__main__":
    train()
