#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Script for Legal LLM

Purpose:
--------
This script performs supervised fine-tuning (SFT) on a pretrained GPT model
using instruction-style conversational data. It adapts the base language model
to follow prompts, answer questions, and respond in a controlled, domain-
specific manner (Indian legal context).

This is the final training stage before inference and deployment.

Key Features:
-------------
- Cosine learning rate decay with linear warmup
- Automatic Mixed Precision (AMP) with bfloat16 / float16
- Gradient accumulation for large effective batch sizes on limited VRAM
- Gradient clipping for stability
- Periodic evaluation and checkpointing
- Checkpoint-compatible with pretraining architecture

Training Objective:
-------------------
- Standard causal language modeling loss (next-token prediction)
- Inputs and labels are offset by one token
- Loss is computed over all tokens (no masking of prompt vs response)
- Suitable for instruction-following behavior learned implicitly from data

Data Format:
------------
- train.bin  : flat stream of token IDs (inputs)
- labels.bin : identical stream, shifted internally to form targets
- Data is sliced into fixed-length windows during training
- No explicit sample boundaries are preserved

Model Loading:
--------------
- Loads a pretrained GPT checkpoint (from pretraining stage)
- Restores both model weights and architectural configuration
- Uses the exact same GPT and GPTConfig definitions to ensure compatibility

Important Compatibility Fix:
----------------------------
A monkey-patch is applied to ensure GPTConfig is discoverable during checkpoint
deserialization. This prevents the common:
  "AttributeError: Can't get attribute 'GPTConfig'"
error when loading older checkpoints.

Optimization Strategy:
----------------------
- AdamW optimizer with decoupled weight decay
- Very low learning rate compared to pretraining
- Cosine decay schedule to gently converge
- Gradient accumulation to simulate larger batch sizes
- Gradient norm clipping to prevent instability

Precision Handling:
-------------------
- Uses bfloat16 when supported (RTX 30/40 series)
- Falls back to float16 otherwise
- AMP autocast for forward pass
- GradScaler enabled only when necessary

Safety and Debugging Checks:
----------------------------
- Verifies dataset size relative to model context window
- Prevents silent training on insufficient data
- Logs learning rate, loss, and step timing
- Saves intermediate checkpoints after warmup

Outputs:
--------
- Intermediate SFT checkpoints during training
- Final fine-tuned model:
    legal_llm_sft_final.pt

Constraints and Warnings:
-------------------------
- Tokenizer vocabulary must match model vocab_size
- Instruction data must be sufficiently large
- Changing sequence length or architecture invalidates checkpoints
- This script assumes single-GPU training

This script does not explore.
It refines.

Where pretraining builds knowledge,
SFT teaches judgment.
"""


import sys
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# --- IMPORT MODEL CLASSES ---
try:
    from model import GPT, GPTConfig
except ImportError:
    print("CRITICAL ERROR: Could not import 'model.py'. Make sure it is in the same folder.")
    sys.exit(1)

# --- THE FIX: MONKEY PATCH ---
# This fixes the "AttributeError: Can't get attribute 'GPTConfig'"
setattr(sys.modules['__main__'], 'GPTConfig', GPTConfig)

# --- Configuration ---
DATA_DIR = Path("../data/instruction")
CKPT = Path("../checkpoints/base_pretrained/final_pretrained.pt")
OUT = Path("../checkpoints/sft")
OUT.mkdir(exist_ok=True, parents=True)

# Hyperparameters
BATCH_SIZE = 4          # Micro-batch size (limited by VRAM)
GRAD_ACCUM_STEPS = 8    # 4 * 8 = 32 effective batch size
MAX_LR = 1e-5           # Lower LR for SFT than pre-training
MIN_LR = 1e-6           
WARMUP_ITERS = 100
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 if available (RTX 30/40 series), otherwise float16
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def load_bin(path):
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    return torch.from_numpy(np.fromfile(path, dtype=np.int32)).long()

def get_batch(x, y, seq_len, batch_size):
    # Safety check
    # We need +1 room because we are shifting the targets
    if len(x) <= seq_len + 1:
        ix = torch.zeros(batch_size, dtype=torch.long)
    else:
        # Subtract 1 extra from the range to ensure we don't go out of bounds when shifting y
        ix = torch.randint(0, len(x) - seq_len - 1, (batch_size,))
    
    xb = torch.stack([x[i : i+seq_len] for i in ix])
    
    # --- CRITICAL FIX: SHIFT TARGETS ---
    # Since y is a copy of x, we grab the slice offset by 1.
    # Input:  [A, B, C]
    # Target: [B, C, D]
    yb = torch.stack([y[i+1 : i+seq_len+1] for i in ix])
    
    return xb.to(DEVICE), yb.to(DEVICE)

# --- Cosine Learning Rate Scheduler ---
def get_lr(it):
    if it < WARMUP_ITERS:
        return MAX_LR * (it + 1) / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

@torch.no_grad()
def estimate_loss(model, x, y, seq_len):
    out = {}
    model.eval()
    for split, x_data, y_data in [('train', x, y)]: # simplified for now
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(x_data, y_data, seq_len, BATCH_SIZE)
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    print(f"--- Starting SFT on {DEVICE} using {DTYPE} ---")
    
    # 1. LOAD MODEL FIRST (To get seq_len)
    print(f"Loading checkpoint from {CKPT}...")
    try:
        ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    except Exception as e:
        print(f"ERROR Loading Checkpoint: {e}")
        return

    cfg = ckpt["config"]
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    print("Model loaded successfully.")

    # 2. LOAD DATA
    print("Loading data...")
    data_x = load_bin(DATA_DIR / "train.bin")
    data_y = load_bin(DATA_DIR / "labels.bin")

    # --- SAFETY CHECK ---
    print(f"\n[DEBUG] Data Stats:")
    print(f"  Model Context Window (seq_len): {cfg.seq_len}")
    print(f"  Total Tokens in train.bin:      {len(data_x)}")
    
    if len(data_x) <= cfg.seq_len:
        print("\n❌ CRITICAL ERROR: Your training data is too small!")
        print(f"   You have {len(data_x)} tokens, but the model needs chunks of {cfg.seq_len}.")
        print("   -> Go back to your data preparation step.")
        print("   -> Ensure you generated enough data in 'train.bin'.")
        sys.exit(1)
    
    # Simple split (90/10)
    n = int(0.9 * len(data_x))
    train_x, val_x = data_x[:n], data_x[n:]
    train_y, val_y = data_y[:n], data_y[n:]
    
    # 3. OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))
    
    # Scaler for mixed precision (Updated syntax)
    scaler = torch.amp.GradScaler('cuda', enabled=(DTYPE == torch.float16)) 
    
    t0 = time.time()
    
    for iter_num in range(1, MAX_ITERS + 1):
        
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS:
            losses = estimate_loss(model, train_x, train_y, cfg.seq_len)
            print(f"step {iter_num}: train loss {losses['train']:.4f}")
            
            if iter_num > 500:
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "iter": iter_num
                }, OUT / f"ckpt_sft_{iter_num}.pt")

        for micro_step in range(GRAD_ACCUM_STEPS):
            xb, yb = get_batch(train_x, train_y, cfg.seq_len, BATCH_SIZE)
            
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
                loss = loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"step {iter_num} | lr {lr:.2e} | loss {loss.item() * GRAD_ACCUM_STEPS:.4f} | time {dt*1000:.2f}ms")

    torch.save({"model": model.state_dict(), "config": cfg}, OUT / "legal_llm_sft_final.pt")
    print("🎉 CHAT SFT COMPLETE")

if __name__ == "__main__":
    train()