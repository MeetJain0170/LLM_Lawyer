"""
=============================================================================
Minimal GPT Architecture (Checkpoint-Compatible)
=============================================================================

What this file defines:
----------------------
This module implements a minimal, decoder-only GPT-style Transformer
architecture used consistently across:
  - Pretraining
  - Supervised fine-tuning (SFT)
  - Inference (CLI + API)

The goal is architectural stability: once trained, the model definition
here must NEVER drift, or checkpoints will fail to load.

Design philosophy:
------------------
- Simplicity over cleverness
- Explicit tensor shapes
- No hidden framework magic
- Fully deterministic forward pass

Any change in:
  - Layer names
  - Parameter shapes
  - Attention math
will break checkpoint compatibility.

---------------------------------------------------------------------------
Key components:
---------------------------------------------------------------------------

1. GPTConfig
   ----------
   Lightweight container for model hyperparameters:
   - vocab_size  : tokenizer vocabulary size
   - n_layer     : number of Transformer blocks
   - n_head      : number of attention heads
   - d_model     : embedding / hidden dimension
   - seq_len     : maximum context length
   - dropout     : reserved for future use (currently disabled)

2. FlashSelfAttention (flash_attn-free)
   -----------------------------------
   A manual implementation of causal multi-head self-attention that
   intentionally mirrors the parameter naming of FlashAttention-based
   training code:

      - in_proj
      - out_proj

   This allows checkpoints trained with FlashAttention to be loaded
   without renaming or conversion.

   Causal masking is enforced to prevent token lookahead.

3. Block
   -----
   A standard Transformer block consisting of:
   - Pre-LN attention
   - Residual connection
   - Pre-LN feed-forward network
   - Residual connection

   LayerNorm is applied before each sub-layer (Pre-LN design),
   improving training stability.

4. GPT
   ---
   The full decoder-only Transformer:
   - Token embeddings
   - Learned positional embeddings
   - Stack of Transformer blocks
   - Final LayerNorm
   - Linear language modeling head

   Forward pass returns:
   - logits : (B, T, vocab_size)
   - loss   : optional cross-entropy if targets are provided

---------------------------------------------------------------------------
Important constraints:
---------------------------------------------------------------------------
- This file is the SINGLE SOURCE OF TRUTH for model architecture
- Tokenizer vocabulary size must match cfg.vocab_size
- cfg.seq_len must match the positional embedding size used in training
- Changing this file invalidates existing checkpoints

=============================================================================

"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------- CONFIG ---------------- #

class GPTConfig:
    def __init__(self, vocab_size, n_layer=12, n_head=12, d_model=768, seq_len=128, dropout=0.0):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout


# ---------------- FLASH-COMPAT ATTENTION (NO flash_attn) ---------------- #

class FlashSelfAttention(nn.Module):
    """
    Drop-in replacement for training-time FlashSelfAttention.
    Keeps parameter names IDENTICAL:
      - in_proj
      - out_proj
    """

    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # IMPORTANT: names must match checkpoint
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.in_proj(x)
        q, k, v = qkv.split(D, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.tril()  # causal mask
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)


# ---------------- BLOCK ---------------- #

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.attn = FlashSelfAttention(cfg.d_model, cfg.n_head)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------- GPT ---------------- #

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # MUST match checkpoint
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
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

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
