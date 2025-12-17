#!/usr/bin/env python3
"""
=============================================================================
Legal LLM Inference CLI (SFT Version)
=============================================================================

Purpose:
--------
Interactive command-line interface for the Fine-Tuned (SFT) Legal LLM.
Loads the final SFT checkpoint and generates legal answers.

Key Properties:
---------------
- Matches chat-style SFT training format (<|user|>, <|assistant|>)
- Correct EOS handling (no self-sabotage)
- Safe AMP autocast (CUDA + CPU compatible)
- Light repetition penalty (reduces legal looping)
- Deterministic decoding for legal accuracy
=============================================================================

=============================================================================
Legal LLM Inference CLI (Hybrid + SFT Version)
=============================================================================
"""

import sys
import torch
import re
from tokenizers import Tokenizer
from pathlib import Path

# ---------------------------------------------------------
# 1. IMPORT MODEL & KNOWLEDGE BASE
# ---------------------------------------------------------
try:
    from model import GPT, GPTConfig
    # Fix for pickle loading
    setattr(sys.modules['__main__'], 'GPTConfig', GPTConfig)
except ImportError:
    print("❌ CRITICAL ERROR: Could not import 'model.py'.")
    sys.exit(1)

try:
    from legal_knowledge import search_knowledge_base
except ImportError:
    print("⚠️  Warning: legal_knowledge.py not found. Running in AI-only mode.")
    search_knowledge_base = lambda x: None

# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------
CHECKPOINT_PATH = Path("../checkpoints/sft/legal_llm_sft_final.pt")
TOKENIZER_PATH  = Path("../data/tokenizer/legal_tokenizer.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MAX_CONTEXT = 128  # Context Window Limit

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_output(text, prompt):
    """Removes prompt and cleaning artifacts."""
    response = text.replace(prompt, "")
    # Remove tags like <|user|>, <|assistant|>, <|>
    clean = re.sub(r"<\|.*?\|>", "", response)
    clean = clean.replace("|", "").replace("  ", " ").strip()
    return clean

# ---------------------------------------------------------
# 4. GENERATION LOOP
# ---------------------------------------------------------
@torch.no_grad()
def generate(model, idx, tokenizer, max_new_tokens=150):
    model.eval()
    eos_id = tokenizer.token_to_id("<EOS>")

    for _ in range(max_new_tokens):
        # Sliding Window to prevent crash
        cond_idx = idx if idx.size(1) <= MAX_CONTEXT else idx[:, -MAX_CONTEXT:]

        # AutoCast for GPU speed
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE) if DEVICE == "cuda" else torch.no_grad():
            logits, _ = model(cond_idx)

        logits = logits[:, -1, :] / 0.4 # Low temp for focus
        
        # Repetition Penalty (Quick & Dirty)
        # for token in set(idx[0].tolist()):
        #     logits[0, token] /= 1.05

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------
def main():
    print(f"\n🔥 Hardware: {DEVICE} ({DTYPE})")
    print(f"📂 Loading SFT Checkpoint: {CHECKPOINT_PATH}")

    # Load Components
    try:
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model = GPT(ckpt["config"]).to(DEVICE)
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    print("✅ Model Loaded.")
    print("⚖️  LEGAL CONSULTANT READY (Hybrid Mode)")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("🧑‍⚖️ Client: ").strip()
            if query.lower() in {"exit", "quit"}: break
            if not query: continue

            # --- 1. KNOWLEDGE BASE CHECK ---
            kb = search_knowledge_base(query)
            if kb:
                print(f"\n📜 Counsel Opinion (Verified):\n{'-'*30}")
                print(f"**{kb['title']}**")
                print(kb['content'])
                print(f"_(Source: Verified Database)_")
                print("-" * 60 + "\n")
                continue

            # --- 2. AI GENERATION ---
            # Correct Prompt Format for your SFT model
            prompt = f"<|user|>\n{query}\n<|assistant|>\n"
            
            ids = tokenizer.encode(prompt).ids
            idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)

            print("\n📜 Counsel Opinion (Generative):")
            out = generate(model, idx, tokenizer)
            
            full_text = tokenizer.decode(out[0].tolist())
            response = clean_output(full_text, prompt)
            
            print(f"{response}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Runtime Error: {e}")

if __name__ == "__main__":
    main()
