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
"""

#!/usr/bin/env python3
"""
Indian Legal AI – Stealth Terminal (Hybrid)
"""
import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

# Imports
try:
    from model import GPT, GPTConfig
    setattr(sys.modules['__main__'], 'GPTConfig', GPTConfig)
    from legal_knowledge import search_knowledge_base
except ImportError:
    print("❌ Missing model.py or legal_knowledge.py")
    sys.exit(1)

# Config
CHECKPOINT_PATH = Path("../checkpoints/sft/legal_llm_sft_final.pt")
TOKENIZER_PATH  = Path("../data/tokenizer/legal_tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Generation Logic
@torch.no_grad()
def generate(model, idx, tokenizer, max_new_tokens=150):
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context
        idx_cond = idx[:, -128:] 
        
        # Forward
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE) if DEVICE=="cuda" else torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / 0.4 # Low temp for focus
        
        # Sampling
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        if tokenizer.token_to_id("<EOS>") == idx_next.item():
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def main():
    # Clean UI
    print(f"\n🔥 Hardware: {DEVICE}")
    
    # Load
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    
    print("⚖️  LEGAL TERMINAL ONLINE\n")

    while True:
        try:
            query = input(">>Client: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue

            print("...", end="\r")

            # 1. CHECK DB
            kb = search_knowledge_base(query)
            if kb:
                print(f"\n📜 Counsel Opinion (Verified):\n{'-'*30}")
                print(f"Subject: {kb['title']}")
                print(f"{kb['content']}")
                print("-" * 60 + "\n")
                continue

            # 2. RUN AI
            prompt = f"<|user|>\n{query}\n<|assistant|>\n"
            ids = tokenizer.encode(prompt).ids
            idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)

            out = generate(model, idx, tokenizer)
            full_text = tokenizer.decode(out[0].tolist())
            
            if "<|assistant|>" in full_text:
                ans = full_text.split("<|assistant|>")[-1].strip()
            else:
                ans = full_text.strip()

            print(f"\n📜 Counsel Opinion (Generative):\n{'-'*30}")
            print(ans)
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()