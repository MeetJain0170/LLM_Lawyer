import torch
import sys
import time
import re
import warnings
from pathlib import Path

# --- CONFIG ---
# We use the ORIGINAL checkpoint for GPU (Higher Quality)
CHECKPOINT_PATH = Path("../checkpoints/sft/legal_llm_sft_final.pt")
TOKENIZER_PATH = Path("../data/tokenizer/legal_tokenizer.json")
DEVICE = "cuda"  # 🔥 RTX 4070 Power
MAX_CONTEXT = 128  # Fixes the "Index out of range" crash

# Suppress those annoying warnings
warnings.filterwarnings("ignore")

try:
    from model import GPT, GPTConfig
    setattr(sys.modules['__main__'], 'GPTConfig', GPTConfig)
    from tokenizers import Tokenizer
except ImportError:
    print("❌ Error: Missing model.py")
    sys.exit(1)

def clean_output(text, prompt):
    """
    Cleans up the messy output from the SFT model.
    """
    # 1. Remove the prompt so we just see the answer
    response = text.replace(prompt, "")
    
    # 2. Extract meaningful sentences if tags are broken
    clean = re.sub(r"<\|.*?\|>", "", response)
    clean = clean.replace("|", "").replace("  ", " ").strip()
    
    # 3. Emergency Cleanup: If it repeats "User", cut it off
    if "User" in clean:
        clean = clean.split("User")[0]
        
    return clean

def main():
    if not torch.cuda.is_available():
        print("❌ Error: CUDA not found. Please run 'infer_quantized.py' instead.")
        sys.exit(1)

    print(f"🔥 Initializing CUDA on {torch.cuda.get_device_name(0)}...")
    
    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    
    # 2. Load Model (Original BF16)
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model = GPT(ckpt["config"]).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print("✅ GPU Model Ready (BF16 Mode).\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    while True:
        query = input("User: ").strip()
        if query.lower() in ["quit", "exit"]: break
        if not query: continue
        
        # Prepare Prompt
        prompt = f"<|user|>\n{query}\n<|assistant|>\n"
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        start = time.time()
        
        with torch.no_grad():
            # Use AutoCast for speed (BF16)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                # Generate up to 150 tokens
                for _ in range(150):
                    # --- SLIDING WINDOW FIX ---
                    # Only look at the last 128 tokens to prevent crashing
                    cond_idx = idx if idx.size(1) <= MAX_CONTEXT else idx[:, -MAX_CONTEXT:]
                    
                    logits, _ = model(cond_idx)
                    logits = logits[:, -1, :] / 0.5 # Temp 0.5 for stability
                    
                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    # Stop if EOS
                    if idx_next.item() == tokenizer.token_to_id("<EOS>"):
                        break
                        
                    idx = torch.cat((idx, idx_next), dim=1)

        end = time.time()
        
        # Decode & Clean
        raw_output = tokenizer.decode(idx[0].tolist())
        final_response = clean_output(raw_output, prompt)
        
        print(f"\nLawyer (GPU): {final_response}")
        print(f"(Latency: {end - start:.4f}s)\n")

if __name__ == "__main__":
    main()