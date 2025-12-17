import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# --- CONFIG ---
CKPT_PATH = Path("../checkpoints/sft/legal_llm_sft_final.pt")
# CORRECTED PATH BELOW:
TOKENIZER_PATH = Path("../data/tokenizer/legal_tokenizer.json") 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from model import GPT, GPTConfig
    # Fix for pickle error
    setattr(sys.modules['__main__'], 'GPTConfig', GPTConfig)
    from tokenizers import Tokenizer
except ImportError:
    print("❌ Missing model.py or tokenizers")
    sys.exit(1)

def main():
    print(f"--- 🩺 MODEL X-RAY DIAGNOSTIC ---")
    
    # 1. Load Model
    print(f"Loading {CKPT_PATH}...")
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model = GPT(checkpoint["config"]).to(DEVICE)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        print("✅ Model Loaded.")
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")
        return

    # 2. Load Tokenizer
    print(f"Loading {TOKENIZER_PATH}...")
    try:
        enc = Tokenizer.from_file(str(TOKENIZER_PATH))
        print("✅ Tokenizer Loaded.")
    except Exception as e:
        print(f"❌ Tokenizer Load Failed: {e}")
        return

    # 3. The Test Prompt
    query = "What is the punishment for theft?"
    # Use the format your tokenizer/data expects. 
    # If you used <|user|>, keep it. If unsure, we try a standard format.
    prompt = f"<|user|>\n{query}\n<|assistant|>\n"
    
    try:
        ids = enc.encode(prompt).ids
    except:
        print("⚠️ Tokenizer failed to encode special tokens. Trying raw text...")
        ids = enc.encode(prompt, add_special_tokens=False).ids

    idx = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    print(f"\nPrompt: {repr(prompt)}")
    print(f"Token IDs: {ids}")

    # 4. PREDICT NEXT TOKEN (Top 5 Analysis)
    with torch.no_grad():
        logits, _ = model(idx)
        last_token_logits = logits[:, -1, :]
        probs = F.softmax(last_token_logits, dim=-1)
        
        # Get Top 5 candidates for the FIRST generated word
        top_probs, top_indices = torch.topk(probs, 5)
        
        print("\n--- 🧠 WHAT THE MODEL WANTS TO SAY NEXT ---")
        for i in range(5):
            token_id = top_indices[0][i].item()
            probability = top_probs[0][i].item()
            decoded_token = enc.decode([token_id])
            print(f"Option {i+1}: '{decoded_token}' (Confidence: {probability:.4f})")

    # 5. FORCE GENERATION (Safety Off)
    print("\n--- 🔨 FORCING GENERATION (Ignoring Stop Tokens) ---")
    
    for _ in range(50):
        with torch.no_grad():
            logits, _ = model(idx)
            logits = logits[:, -1, :] / 0.6  # Low temp for stability
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            token_str = enc.decode([idx_next.item()])
            print(token_str, end="", flush=True)
            
            idx = torch.cat((idx, idx_next), dim=1)
    print("\n\n--------------------------------------------")

if __name__ == "__main__":
    main()