import torch
import os
import sys
from pathlib import Path

# --- CONFIG ---
INPUT_CKPT = Path("../checkpoints/sft/legal_llm_sft_final.pt")
OUTPUT_CKPT = Path("../checkpoints/sft/legal_llm_quantized.pt")
DEVICE = "cpu" # Quantization is usually done/run on CPU for PyTorch dynamic

try:
    from model import GPT, GPTConfig
    # Monkey patch for pickle
    setattr(sys.modules['__main__'], 'GPTConfig', GPTConfig)
except ImportError:
    print("❌ Error: model.py not found.")
    sys.exit(1)

def quantize():
    print(f"Loading fp32 model from {INPUT_CKPT}...")
    
    # 1. Load the original model
    ckpt = torch.load(INPUT_CKPT, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print(f"Original Size: {os.path.getsize(INPUT_CKPT) / 1024 / 1024:.2f} MB")

    # 2. Apply Dynamic Quantization (Linear layers only)
    # This converts the heavy matrix multiplication weights to 8-bit integers
    print("Quantizing...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},  # We only quantize Linear layers
        dtype=torch.qint8
    )

    # 3. Save
    print(f"Saving to {OUTPUT_CKPT}...")
    # We save the state dict of the quantized model
    # Note: Loading this back requires slightly different code (see below)
    torch.save(quantized_model.state_dict(), OUTPUT_CKPT)
    
    print(f"Quantized Size: {os.path.getsize(OUTPUT_CKPT) / 1024 / 1024:.2f} MB")
    print("✅ Done! Deployment ready.")

if __name__ == "__main__":
    quantize()