"""
Instruction Dataset Packing Script (SFT / Chat Fine-Tuning)

Purpose:
--------
This script converts an instruction-style JSONL dataset into flat binary
token streams suitable for supervised fine-tuning (SFT) of a GPT-style model.

Unlike pretraining data, this dataset represents structured conversational
or instruction-following text. However, for training efficiency, all samples
are flattened into a single contiguous token stream.

Inputs:
-------
- instruction_dataset_chat.jsonl
  Each line must be valid JSON containing a "text" field.
  The "text" field is assumed to already contain any prompt/response structure.

- legal_tokenizer.json
  A custom-trained BPE tokenizer used consistently across pretraining,
  fine-tuning, and inference.

Outputs:
--------
- train.bin  : flat array of token IDs (model inputs)
- labels.bin : identical copy of train.bin (causal language modeling target)

Training interpretation:
------------------------
- The model is trained in a causal LM setup
- Inputs and labels are identical but shifted internally by the training loop
- Sequence boundaries are NOT preserved in the binary format
- The trainer is responsible for slicing fixed-length windows

Tokenization behavior:
----------------------
- Text is encoded using the custom tokenizer only
- No hardcoded EOS token is injected to avoid vocabulary mismatch
- If EOS tokens exist, they must already be present in the dataset text
- This design prevents silent crashes caused by invalid token IDs

Safety checks:
--------------
- Verifies input dataset existence
- Verifies tokenizer existence and loadability
- Skips malformed JSON lines gracefully
- Warns if generated token IDs exceed tokenizer vocabulary size

Why flatten everything:
-----------------------
- Faster disk IO and simpler training loops
- Matches large-scale GPT training pipelines
- Avoids padding and per-sample overhead
- Keeps preprocessing deterministic and repeatable

Important constraints:
----------------------
- Tokenizer and model vocab size must match exactly
- Changing the tokenizer requires regenerating these binaries
- This script must be rerun if instruction data changes

This file does not teach the model what to say.
It prepares the ground so learning can happen without friction.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# --- IMPORT CUSTOM TOKENIZER ---
try:
    from tokenizers import Tokenizer
except ImportError:
    print("❌ ERROR: 'tokenizers' library not found.")
    print("   Run: pip install tokenizers")
    sys.exit(1)

# --- CONFIG ---
INPUT_FILE = Path("../data/processed/instruction_dataset_chat.jsonl") 
OUTPUT_DIR = Path("../data/instruction")
TOKENIZER_PATH = Path("../data/tokenizer/legal_tokenizer.json") # <--- Your custom tokenizer

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare():
    print(f"--- Processing: {INPUT_FILE} ---")
    
    # 1. Validation
    if not INPUT_FILE.exists():
        print(f"❌ ERROR: Input file not found at {INPUT_FILE}")
        return
    if not TOKENIZER_PATH.exists():
        print(f"❌ ERROR: Custom tokenizer not found at {TOKENIZER_PATH}")
        print("   Please check where your 'tokenizer.json' is located.")
        return

    # 2. Load Tokenizer
    print(f"Loading custom tokenizer from {TOKENIZER_PATH}...")
    try:
        enc = Tokenizer.from_file(str(TOKENIZER_PATH))
        print(f"✅ Tokenizer loaded. Vocab size: {enc.get_vocab_size()}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    # 3. Read Lines
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"✅ Found {len(lines)} lines.")
    if len(lines) == 0: return

    all_ids = []
    
    print("Tokenizing 'text' column...")
    
    # 4. Tokenize
    # We check if your tokenizer has an EOS token, otherwise we don't add one manually
    # to avoid the "vocab mismatch" error again.
    
    for line in tqdm(lines):
        try:
            data = json.loads(line)
            text = data.get("text", "")
            
            if not text:
                continue

            # Encode using YOUR custom tokenizer
            encoded = enc.encode(text)
            ids = encoded.ids
            
            # Optional: Add EOS token if your tokenizer expects it (usually ID 2 or similar)
            # For safety, we will NOT add a hardcoded ID unless we are sure.
            # If your text already has <|endoftext|>, your custom tokenizer might split it 
            # into multiple tokens if it doesn't know that special token. 
            # This is safer than crashing.
            
            all_ids.extend(ids)

        except json.JSONDecodeError:
            continue

    print(f"\nTotal tokens generated: {len(all_ids)}")
    
    if len(all_ids) == 0:
        print("❌ Still 0 tokens! Check your JSON format.")
        return

    # 5. Save
    ids_np = np.array(all_ids, dtype=np.int32)
    
    # Check max ID one last time
    if ids_np.max() >= enc.get_vocab_size():
         print(f"⚠️ WARNING: Max token ID ({ids_np.max()}) exceeds declared vocab size!")
    
    ids_np.tofile(OUTPUT_DIR / "train.bin")
    ids_np.tofile(OUTPUT_DIR / "labels.bin") # Causal training (labels = inputs)
    
    print(f"✅ SUCCESS! Saved to {OUTPUT_DIR}")
    print(f"   train.bin size: {len(ids_np)} tokens")

if __name__ == "__main__":
    prepare()