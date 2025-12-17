#!/usr/bin/env python3
"""
Pack chat-style instruction data with LOSS MASKING
"""

import json
import pickle
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

DATA = Path("../data/processed/instruction_dataset_chat.jsonl")
TOKENIZER = Path("../data/tokenizer/legal_tokenizer.json")
OUT_DIR = Path("../data/instruction")

def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_ids = []
    labels = []

    with DATA.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]

            ids = tokenizer.encode(text).ids

            # Find where assistant starts
            assist_token = tokenizer.encode("<|assistant|>").ids
            assist_pos = None
            for i in range(len(ids) - len(assist_token)):
                if ids[i:i+len(assist_token)] == assist_token:
                    assist_pos = i + len(assist_token)
                    break

            if assist_pos is None:
                continue

            lbl = [-1] * assist_pos + ids[assist_pos:]

            input_ids.extend(ids)
            labels.extend(lbl)

    np.array(input_ids, dtype=np.int32).tofile(OUT_DIR / "train.bin")
    np.array(labels, dtype=np.int32).tofile(OUT_DIR / "labels.bin")

    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(
            {"vocab_size": tokenizer.get_vocab_size(), "format": "chat-sft"},
            f
        )

    print("🎉 Packed chat-style SFT data with masking")

if __name__ == "__main__":
    main()
