#!/usr/bin/env python3
"""
Pack pretraining data into binary format for GPT training.

Inputs:
  ../data/tokenizer/legal_tokenizer.json
  ../data/processed/final_dataset.jsonl

Outputs:
  ../data/pretrain/train.bin
  ../data/pretrain/val.bin
  ../data/pretrain/meta.pkl
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def load_jsonl_texts(path: Path):
    """Loads each chunk’s text from final_dataset.jsonl."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if text:
                yield text


def pack(tokenizer_path: Path, dataset_path: Path, out_dir: Path):

    print(f"🔍 Loading tokenizer: {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))

    print(f"📄 Reading dataset: {dataset_path}")
    texts = list(load_jsonl_texts(dataset_path))
    print(f"✔ Loaded {len(texts)} chunks")

    random.shuffle(texts)
    val_size = max(1, int(len(texts) * 0.01))

    val_texts = texts[:val_size]
    train_texts = texts[val_size:]

    print(f"📘 Train size: {len(train_texts)}")
    print(f"📙 Val size:   {len(val_texts)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = []
    val_ids = []

    print("🔥 Encoding training data...")
    for i, txt in enumerate(train_texts):
        ids = tok.encode(txt).ids
        train_ids.extend(ids)
        if i % 5000 == 0 and i > 0:
            print(f"  {i}/{len(train_texts)} chunks encoded")

    print("⚙️ Encoding validation data...")
    for txt in val_texts:
        val_ids.extend(tok.encode(txt).ids)

    train_arr = np.array(train_ids, dtype=np.uint32)
    val_arr = np.array(val_ids, dtype=np.uint32)

    (out_dir / "train.bin").write_bytes(train_arr.tobytes())
    (out_dir / "val.bin").write_bytes(val_arr.tobytes())

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "tokenizer_path": str(tokenizer_path)
    }

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("🎉 Done! Saved:")
    print(f"  train.bin → {out_dir}")
    print(f"  val.bin →   {out_dir}")
    print(f"  meta.pkl →  {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="../data/tokenizer/legal_tokenizer.json")
    parser.add_argument("--data", default="../data/processed/final_dataset.jsonl")
    parser.add_argument("--out", default="../data/pretrain")
    args = parser.parse_args()

    pack(Path(args.tokenizer), Path(args.data), Path(args.out))


if __name__ == "__main__":
    main()
    