#!/usr/bin/env python3
"""
Train a BPE tokenizer from scratch using HuggingFace `tokenizers`
on the preprocessed legal corpus.

Input:
  ../data/processed/final_dataset.jsonl   (each line has a "text" field)

Output:
  ../data/tokenizer/legal_tokenizer.json

Usage (from backend/src):

  python3 hf_train_tokenizer.py \
      --data ../data/processed/final_dataset.jsonl \
      --vocab 32000
"""

import argparse
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from tokenizers.processors import TemplateProcessing


def jsonl_text_iterator(jsonl_path: Path) -> Iterator[str]:
    """
    Stream all "text" fields from final_dataset.jsonl.
    This avoids loading everything into RAM at once.
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "")
            if not text:
                continue

            yield text


def train_tokenizer(
    data_path: Path,
    vocab_size: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base BPE model; <UNK> is required
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # Normalization: NFKC + lowercase
    tokenizer.normalizer = Sequence([
        NFKC(),
        Lowercase(),
    ])

    # Simple whitespace splitting is fine for legal text
    tokenizer.pre_tokenizer = Whitespace()

    # Special tokens
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
    )

    print(f"📄 Training BPE tokenizer on: {data_path}")
    print(f"🔢 Target vocab size: {vocab_size}")

    iterator = jsonl_text_iterator(data_path)
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    # Post-processing template: add BOS/EOS automatically for encode()
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> <BOS> $B <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    # Save tokenizer as a single JSON file
    out_path = out_dir / "legal_tokenizer.json"
    tokenizer.save(str(out_path))

    print("🎉 Tokenizer training complete!")
    print(f"💾 Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to final_dataset.jsonl (processed corpus)",
    )
    parser.add_argument(
        "--vocab",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path("../data/tokenizer")

    train_tokenizer(
        data_path=data_path,
        vocab_size=args.vocab,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
