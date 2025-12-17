#!/usr/bin/env python3
"""
=============================================================================
HuggingFace BPE Tokenizer Training Script (Legal Corpus)
=============================================================================

What this file does:
--------------------
This script trains a Byte Pair Encoding (BPE) tokenizer from scratch using
HuggingFace's `tokenizers` library, specifically for a legal text corpus.
The resulting tokenizer is saved as a single JSON file and later used
consistently across pretraining, SFT, and inference.

Why this exists:
----------------
- Legal language has domain-specific vocabulary (sections, articles, clauses)
- Generic tokenizers split legal text poorly
- Training a custom tokenizer improves compression, consistency, and model
  performance while reducing sequence length

Input:
------
- A preprocessed JSONL file where each line contains a "text" field
- The dataset is streamed line-by-line to avoid loading large corpora into RAM

Output:
-------
- A single tokenizer file: legal_tokenizer.json
- Stored under ../data/tokenizer/

Tokenizer design choices:
-------------------------
- Model: BPE (Byte Pair Encoding)
- Normalization: Unicode NFKC + lowercasing for canonical text form
- Pre-tokenization: Simple whitespace (sufficient for structured legal text)
- Special tokens:
    <PAD> : padding
    <UNK> : unknown tokens (required by BPE)
    <BOS> : beginning of sequence
    <EOS> : end of sequence

Post-processing:
----------------
- Automatically wraps encoded text with <BOS> and <EOS>
- Ensures consistency between training and inference pipelines

How it works (high-level):
--------------------------
1. Stream text samples from the JSONL dataset
2. Learn BPE merge rules up to the target vocabulary size
3. Assign IDs to learned tokens and special tokens
4. Attach a post-processing template for BOS/EOS handling
5. Save everything into a single portable JSON file

Assumptions:
------------
- The dataset has already been cleaned and preprocessed
- The "text" field contains the full training string
- This tokenizer will NOT be changed after model training begins

=============================================================================

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
