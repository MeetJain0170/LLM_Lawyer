#!/usr/bin/env python3
"""
FULL DATASET PIPELINE
---------------------

1. Loads all text sources:
   - books
   - grammar/txt
   - legal/commentary/txt
   - constitution
   - acts_json
   - kanoon/cases
   - qa_pairs

2. Cleans the text

3. Chunks into 512–1024 token segments

4. Writes final JSONL dataset

"""

import json
import re
from pathlib import Path
import argparse

# ---------------- CLEANING ---------------- #

def clean_text(text: str) -> str:
    """Basic universal cleaning."""
    text = text.replace("\r", "\n")

    # Remove long repeated whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove standalone page numbers
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        s = line.strip()
        if re.fullmatch(r"\d+", s):
            continue
        if re.fullmatch(r"-\s*\d+\s*-", s):
            continue
        clean_lines.append(line)

    text = "\n".join(clean_lines)
    return text.strip()


# ---------------- CHUNKING ---------------- #

def chunk_text(text: str, max_tokens=512):
    """Chunks by approx tokens (very simple whitespace split)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk_words = words[i:i + max_tokens]
        if len(chunk_words) < 30:
            continue
        chunks.append(" ".join(chunk_words))
    return chunks


# ---------------- LOADING SOURCES ---------------- #

def load_txt_files(root: Path, source_label: str):
    data = []
    for path in root.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, max_tokens=512)

        for i, c in enumerate(chunks):
            data.append({
                "source": source_label,
                "path": str(path),
                "chunk_id": i,
                "text": c
            })
    return data


def load_jsonl_cases(path: Path):
    data = []
    if not path.exists():
        return data

    with path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue

            text = obj.get("full_text") or obj.get("summary") or ""
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned, max_tokens=512)
            for i, c in enumerate(chunks):
                data.append({
                    "source": "case_law",
                    "path": str(path),
                    "chunk_id": i,
                    "text": c
                })
    return data


# ---------------- MAIN PIPELINE ---------------- #

def main(data_root: Path, out_file: Path):
    final_data = []

    print("📘 Loading books...")
    final_data += load_txt_files(data_root / "books", "fiction")

    print("📗 Loading grammar...")
    final_data += load_txt_files(data_root / "grammar/txt", "grammar")

    print("📕 Loading legal commentary...")
    final_data += load_txt_files(data_root / "legal_commentary/txt", "commentary")

    print("📙 Loading constitution...")
    final_data += load_txt_files(data_root / "legal", "constitution")

    print("📘 Loading acts...")
    # acts_json contains Python dictionary-like text but we treat as raw text
    final_data += load_txt_files(data_root / "legal/acts_json", "acts")

    print("📘 Loading Kanoon cases...")
    kanoon_root = data_root / "raw/kanoon"

    # Load master cases first
    master_path = kanoon_root / "cases_master.jsonl"
    if master_path.exists():
        print("📘 Loading cases_master.jsonl...")
        final_data += load_jsonl_cases(master_path)

    # Load category-wise cases (criminal, cybercrime, etc.)
    for category in kanoon_root.iterdir():
        if category.is_dir() and category.name != "cases":
            jsonl = category / f"{category.name}.jsonl"
            if jsonl.exists():
                print(f"📘 Loading {jsonl.name}...")
                final_data += load_jsonl_cases(jsonl)


    print("📘 Loading QnA synthetic pairs...")
    for jsonl in (data_root / "qa_pairs").glob("*.jsonl"):
        with jsonl.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = obj.get("question", "")
                    a = obj.get("answer", "")
                    text = clean_text(q + "\n" + a)
                    chunks = chunk_text(text)
                    for i, c in enumerate(chunks):
                        final_data.append({
                            "source": "qna",
                            "path": str(jsonl),
                            "chunk_id": i,
                            "text": c
                        })
                except:
                    continue

    print(f"✅ Total chunks: {len(final_data)}")
    print(f"💾 Saving JSONL to: {out_file}")

    with out_file.open("w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("🎉 Dataset build complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()
    main(Path(args.data_root), Path(args.out))
