#!/usr/bin/env python3
"""
Batch PDF → TXT converter with light cleaning.

- Recursively scans an input directory for .pdf files
- Extracts text using pdfminer.six
- Cleans:
    - normalizes newlines
    - removes pure page-number lines (e.g. "7", "- 7 -")
    - collapses multiple blank lines
- Writes .txt files to a mirrored structure under output dir

Usage:
  python3 pdf_to_txt_converter.py \
      --in_dir /path/to/input_pdfs \
      --out_dir /path/to/output_txt
"""

import argparse
import re
from pathlib import Path

from pdfminer.high_level import extract_text


def clean_text(text: str) -> str:
    """Apply simple generic cleanup to extracted PDF text."""
    # Normalize newlines
    text = text.replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Drop pure page numbers like "7" or "- 7 -"
        if re.fullmatch(r"\d+", stripped):
            continue
        if re.fullmatch(r"-\s*\d+\s*-", stripped):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Collapse 3+ blank lines → 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def convert_pdf(pdf_path: Path, txt_path: Path) -> None:
    """Convert a single PDF to text and save."""
    print(f"🧾 Converting: {pdf_path}")

    try:
        text = extract_text(str(pdf_path))
    except Exception as e:
        print(f"❌ Failed to extract from {pdf_path}: {e}")
        return

    if not text or not text.strip():
        print(f"⚠️ No text extracted from {pdf_path}, skipping.")
        return

    cleaned = clean_text(text)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(cleaned, encoding="utf-8")
    print(f"✅ Saved: {txt_path}")


def main(in_dir: Path, out_dir: Path) -> None:
    if not in_dir.exists():
        raise SystemExit(f"Input dir does not exist: {in_dir}")

    pdf_files = list(in_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"⚠️ No PDFs found under {in_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) under {in_dir}")

    for pdf_path in pdf_files:
        # Mirror directory structure under out_dir
        rel = pdf_path.relative_to(in_dir)
        txt_path = out_dir / rel.with_suffix(".txt")
        convert_pdf(pdf_path, txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert PDFs to TXT.")
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Input directory containing PDFs (recursively scanned).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for TXT files.",
    )

    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    main(in_dir, out_dir)
