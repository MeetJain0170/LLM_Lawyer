#!/usr/bin/env python3
"""
Grammar & Writing Guides Downloader + TXT Converter

Downloads:
- Write Good (README)
- Technical Writing Handbook (PDF)
- Plain English Campaign PDFs (How to write clearly, Sentence construction, Grammar tips)
- OpenStax Writing Guide (PDF)

Then:
- Saves originals under: <out_dir>/raw/
- Converts PDFs to plain text under: <out_dir>/txt/
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict

import requests
from pdfminer.high_level import extract_text


RESOURCES = [
    {
        "name": "write_good_readme",
        "url": "https://raw.githubusercontent.com/btford/write-good/master/README.md",
        "type": "text",
        "ext": "txt",
    },
    {
        "name": "plain_english_how_to_write_plain_english",
        "url": "https://mstrust.org.uk/sites/default/files/plain_english_campaign_how_to_write_plain.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "writing_tips_plain_language",
        "url": "https://www.rch.org.au/uploadedFiles/Main/Content/ethics/Writing%20Tips.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "plain_english_handbook_sec",
        "url": "https://www.sec.gov/pdf/plainenglishhandbook.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "elements_of_style",
        "url": "https://faculty.harvard.edu/files/writingproject/files/elements.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "oecd_plain_english_guide",
        "url": "https://www.oecd.org/gov/regulatory-policy/44787878.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
]



HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MeetGrammarScraper/1.0)"
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with streaming and basic error handling."""
    print(f"⬇️  Downloading: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with dest_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Saved: {dest_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def save_text(content: str, dest_path: Path) -> None:
    dest_path.write_text(content, encoding="utf-8")
    print(f"✅ Saved text: {dest_path}")


def convert_pdf_to_txt(pdf_path: Path, txt_path: Path) -> bool:
    """Convert a PDF file to plain text using pdfminer.six."""
    try:
        print(f"🧾 Converting PDF → TXT: {pdf_path.name}")
        text = extract_text(str(pdf_path))
        # basic cleanup
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        save_text(text, txt_path)
        return True
    except Exception as e:
        print(f"❌ Failed to convert {pdf_path} to text: {e}")
        return False


def main(out_dir: Path):
    raw_dir = out_dir / "raw"
    txt_dir = out_dir / "txt"

    raw_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    for res in RESOURCES:
        name = res["name"]
        url = res["url"]
        rtype = res["type"]
        ext = res["ext"]

        raw_path = raw_dir / f"{name}.{ext}"

        # 1) Download
        ok = download_file(url, raw_path)
        if not ok:
            continue

        # 2) Convert to TXT if needed
        if rtype == "pdf":
            txt_path = txt_dir / f"{name}.txt"
            convert_pdf_to_txt(raw_path, txt_path)
        elif rtype == "text":
            # For markdown/text, just re-read and save as txt (ensures UTF-8)
            content = raw_path.read_text(encoding="utf-8", errors="ignore")
            txt_path = txt_dir / f"{name}.txt"
            save_text(content, txt_path)
        else:
            print(f"ℹ️ Unknown type {rtype} for {name}, skipping conversion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for grammar resources (e.g. /home/meet/LLM_Lawyer/backend/data/grammar)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    main(out_dir)
