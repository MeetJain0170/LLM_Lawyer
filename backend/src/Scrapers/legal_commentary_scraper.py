#!/usr/bin/env python3
"""
LEGAL COMMENTARY SCRAPER
-------------------------
Downloads open-access legal commentary resources:
- NALSA Legal Literacy Handbook
- PRS Bill Summaries (selected criminal law bills)
- CHRI Legal Guides
- Judicial Academy Modules (open PDFs)

Then converts all PDFs to plaintext for dataset usage.

"""

import argparse
import os
from pathlib import Path
import requests
from pdfminer.high_level import extract_text

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MeetLegalBot/1.0)"
}

RESOURCES = [
    {
        "name": "Legal_Literacy_Handbook",
        "url": "https://cdnbbsr.s3waas.gov.in/s3f8df2e15374e3dc37766e59ac494f0fd/uploads/2025/04/20250423609862318.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "UNODC_Criminal_Justice_India",
        "url": "https://www.unodc.org/documents/india//publications/compendium/Compendium-India.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "NCW_Womens_Legal_Rights",
        "url": "https://ncw.nic.in/sites/default/files/LegalRightsHandbook.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "NHRC_Know_Your_Rights",
        "url": "https://nhrc.nic.in/sites/default/files/Know_Your_Rights.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "Delhi_Criminal_Justice_Flow",
        "url": "https://delhi.gov.in/sites/default/files/2021-09/CRIMINAL%20JUSTICE%20SYSTEM.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "TNUSRL_IPC_Notes",
        "url": "https://tnusrlaw.in/wp-content/uploads/2023/05/IPC-Notes.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "NUALS_CrPC_Module",
        "url": "https://www.nuals.ac.in/images/CrPc%20Module.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
    {
        "name": "NALSA_Arrested_Rights",
        "url": "https://cdnbbsr.s3waas.gov.in/s3595e6bfda3e7dcf4937c2c8d5b97ca71/uploads/2024/04/202404013039178084.pdf",
        "type": "pdf",
        "ext": "pdf",
    },
]

def download(url, dest):
    try:
        print(f"⬇️  Downloading: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Saved: {dest}")
        return True
    except Exception as e:
        print(f"❌ Failed: {url} → {e}")
        return False


def pdf_to_txt(pdf_path, txt_path):
    try:
        print(f"📘 Converting PDF → TXT: {pdf_path.name}")
        text = extract_text(str(pdf_path))
        text = text.replace("\r", "\n")
        txt_path.write_text(text, encoding="utf-8")
        print(f"✅ Saved text: {txt_path}")
    except Exception as e:
        print(f"❌ Failed PDF conversion {pdf_path}: {e}")


def main(outdir: Path):
    raw_dir = outdir / "raw"
    txt_dir = outdir / "txt"

    raw_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    for res in RESOURCES:
        name = res["name"]
        url = res["url"]
        ext = res["ext"]

        raw_path = raw_dir / f"{name}.{ext}"

        if download(url, raw_path):
            if ext == "pdf":
                txt_path = txt_dir / f"{name}.txt"
                pdf_to_txt(raw_path, txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    main(Path(args.out))
