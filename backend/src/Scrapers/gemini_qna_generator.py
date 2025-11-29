#!/usr/bin/env python3
"""
LEGAL SYNTHETIC Q&A GENERATOR (Gemini 2.5 Flash)
Categories:
- Criminal Law
- Property Law
- Domestic Violence
- Cyber Crime
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Set

import google.generativeai as genai


# ------------------------------
# CONFIG
# ------------------------------

MODEL_NAME = "models/gemini-2.5-flash"
PAIRS_PER_CALL = 12
MAX_RETRIES = 5

CATEGORIES = {
    "criminal": "Indian Criminal Law: IPC, CrPC, offences, bail, punishment, defences",
    "property": "Property Law: ownership, transfer, easements, registry, mortgage, disputes",
    "domestic_violence": "Domestic Violence Act 2005: protection, residence orders, rights",
    "cybercrime": "Cyber Crime: IT Act 2000, online fraud, data theft, privacy violations"
}

OUT_FILE = Path("backend/data/qa_pairs/synthetic.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------
# INIT
# ------------------------------
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY missing")
    genai.configure(api_key=api_key)


# ------------------------------
# PROMPT BUILDER
# ------------------------------
def build_prompt(category_desc: str) -> str:
    return f"""
Generate {PAIRS_PER_CALL} high-quality, unique, non-repetitive Indian legal Q&A pairs.

Category: {category_desc}

RULES:
1. Output ONLY valid JSONL lines (one JSON per line).
2. Each JSON must contain:
   - "question"
   - "context"
   - "answer"
3. NO numbering, NO bullet points, NO markdown.
4. Questions must be diverse: scenarios, rights, punishments, bail, procedures.
5. Context must include either:
   - a legal section OR
   - a factual scenario OR
   - a legal explanation (max 3–4 lines)
6. Answers must be 80–200 words, accurate to Indian law.
7. Absolutely NO duplicates.
8. Style: natural, clear, lawyer-like, Indian legal tone.

Example JSONL:
{{"question": "What is Section 420 IPC?", "context": "Section 420 IPC deals with cheating...", "answer": "..."}}
"""



# ------------------------------
# GEMINI CALL + RETRY
# ------------------------------
def call_gemini(prompt: str) -> List[Dict]:
    model = genai.GenerativeModel(MODEL_NAME)

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            lines = [l.strip() for l in response.text.split("\n") if l.strip()]

            results = []
            for line in lines:
                try:
                    results.append(json.loads(line))
                except:
                    continue
            return results

        except Exception as e:
            print(f"⚠️ Gemini error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(2)

    print("❌ FAILED after max retries.")
    return []


# ------------------------------
# SIGNATURE FOR DEDUP
# ------------------------------
def signature(q: str, a: str) -> str:
    return (q.lower().strip() + "||" + a.lower().split(".")[0]).strip()


# ------------------------------
# MAIN GENERATION LOOP
# ------------------------------
def generate_synthetic_qna(total_pairs=10000):
    init_gemini()

    seen: Set[str] = set()
    generated = 0

    with open(OUT_FILE, "a", encoding="utf-8") as f:

        while generated < total_pairs:
            cat = random.choice(list(CATEGORIES.keys()))
            prompt = build_prompt(CATEGORIES[cat])

            batch = call_gemini(prompt)

            for qa in batch:
                q = qa.get("question", "")
                a = qa.get("answer", "")

                if len(q) < 12 or len(a) < 60:
                    continue

                sig = signature(q, a)
                if sig in seen:
                    continue

                seen.add(sig)
                qa["category"] = cat

                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                generated += 1

                print(f"  ✓ {generated}/{total_pairs}", end="\r")

            time.sleep(0.5)

    print(f"\n\n🎉 DONE: {generated} Q&A saved.")
    print(f"📁 File: {OUT_FILE}")


# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    generate_synthetic_qna(total_pairs=10000)
