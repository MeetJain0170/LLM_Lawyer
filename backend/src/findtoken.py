import os
from pathlib import Path

# Start looking from the current folder's parent (backend)
ROOT_DIR = Path("..").resolve()

print(f"🕵️  Searching for tokenizer files in: {ROOT_DIR}\n")

found = []

# Walk through all folders
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        # Look for common tokenizer filenames
        if "tokenizer" in file or "vocab" in file or file.endswith(".model"):
            full_path = Path(root) / file
            print(f"✅ FOUND: {full_path}")
            found.append(full_path)

print("\n" + "-"*50)
if len(found) == 0:
    print("❌ NO TOKENIZER FOUND!")
    print("   You might need to download or retrain your tokenizer.")
else:
    print(f"🎉 Found {len(found)} potential tokenizer(s).")
    print("   Update your 'debug_model.py' TOKENIZER_PATH to match one of these.")