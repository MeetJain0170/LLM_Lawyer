import os
import json

folder = "."   # current folder
query = "Punishment for theft"

for fname in os.listdir(folder):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(folder, fname)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = json.dumps(data).lower()

        if query.lower() in text:
            print("FOUND IN:", fname)

    except Exception as e:
        print("Error reading", fname, e)
