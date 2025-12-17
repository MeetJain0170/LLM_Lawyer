#!/usr/bin/env python3
"""
=============================================================================
Legal LLM Flask API Server
=============================================================================

What this file does:
--------------------
This script exposes a locally trained Legal Language Model (LLM) as a REST API
using Flask. It wraps the same inference logic used in `infer_cli.py` and makes
it accessible to a web frontend (or any HTTP client).

High-level flow:
----------------
1. On startup:
   - Loads the tokenizer from disk
   - Loads the fine-tuned GPT checkpoint (SFT)
   - Moves the model to GPU if available, otherwise CPU
   - Sets the model to evaluation mode

2. Runtime:
   - A Flask server listens for incoming requests
   - Requests hit the /chat endpoint with a user message
   - The message is wrapped in a fixed legal prompt
   - The prompt is tokenized and passed to the model
   - Tokens are generated step-by-step using temperature + top-k sampling
   - Generation stops on EOS or forbidden tokens
   - The decoded response is returned as JSON

Thread safety:
--------------
A global threading.Lock is used around model inference to ensure that
concurrent HTTP requests do not execute model.forward() simultaneously,
which can corrupt CUDA state or produce nondeterministic outputs.

Why this exists:
----------------
- Separates model inference from UI concerns
- Enables a browser-based frontend for the Legal LLM
- Makes the model usable via HTTP instead of CLI-only access

Endpoints:
----------
GET  /health  -> Check server + model status
POST /chat    -> Run inference on a legal query

Assumptions:
------------
- Tokenizer and model checkpoint exist on disk
- Prompt format matches training-time assumptions
- This server runs locally (not production-hardened)

=============================================================================
"""
#!/usr/bin/env python3
"""
Legal LLM Hybrid API Server
Integrates Local Knowledge Base + SFT Inference
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

# ---------------------------------------------------------
# 1. IMPORT KNOWLEDGE BASE (The "Expert" Brain)
# ---------------------------------------------------------
try:
    from legal_knowledge import search_knowledge_base
except ImportError:
    print("❌ Critical: legal_knowledge.py not found.")
    sys.exit(1)

# ---------------------------------------------------------
# 2. IMPORT INFERENCE CORE (The "Generative" Brain)
# ---------------------------------------------------------
try:
    from infer_cli import generate
    from model import GPT, GPTConfig
    # Fix for pickle loading issues
    setattr(sys.modules['__main__'], 'GPTConfig', GPTConfig)
except ImportError:
    print("❌ Failed to import infer_cli or model.")
    sys.exit(1)

# ---------------------------------------------------------
# 3. APP CONFIGURATION
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)
lock = threading.Lock()

CHECKPOINT_PATH = Path("../checkpoints/sft/legal_llm_sft_final.pt")
TOKENIZER_PATH  = Path("../data/tokenizer/legal_tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔥 Hardware: {DEVICE}")

# ---------------------------------------------------------
# 4. LOAD AI MODELS
# ---------------------------------------------------------
print(f"📂 Loading Tokenizer from {TOKENIZER_PATH}...")
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

print(f"📂 Loading SFT Model from {CHECKPOINT_PATH}...")
try:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print("✅ Hybrid Legal System Ready.")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 5. API ROUTES
# ---------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "mode": "hybrid-sft", "device": DEVICE})

@app.route("/chat", methods=["POST"])
def chat():
    # force=True handles clients that forget 'Content-Type: application/json'
    data = request.get_json(force=True) 
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    # --- PHASE 1: EXACT MATCH (The "Expert" Check) ---
    # We check our trusted database first for perfect accuracy.
    kb_result = search_knowledge_base(user_query)
    if kb_result:
        response = (
            f"**{kb_result['title']}**\n\n"
            f"{kb_result['content']}\n\n"
            f"_(Verified Legal Database)_"
        )
        return jsonify({"response": response})

    # --- PHASE 2: GENERATIVE FALLBACK (The SFT Model) ---
    # If the DB doesn't know, we let the AI try.
    
    # Format matches your training data: <|user|> ... <|assistant|>
    prompt = f"<|user|>\n{user_query}\n<|assistant|>\n"

    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    with lock, torch.no_grad():
        try:
            out = generate(
                model=model,
                idx=idx,
                tokenizer=tokenizer,
                max_new_tokens=150, # Short limit to reduce hallucinations
            )
        except Exception as e:
            return jsonify({"response": f"AI Error: {str(e)}"}), 500

    full_text = tokenizer.decode(out[0].tolist())
    
    # Extract just the answer part
    if "<|assistant|>" in full_text:
        response = full_text.split("<|assistant|>")[-1].strip()
    else:
        response = full_text[len(prompt):].strip()

    # --- CLEANUP PATCH ---
    # Removes the "confused" artifacts we saw in testing
    response = response.replace("| >", "").replace("<|", "").strip()
    
    # If the model panicked and output nothing or garbage
    if len(response) < 5:
        response = "I am analyzing the legal context. Please try asking about specific topics like 'Theft', 'Murder', or 'Privacy'."

    return jsonify({"response": response})

if __name__ == "__main__":
    # Start the server
    print("🚀 Server running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
