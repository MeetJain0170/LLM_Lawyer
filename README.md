# 🏛️ LLM Lawyer   End-to-End Legal Language Model (India)

A **from-scratch, domain-specialized Large Language Model** trained on Indian legal text, statutes, case law, and instruction-style question answering.

This repository implements the **complete Track-B LLM pipeline**   where every token, weight, and behavior originates from this codebase.  
No pretrained foundation models. No LoRA. No shortcut APIs in the learning loop.

Only **data → tokens → weights → reasoning**.

---

## 🧭 Project Scope

This project covers the **entire lifecycle of a GPT-style LLM**:

- Raw legal data collection (scrapers + PDFs)
- Unified corpus construction
- Custom BPE tokenizer training
- GPT pretraining from scratch
- Instruction & chat dataset generation
- Supervised Fine-Tuning (SFT)
- Hybrid inference (verified KB + generative LLM)
- CLI and REST API
- Quantization for deployment
- External evaluation via AI judge

Trained locally on **WSL + NVIDIA RTX 4070**.

---

## 🗺️ High-Level Pipeline

```

Scrapers / PDFs / Q&A
↓
final_dataset.jsonl
↓
Custom BPE Tokenizer
↓
Binary Pretraining Data
↓
GPT Pretraining (FlashAttention)
↓
Instruction / Chat Dataset
↓
Supervised Fine-Tuning (SFT)
↓
Hybrid Inference (KB + LLM)
↓
CLI / REST API / Quantized Model

```

Each stage produces **immutable artifacts** consumed by the next stage.  
Changing any upstream artifact invalidates everything downstream   intentionally.

---

## 📁 Repository Structure

```

LLM_Lawyer/
├── backend/
│   ├── checkpoints/
│   │   ├── base_pretrained/final_pretrained.pt
│   │   └── sft/
│   │       ├── legal_llm_sft_final.pt
│   │       └── legal_llm_quantized.pt
│   │
│   ├── data/
│   │   ├── processed/
│   │   ├── tokenizer/
│   │   ├── pretrain/
│   │   └── instruction/
│   │
│   └── src/
│       ├── model.py
│       ├── hf_train_tokenizer.py
│       ├── pack_pretraining_data.py
│       ├── pretrain.py
│       ├── pack_instruction_data.py
│       ├── sft_train.py
│       ├── infer_cli.py
│       ├── api_server.py
│       └── quantize.py
│──frontend/
|   ├── index.html
|   ├── script.js
|   └── README.md
├── README.md
├── setupguide.txt
└── setuphardware.md


```

---

## 🧠 Data Collection & Corpus Construction

### Sources
- Indian Kanoon-style case law
- IPC, CrPC, Evidence Act, IT Act, Constitution
- Legal commentary and public-domain texts
- Synthetic legal Q&A (Gemini-assisted)
- Gutenberg books to learn english
- Grammar books to learn grammer, preposition, adjectives, verbs, etc.

All of this uploaded on Hugging Face here - https://huggingface.co/datasets/Meet-Jain-0170/LLM_Lawyer_Data
### Corpus Assembly
**Script:**  
`backend/src/pipeline/build_dataset.py`

All sources are cleaned, normalized, deduplicated, and merged into:

```

backend/data/processed/final_dataset.jsonl

````

Each line contains:
```json
{ "text": "Section 420 of the IPC deals with cheating..." }
````

This file is the **single source of truth** for tokenizer training and pretraining.

---

## 🔤 Tokenizer Training (BPE, From Scratch)

**Script:**
`backend/src/hf_train_tokenizer.py`

**Design**

* BPE tokenizer (`tokenizers`)
* Normalization: Unicode NFKC + lowercase
* Pre-tokenizer: whitespace
* Special tokens: `<PAD> <UNK> <BOS> <EOS>`
* BOS/EOS injected via post-processing template

**Output**

```
backend/data/tokenizer/legal_tokenizer.json
```

The tokenizer is the **root dependency** of the entire system.
Changing it invalidates all binary data and checkpoints.

---

## 🧠 GPT Pretraining (From Scratch)

### Binary Packing

**Script:**
`backend/src/pack_pretraining_data.py`

Produces:

```
backend/data/pretrain/
├── train.bin
├── val.bin
└── meta.pkl
```

Flat, contiguous token streams optimized for high-throughput training.

---

### Pretraining

**Script:**
`backend/src/pretrain.py`

**Architecture**

* Decoder-only GPT
* 12 layers, 12 heads, d_model = 768
* Learned positional embeddings
* Pre-LayerNorm blocks
* FlashAttention v2
* Gradient checkpointing
* BF16 mixed precision
* Crash-safe checkpointing with resume

**Output**

```
backend/checkpoints/base_pretrained/final_pretrained.pt
```

This is the **foundation model**.
It learns legal language structure   not instructions.

---

## 🧑‍⚖️ Instruction & Chat Dataset

### Instruction Format

**Script:**
`backend/src/prepare_chat_data.py`

Produces Alpaca-style and chat-style instruction datasets:

```
backend/data/processed/
├── instruction_dataset.jsonl
└── instruction_dataset_chat.jsonl
```

Chat format:

```
<|user|>
What is Section 420 IPC?
<|assistant|>
Section 420 IPC deals with cheating...
```

---

### Instruction Packing

**Script:**
`backend/src/pack_instruction_data.py`

Produces:

```
backend/data/instruction/
├── train.bin
├── labels.bin
├── val.bin
└── meta.pkl
```

Flat binary streams ensure deterministic SFT behavior.

---

## 🎯 Supervised Fine-Tuning (SFT)

**Script:**
`backend/src/sft_train.py`

**Properties**

* Loads pretrained GPT checkpoint
* Architecture-locked compatibility
* Cosine LR decay with warmup
* Gradient accumulation
* Mixed precision (BF16 / FP16)
* Gradient clipping
* Periodic checkpointing

**Output**

```
backend/checkpoints/sft/
├── ckpt_sft_*.pt
└── legal_llm_sft_final.pt
```

SFT teaches **judgment and instruction following** on top of pretrained knowledge.

---

## 💬 Hybrid Inference (CLI)

**Script:**
`backend/src/infer_cli.py`

Inference operates in **two phases**:

1. **Verified Knowledge Base**

   * Exact / keyword match from `legal_knowledge.py`
   * Deterministic, citation-safe answers

2. **Generative Fallback**

   * SFT-trained GPT
   * Low-temperature decoding
   * Context window safety
   * Output cleanup & artifact removal

Run:

```bash
python infer_cli.py
```

---

## 🌐 REST API Server

**Script:**
`backend/src/api_server.py`

**Endpoints**

* `GET /health`   model & device status
* `POST /chat`   hybrid legal inference

**Features**

* Thread-safe inference (CUDA lock)
* Same generation logic as CLI
* Frontend-ready JSON responses

---

## ⚡ Quantization

**Script:**
`backend/src/quantize.py`

* Dynamic INT8 quantization (Linear layers)
* CPU-friendly deployment
* Reduced memory footprint

**Output**

```
backend/checkpoints/sft/legal_llm_quantized.pt
```

---

## 📊 Evaluation

**Script:**
`backend/src/evaluation.py`

* Uses an external LLM as an **impartial judge**
* Compares model answers to expert answers
* Returns structured JSON scores and explanations

This is **evaluation only**   never used in training.

---

## 🧠 Model Architecture (Single Source of Truth)

**File:**
`backend/src/model.py`

Defines:

* `GPTConfig`
* Decoder-only GPT
* FlashAttention-compatible attention
* Checkpoint-stable parameter naming

Any change here invalidates **all checkpoints**.

---

## 🔁 Reproducibility Contract

Given:

* `final_dataset.jsonl`
* `legal_tokenizer.json`
* Training scripts and configs

This pipeline deterministically reproduces:

* Token boundaries
* Binary datasets
* Model architecture
* Checkpoint compatibility
* Inference behavior

---

## 🚀 Future Work

* Larger conversational SFT corpus
* Preference optimization (DPO / RLHF-lite)
* Citation-grounded RAG
* Multilingual Indian law
* Dockerized GPU deployment

---

## 🏁 Summary

**LLM Lawyer** is a complete, end-to-end implementation of a **Track-B Large Language Model**, built the hard way   from raw legal text to a deployable reasoning system.

Nothing here is inherited.
Nothing is abstracted away.
Nothing learns outside this repository.

The project demonstrates:

- How a **custom tokenizer** defines the linguistic universe of a model  
- How **binary-packed corpora** enable stable, high-throughput GPT training  
- How **from-scratch pretraining** builds domain intuition before instruction bias  
- How **supervised fine-tuning** teaches judgment without erasing knowledge  
- How **hybrid inference** balances deterministic law with probabilistic reasoning  

This is not a chatbot pretending to know the law.  
It is a system that **learned legal language step by step**, under explicit constraints.

---

## ⚖️ Intended Use & Disclaimer

This project is:

- A research and learning artifact
- A demonstration of LLM systems engineering
- A legal-domain modeling experiment

It is **not**:

- A substitute for professional legal advice
- A certified legal expert system
- A production-hardened service

Always verify legal conclusions with qualified professionals.

---

## 🧑‍💻 Author

**Meet Jain**  - Primary author and engineer

**Atlas SkillTech University** - TY B.Tech (AI / ML)  

---

## 📌 Closing Note

Large Language Models are not trained.  
They are **constructed**.

This repository documents that construction    
cleanly, explicitly, and without illusion.

If you understand this codebase,  
you don’t just *use* LLMs.

You **own the process**.

