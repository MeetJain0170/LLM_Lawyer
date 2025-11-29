# LLM Lawyer - WSL Environment Setup Guide

This guide will help you set up and boot into the LLM Lawyer development environment on Windows Subsystem for Linux (WSL).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [WSL Setup](#wsl-setup)
3. [Environment Setup](#environment-setup)
4. [Project Structure](#project-structure)
5. [Running the Project](#running-the-project)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Windows Requirements
- Windows 10 version 2004+ or Windows 11
- WSL 2 installed (recommended)
- At least 8GB RAM (16GB+ recommended for training)
- Sufficient disk space (50GB+ recommended)

### Verify WSL Installation
```bash
# In Windows PowerShell (as Administrator)
wsl --list --verbose
```

If WSL is not installed:
```powershell
# Install WSL 2 with Ubuntu
wsl --install -d Ubuntu
```

---

## WSL Setup

### 1. Access WSL Terminal

**Option A: From Windows Terminal**
- Open Windows Terminal
- Click the dropdown arrow next to the `+` tab
- Select "Ubuntu" or your WSL distribution

**Option B: From Start Menu**
- Search for "Ubuntu" in Start Menu
- Click to open

**Option C: From PowerShell**
```powershell
wsl
```

### 2. Navigate to Project Directory

```bash
# Navigate to your project (adjust path if different)
cd ~/LLM_Lawyer

# Verify you're in the right directory
pwd
# Should output: /home/meet/LLM_Lawyer
```

### 3. Update System Packages

```bash
# Update package lists
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential python3-pip python3-venv git curl
```

---

## Environment Setup

### 1. Create Python Virtual Environment

```bash
# Navigate to project root
cd ~/LLM_Lawyer

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv)
```

**Note:** Always activate the virtual environment before working on the project:
```bash
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
# Make sure you're in the project root and venv is activated
cd ~/LLM_Lawyer
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 3. Install PyTorch (GPU Support - Optional)

If you have an NVIDIA GPU and want CUDA support:

```bash
# Check CUDA version (if NVIDIA GPU available)
nvidia-smi

# Install PyTorch with CUDA (adjust version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (default):
# Already included in requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
cd ~/LLM_Lawyer
nano .env
```

Add the following (replace with your actual API key):

```bash
# Google Gemini API Key (required for Q&A generation)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Proxy settings for scrapers (if needed)
# PROXIES=http://user:pass@host:port
```

**Get Gemini API Key:**
1. Visit: https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy and paste into `.env` file

**Load environment variables:**
```bash
# Add to ~/.bashrc or ~/.zshrc for permanent loading
echo 'export $(cat ~/LLM_Lawyer/.env | xargs)' >> ~/.bashrc
source ~/.bashrc

# Or load manually each session:
export $(cat ~/LLM_Lawyer/.env | xargs)
```

### 5. Verify Installation

```bash
# Check Python version (should be 3.8+)
python3 --version

# Check installed packages
pip list

# Verify key packages
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import tokenizers; print('Tokenizers: OK')"
python3 -c "import flask; print('Flask: OK')"
python3 -c "import google.generativeai; print('Google AI: OK')"
```

---

## Project Structure

```
LLM_Lawyer/
├── backend/
│   ├── data/                    # Dataset files
│   │   ├── books/              # Fiction books (txt files)
│   │   ├── grammar/            # Grammar guides
│   │   ├── legal/              # Legal documents
│   │   ├── legal_commentary/   # Legal commentary PDFs
│   │   ├── processed/          # Processed datasets
│   │   ├── qa_pairs/           # Q&A pairs
│   │   ├── raw/                # Raw scraped data
│   │   └── tokenizer/          # Trained tokenizer
│   ├── src/
│   │   ├── pipeline/           # Data processing pipeline
│   │   ├── Scrapers/           # Web scrapers
│   │   ├── reference/          # Reference implementations
│   │   ├── hf_train_tokenizer.py
│   │   ├── pack_pretraining_data.py
│   │   └── training_loop.py
│   └── venv/                   # Backend virtual environment
├── frontend/                    # Frontend (if exists)
├── requirements.txt            # Python dependencies
├── SETUP_WSL.md               # This file
└── venv/                      # Root virtual environment
```

---

## Running the Project

### Quick Start: Boot into Environment

Create a startup script for easy access:

```bash
# Create startup script
cat > ~/start_llm_lawyer.sh << 'EOF'
#!/bin/bash
# LLM Lawyer Environment Startup Script

echo "🚀 Starting LLM Lawyer Environment..."
cd ~/LLM_Lawyer

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "✅ Environment variables loaded"
fi

# Show project status
echo ""
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(python3 --version)"
echo "📦 Virtual environment: $(which python3)"
echo ""
echo "Available commands:"
echo "  - python3 backend/src/pipeline/build_dataset.py --help"
echo "  - python3 backend/src/hf_train_tokenizer.py --help"
echo "  - python3 backend/src/training_loop.py"
echo ""
echo "Environment ready! 🎉"
EOF

# Make executable
chmod +x ~/start_llm_lawyer.sh
```

**Usage:**
```bash
# Run startup script
~/start_llm_lawyer.sh
```

### Common Tasks

#### 1. Build Dataset
```bash
cd ~/LLM_Lawyer
source venv/bin/activate
export $(cat .env | xargs)

python3 backend/src/pipeline/build_dataset.py \
    --data_root backend/data \
    --out backend/data/processed/final_dataset.jsonl
```

#### 2. Train Tokenizer
```bash
cd ~/LLM_Lawyer
source venv/bin/activate

python3 backend/src/hf_train_tokenizer.py \
    --data backend/data/processed/final_dataset.jsonl \
    --vocab 32000
```

#### 3. Pack Pretraining Data
```bash
cd ~/LLM_Lawyer
source venv/bin/activate

python3 backend/src/pack_pretraining_data.py \
    --tokenizer backend/data/tokenizer/legal_tokenizer.json \
    --data backend/data/processed/final_dataset.jsonl \
    --out backend/data/pretrain
```

#### 4. Run Scrapers

**Grammar Scraper:**
```bash
cd ~/LLM_Lawyer
source venv/bin/activate

python3 backend/src/Scrapers/grammar_scraper.py \
    --out backend/data/grammar
```

**Legal Commentary Scraper:**
```bash
python3 backend/src/Scrapers/legal_commentary_scraper.py \
    --out backend/data/legal_commentary
```

**Gemini Q&A Generator:**
```bash
export $(cat .env | xargs)  # Load GEMINI_API_KEY
python3 backend/src/Scrapers/gemini_qna_generator.py
```

**Kanoon Scraper:**
```bash
python3 backend/src/Scrapers/hybrid_kanoon_scraper2.py \
    --output_root backend/data/raw/kanoon \
    --pages 5 \
    --sleep 0.5
```

#### 5. Run Flask Backend (if implemented)
```bash
cd ~/LLM_Lawyer
source venv/bin/activate
export $(cat .env | xargs)

cd backend/src/reference
python3 flask_backend.py
```

---

## Troubleshooting

### Issue: "Command not found" after installing packages

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### Issue: Permission denied errors

**Solution:**
```bash
# Fix file permissions
chmod +x backend/src/*.py
chmod +x backend/src/Scrapers/*.py
chmod +x backend/src/pipeline/*.py
```

### Issue: CUDA/GPU not detected

**Solution:**
```bash
# Check if NVIDIA drivers are installed in WSL
nvidia-smi

# If not available, install CUDA toolkit for WSL
# Follow: https://docs.nvidia.com/cuda/wsl-user-guide/
```

### Issue: Out of memory during training

**Solution:**
```bash
# Reduce batch size in training_loop.py
# Use CPU instead of GPU
# Process data in smaller chunks
```

### Issue: WSL is slow

**Solution:**
```bash
# Store project in WSL filesystem (not Windows filesystem)
# Windows filesystem: /mnt/c/...
# WSL filesystem: ~/ or /home/...

# Move project if needed:
# mv /mnt/c/path/to/LLM_Lawyer ~/LLM_Lawyer
```

### Issue: Environment variables not loading

**Solution:**
```bash
# Check .env file exists
ls -la ~/LLM_Lawyer/.env

# Manually load
export $(cat ~/LLM_Lawyer/.env | xargs)

# Verify
echo $GEMINI_API_KEY
```

### Issue: Port already in use (Flask)

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port in flask_backend.py
```

### Issue: PDF extraction fails

**Solution:**
```bash
# Install additional PDF libraries
sudo apt install -y poppler-utils

# Reinstall pdfminer
pip install --upgrade pdfminer.six
```

---

## Daily Workflow

### Morning Startup Routine

```bash
# 1. Open WSL terminal
wsl

# 2. Navigate to project
cd ~/LLM_Lawyer

# 3. Activate environment
source venv/bin/activate

# 4. Load environment variables
export $(cat .env | xargs)

# 5. Verify everything works
python3 -c "import torch; print('Ready!')"
```

### Create an Alias (Optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# LLM Lawyer shortcuts
alias llm='cd ~/LLM_Lawyer && source venv/bin/activate && export $(cat .env | xargs)'
alias llm-data='cd ~/LLM_Lawyer/backend/data'
alias llm-src='cd ~/LLM_Lawyer/backend/src'
```

Then reload:
```bash
source ~/.bashrc
```

Now you can just type `llm` to boot into the environment!

---

## Additional Resources

- **WSL Documentation:** https://docs.microsoft.com/en-us/windows/wsl/
- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **Google Gemini API:** https://ai.google.dev/
- **HuggingFace Tokenizers:** https://huggingface.co/docs/tokenizers/

---

## Quick Reference Commands

```bash
# Boot into environment
cd ~/LLM_Lawyer && source venv/bin/activate && export $(cat .env | xargs)

# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h

# Check memory
free -h

# Monitor processes
htop  # Install with: sudo apt install htop

# View logs
tail -f backend/data/raw/kanoon/scraper.log
```

---

**Last Updated:** 2024
**WSL Version:** 2.0
**Python Version:** 3.12

