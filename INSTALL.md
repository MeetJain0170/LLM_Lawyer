# Installation Guide - LLM Lawyer

This guide explains how to install all dependencies for the LLM Lawyer project.

## Quick Start

```bash
# 1. Navigate to project
cd ~/LLM_Lawyer

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install base requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Requirements Files Explained

### 1. `requirements.txt` (Required)
**Base dependencies for all functionality**
- Core ML libraries (PyTorch, NumPy)
- Tokenization (HuggingFace tokenizers)
- Web framework (Flask)
- Scraping tools (requests, BeautifulSoup)
- PDF processing (pdfminer)
- Google AI (Gemini API)

**Install:**
```bash
pip install -r requirements.txt
```

### 2. `requirements-gpu.txt` (Optional)
**GPU acceleration and optimization**
- Flash Attention (faster training)
- CUDA-enabled PyTorch versions

**Install (after base requirements):**
```bash
# First, install PyTorch with CUDA from pytorch.org
# Then:
pip install -r requirements-gpu.txt
```

**Note:** Flash Attention requires specific CUDA setup. See [Flash Attention Installation](#flash-attention-installation) below.

### 3. `requirements-dev.txt` (Optional)
**Development tools and testing**
- Testing framework (pytest)
- Code formatting (black, flake8)
- Type checking (mypy)
- Jupyter notebooks

**Install:**
```bash
pip install -r requirements-dev.txt
```

## Installation Scenarios

### Scenario 1: CPU Only (Basic Setup)
```bash
pip install -r requirements.txt
```

### Scenario 2: GPU with CUDA (Recommended for Training)
```bash
# Step 1: Install PyTorch with CUDA
# Visit https://pytorch.org/get-started/locally/
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install base requirements
pip install -r requirements.txt

# Step 3: Install Flash Attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

### Scenario 3: Full Development Setup
```bash
# Install base requirements
pip install -r requirements.txt

# Install GPU optimizations (if available)
pip install -r requirements-gpu.txt

# Install development tools
pip install -r requirements-dev.txt
```

## Flash Attention Installation

Flash Attention significantly speeds up training but requires specific setup:

### Prerequisites
- CUDA 11.8 or higher
- PyTorch with CUDA support
- GCC compiler
- Ninja build system

### Installation Steps

```bash
# 1. Install build dependencies
sudo apt update
sudo apt install -y build-essential ninja-build

# 2. Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install Flash Attention
pip install flash-attn --no-build-isolation

# Or with specific version:
pip install flash-attn==2.5.6 --no-build-isolation
```

### Verify Flash Attention
```bash
python3 -c "from flash_attn.modules.mha import MHA; print('Flash Attention: OK')"
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional
PROXIES=http://user:pass@host:port
```

Load environment variables:
```bash
export $(cat .env | xargs)
```

## Verification

After installation, verify everything works:

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check key packages
python3 -c "import tokenizers; print('Tokenizers: OK')"
python3 -c "import flask; print('Flask: OK')"
python3 -c "import google.generativeai; print('Google AI: OK')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import requests; print('Requests: OK')"
python3 -c "import bs4; print('BeautifulSoup: OK')"
python3 -c "from pdfminer.high_level import extract_text; print('PDF Miner: OK')"

# Check Flash Attention (if installed)
python3 -c "from flash_attn.modules.mha import MHA; print('Flash Attention: OK')" 2>/dev/null || echo "Flash Attention: Not installed"
```

## Troubleshooting

### Issue: PyTorch CUDA not available
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Flash Attention build fails
```bash
# Install build dependencies
sudo apt install -y build-essential ninja-build

# Try with specific version
pip install flash-attn==2.5.6 --no-build-isolation
```

### Issue: Package conflicts
```bash
# Create fresh virtual environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Out of memory during installation
```bash
# Install packages one by one
pip install torch
pip install tokenizers
pip install flask
# ... etc
```

## System Requirements

### Minimum (CPU Only)
- Python 3.8+
- 8GB RAM
- 20GB disk space

### Recommended (GPU Training)
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8 or 12.1
- 50GB+ disk space

## Package Versions

Key package versions (as of latest update):
- PyTorch: >=2.0.0
- NumPy: >=1.24.0
- Tokenizers: >=0.15.0
- Flask: >=2.3.0
- Google Generative AI: >=0.3.0

For specific version requirements, see `requirements.txt`.

## Next Steps

After installation:
1. Set up environment variables (`.env` file)
2. Read `SETUP_WSL.md` for WSL-specific setup
3. Run `start_environment.sh` to boot into the environment
4. Start with dataset building or training

## Support

For issues:
1. Check `SETUP_WSL.md` for WSL-specific problems
2. Verify all dependencies are installed correctly
3. Check Python and CUDA versions match requirements
4. Review error messages for specific package issues

