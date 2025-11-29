#!/bin/bash
# LLM Lawyer Environment Startup Script
# Quick boot script for WSL environment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   LLM Lawyer Environment Startup      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}📁 Project directory:${NC} $(pwd)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}🔌 Activating virtual environment...${NC}"
source venv/bin/activate

# Check if requirements are installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Dependencies not installed. Installing...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies verified${NC}"
fi

# Load environment variables
if [ -f .env ]; then
    echo -e "${BLUE}🔐 Loading environment variables...${NC}"
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✅ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    cat > .env << 'EOF'
# Google Gemini API Key (required for Q&A generation)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Proxy settings for scrapers
# PROXIES=http://user:pass@host:port
EOF
    echo -e "${YELLOW}⚠️  Please edit .env file and add your GEMINI_API_KEY${NC}"
fi

# Verify Python and key packages
echo ""
echo -e "${BLUE}🔍 System Information:${NC}"
echo -e "  Python: $(python3 --version)"
echo -e "  Virtual env: $(which python3)"
echo -e "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check GPU availability
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "  GPU: ${GREEN}Available${NC} ($(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null))"
else
    echo -e "  GPU: ${YELLOW}Not available (using CPU)${NC}"
fi

# Check disk space
DISK_USAGE=$(df -h "$SCRIPT_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo -e "  Disk: ${RED}Warning: ${DISK_USAGE}% used${NC}"
else
    echo -e "  Disk: ${GREEN}${DISK_USAGE}% used${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Environment Ready! 🎉               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📚 Quick Commands:${NC}"
echo ""
echo "  Dataset Building:"
echo "    python3 backend/src/pipeline/build_dataset.py --help"
echo ""
echo "  Tokenizer Training:"
echo "    python3 backend/src/hf_train_tokenizer.py --help"
echo ""
echo "  Data Packing:"
echo "    python3 backend/src/pack_pretraining_data.py --help"
echo ""
echo "  Scrapers:"
echo "    python3 backend/src/Scrapers/grammar_scraper.py --out backend/data/grammar"
echo "    python3 backend/src/Scrapers/gemini_qna_generator.py"
echo ""
echo -e "${YELLOW}💡 Tip: Type 'exit' to leave the environment${NC}"
echo ""

# Keep shell open with activated environment
exec $SHELL

