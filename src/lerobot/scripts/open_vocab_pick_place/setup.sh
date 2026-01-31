#!/bin/bash
# Quick setup script for open vocabulary pick-and-place

echo "=========================================="
echo "Open Vocabulary Pick-and-Place Setup"
echo "=========================================="

# Install Python dependencies
echo -e "\n1. Installing Python dependencies..."
pip install transformers pillow opencv-python groq -q

# Check for API keys
echo -e "\n2. Checking API keys..."
if [ -z "$GROQ_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: No LLM API key found!"
    echo "   Please set GROQ_API_KEY or OPENAI_API_KEY"
    echo ""
    echo "   For Groq (free tier):"
    echo "   export GROQ_API_KEY='your_key_here'"
    echo ""
    echo "   Get a free key at: https://console.groq.com/"
else
    if [ ! -z "$GROQ_API_KEY" ]; then
        echo "✓ GROQ_API_KEY found"
    fi
    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo "✓ OPENAI_API_KEY found"
    fi
fi

# Check camera
echo -e "\n3. Checking camera..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✓ Camera devices found:"
    ls -1 /dev/video*
else
    echo "⚠️  No camera devices found"
fi

# Test imports
echo -e "\n4. Testing imports..."
python -c "
import torch
from transformers import OwlViTProcessor
print('✓ PyTorch and Transformers installed')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ All imports successful"
else
    echo "✗ Import test failed"
fi

# Create directories
echo -e "\n5. Creating directories..."
mkdir -p debug_output
mkdir -p calibration
echo "✓ Directories created"

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  1. Test in mock mode:"
echo "     python main.py --mock --debug"
echo ""
echo "  2. Run with real robot:"
echo "     python main.py --robot so100"
echo ""
echo "See README.md for more information."
echo ""
