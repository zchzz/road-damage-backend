#!/bin/bash
# Setup script for Road Damage Detection using uv

echo "Road Damage Detection - Environment Setup"
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    
    # Install uv
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        echo "Error: curl is not available. Please install uv manually:"
        echo "Visit: https://github.com/astral-sh/uv"
        exit 1
    fi
    
    # Source the shell profile to make uv available
    source ~/.bashrc 2>/dev/null || source ~/.bash_profile 2>/dev/null || true
fi

echo "✓ uv is available"

# Create virtual environment
echo "Creating virtual environment..."
uv venv .venv

echo "✓ Virtual environment created"

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "✓ Dependencies installed"

echo ""
echo "Setup complete! To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Or run commands directly with uv:"
echo "  uv run python setup_and_demo.py"
echo "  uv run python setup_and_demo.py --local path/to/video.mp4"
echo "  uv run python setup_and_demo.py --youtube"
