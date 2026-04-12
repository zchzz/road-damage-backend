#!/bin/bash

# Script to run the Gradio app using the virtual environment

# Navigate to the project directory
cd /home/bagus/github/road-damage-detection

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create a virtual environment first:"
    echo "python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
echo "📦 Checking if gradio is installed..."
if ! python -c "import gradio" 2>/dev/null; then
    echo "❌ Gradio not found. Installing required packages..."
    uv pip install gradio opencv-python numpy pandas pytubefix pillow matplotlib pyyaml ultralytics
fi

echo "🚀 Starting Road Damage Detection Gradio App..."
echo "📍 The app will be available at:"
echo "   Local: http://0.0.0.0:7860"
echo "   A public shareable link will also be provided"
echo "🔄 Press Ctrl+C to stop the app"
echo ""

# Run the Gradio app
python gradio_app.py
