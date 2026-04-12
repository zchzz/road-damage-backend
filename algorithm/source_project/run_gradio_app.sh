#!/bin/bash

# Road Damage Detection Gradio App Setup and Launch Script
# This script sets up the environment and launches the Gradio web application

set -e  # Exit on any error

echo "🛣️  Road Damage Detection - Gradio App Setup"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "road_damage_detector.py" ]; then
    echo "❌ Error: Please run this script from the road-damage-detection directory"
    exit 1
fi

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment (.venv) not found"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

echo "📦 Activating virtual environment..."
source .venv/bin/activate

echo "🔍 Checking Python version..."
python --version

echo "📋 Installing Gradio and additional dependencies..."
# Install gradio and any missing dependencies
pip install gradio==4.44.0 pandas

echo "🧪 Checking installed packages..."
pip list | grep -E "(gradio|ultralytics|opencv|torch)"

echo ""
echo "🚀 Starting Gradio Application..."
echo "📝 The app will be available at: http://127.0.0.1:7860"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Launch the Gradio app
python gradio_app.py
