#!/bin/bash

echo "🤟 ASL Sign Language Translator - Web App"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q flask flask-cors

# Check if model exists
if [ ! -f "model.pickle" ]; then
    echo "⚠️  Warning: model.pickle not found!"
    echo "Please train the model first:"
    echo "  python scripts/train_classifier.py"
    echo ""
fi

# Start Flask app
echo ""
echo "🚀 Starting web server..."
echo "📡 Open your browser at: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
