#!/bin/bash

# Start script for Financial Document RAG System - Streamlit Demo
# This is a simplified version for demos that runs only Streamlit

echo "🚀 Starting Financial Document RAG System (Streamlit Demo)..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. API keys may be missing."
    echo ""
fi

# Activate virtual environment
source venv/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🎨 Starting Streamlit app..."
echo ""
echo "📊 Streamlit UI will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run streamlit_app.py

