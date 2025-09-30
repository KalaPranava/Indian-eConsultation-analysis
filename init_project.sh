#!/bin/bash

# Project Initialization Script for Windows (Batch equivalent below)
# Usage: bash init_project.sh

echo "🚀 Initializing Indian E-Consultation Analysis Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run project setup
echo "⚙️ Running project setup..."
python scripts/setup_project.py

echo "✅ Project initialization completed!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Review config.yaml and update as needed"
echo "3. Copy .env.example to .env and configure"
echo "4. Run: python scripts/create_sample_data.py"
echo "5. Start API: python scripts/serve_api.py"
echo ""
echo "🌐 API will be available at: http://localhost:8000/docs"