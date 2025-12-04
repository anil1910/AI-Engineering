#!/bin/bash
# SQL Agent Launcher for Mac/Linux
# Make executable with: chmod +x run_agent.sh
# Run with: ./run_agent.sh

echo "=========================================="
echo "   SQL Knowledge Base Agent"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

# Check if anthropic package is installed
if ! python3 -c "import anthropic" &> /dev/null; then
    echo "Installing required packages..."
    pip3 install anthropic
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install required packages"
        exit 1
    fi
fi

echo "Starting SQL Agent..."
echo ""
python3 sql_agent.py
