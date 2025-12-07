#!/bin/bash

# WildMatch Pipeline Launcher
# Allows running original WildMatch, structured WildMatch, or comparison

echo "======================================================================"
echo "WildMatch Pipeline Launcher"
echo "======================================================================"
echo ""
echo "Select an option:"
echo "  1) Run Original WildMatch (text-based VLM + LLM matching)"
echo "  2) Run Structured WildMatch (JSON attributes + similarity matching)"
echo "  3) Compare both approaches side-by-side"
echo "  4) Exit"
echo ""
read -p "Enter option (1-4): " option

case $option in
    1)
        echo ""
        echo "Running Original WildMatch..."
        python3 main.py
        ;;
    2)
        echo ""
        echo "Running Structured WildMatch..."
        python3 main_structured.py
        ;;
    3)
        echo ""
        echo "Running Comparison Experiment..."
        python3 scripts/compare_approaches.py
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please run again and select 1-4."
        exit 1
        ;;
esac
