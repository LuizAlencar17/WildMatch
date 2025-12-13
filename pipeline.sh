#!/bin/bash

# WildMatch Pipeline Launcher
# Allows running different WildMatch variants

echo "======================================================================"
echo "WildMatch Pipeline Launcher"
echo "======================================================================"
echo ""
echo "Select an option:"
echo "  1) Run Original WildMatch (text-based VLM + LLM matching)"
echo "  2) Run Structured WildMatch (JSON attributes + similarity matching)"
echo "  3) Run CLIP-LLM Fusion (CLIP visual + LLM textual fusion)"
echo "  4) Compare Original vs Structured"
echo "  5) Compare Original vs CLIP-LLM Fusion"
echo "  6) Exit"
echo ""
read -p "Enter option (1-6): " option

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
        echo "Running WildMatch-CLIP-LLM-Fusion..."
        python3 main_clip_fusion.py
        ;;
    4)
        echo ""
        echo "Running Comparison: Original vs Structured..."
        python3 scripts/compare_approaches.py
        ;;
    5)
        echo ""
        echo "Running Comparison: Original vs CLIP-LLM Fusion..."
        python3 experiments/run_clip_llm_fusion.py --mode comparison
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please run again and select 1-6."
        exit 1
        ;;
esac