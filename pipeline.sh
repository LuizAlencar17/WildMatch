#!/bin/bash

# WildMatch Experiment Pipeline
# Run experiments across different datasets and image types

echo "======================================================================"
echo "WildMatch Experiment Pipeline"
echo "======================================================================"

# Define datasets and image types
DATASETS=("serengeti" "wcs" "caltech")
IMAGE_TYPES=("full" "cropped")

# ======================================================================
# Experiment 1: Run main.py (WildMatch baseline) on all configurations
# ======================================================================
echo ""
echo "Experiment 1: WildMatch Baseline Pipeline"
echo "----------------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    for image_type in "${IMAGE_TYPES[@]}"; do
        echo ""
        echo "Running: Dataset=$dataset, ImageType=$image_type"
        python3 main.py --dataset "$dataset" --image_type "$image_type"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed for dataset=$dataset, image_type=$image_type"
        else
            echo "SUCCESS: Completed dataset=$dataset, image_type=$image_type"
        fi
    done
done

# ======================================================================
# Experiment 2: Run main_clip_fusion.py on all configurations
# ======================================================================
echo ""
echo "Experiment 2: CLIP-LLM Fusion Pipeline"
echo "----------------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    for image_type in "${IMAGE_TYPES[@]}"; do
        echo ""
        echo "Running: Dataset=$dataset, ImageType=$image_type"
        python3 main_clip_fusion.py --dataset "$dataset" --image_type "$image_type"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed for dataset=$dataset, image_type=$image_type"
        else
            echo "SUCCESS: Completed dataset=$dataset, image_type=$image_type"
        fi
    done
done

# ======================================================================
# Optional: Run specific experiment
# ======================================================================
# Uncomment and modify the lines below to run a specific configuration:
# python3 main.py --dataset serengeti --image_type full
# python3 main_clip_fusion.py --dataset caltech --image_type cropped

echo ""
echo "======================================================================"
echo "All experiments completed!"
echo "Results saved in: results/"
echo "======================================================================"