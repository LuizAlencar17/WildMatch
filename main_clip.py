"""
Main entry point for WildMatch-CLIP pipeline.
Pure CLIP visual similarity for zero-shot species classification.
"""

import os
import argparse
from tqdm import tqdm
import pandas as pd

from src.utils import load_json
from pipelines.wildmatch_clip import WildMatchCLIP


def main(dataset="serengeti", image_type="full"):
    """Main execution function."""

    print("=" * 70)
    print("WildMatch-CLIP Pipeline")
    print("Zero-shot classification using pure CLIP visual embeddings")
    print(f"Dataset: {dataset} | Image Type: {image_type}")
    print("=" * 70)

    # Load knowledge base
    print("\n[1] Loading Knowledge Base...")
    kb_path = f"data/{dataset}/knowledge_base.json"

    if not os.path.exists(kb_path):
        print(f"Error: Knowledge base not found at {kb_path}")
        print("Please run the original WildMatch pipeline first to build the KB.")
        return

    knowledge_base = load_json(kb_path)
    print(f"✓ Knowledge base loaded: {len(knowledge_base)} species")

    # Configuration
    print("\n[2] Pipeline Configuration...")
    clip_model = "ViT-L/14"
    clip_prefix = "camera trap image of an animal. "

    print(f"\nConfiguration:")
    print(f"  CLIP model: {clip_model}")
    print(f"  Score normalization: Enabled")

    # Initialize pipeline
    print("\n[3] Initializing Pipeline...")
    print("  Loading CLIP model (this may take a moment)...")

    pipeline = WildMatchCLIP(
        clip_model=clip_model,
        normalize_scores=True,
        clip_prefix=clip_prefix,
    )
    print("✓ Pipeline ready")

    # Load dataset
    print("\n[4] Loading Dataset...")
    csv_suffix = "_cropped.csv" if image_type == "cropped" else ".csv"
    dataset_csv = f"data/{dataset}/dataset{csv_suffix}"
    df = pd.read_csv(dataset_csv)
    print(f"✓ Dataset loaded: {len(df)} images")

    print(f"\n{'='*70}")

    df_test = df

    correct = 0
    predictions_list = []

    for idx, row in tqdm(
        df_test.iterrows(), total=len(df_test), desc="Processing images"
    ):
        result = pipeline.predict(
            image_path=row["full_path"],
            knowledge_base=knowledge_base,
            verbose=False,
        )

        is_correct = result["prediction"] == row["species_name"]
        if is_correct:
            correct += 1

        # Collect prediction data
        predictions_list.append(
            {
                "image_path": row["full_path"],
                "image_id": os.path.basename(row["full_path"]),
                "true_species": row["species_name"],
                "predicted_species": result["prediction"],
                "confidence": result["confidence"],
                "visual_scores": str(result.get("visual_scores", {})),
                "correct": is_correct,
            }
        )

    accuracy = correct / len(df_test)
    print(f"\n✓ Batch accuracy: {accuracy:.2%}")

    # Save predictions to CSV
    os.makedirs("results", exist_ok=True)
    predictions_df = pd.DataFrame(predictions_list)
    output_path = f"results/predictions/{dataset}_{image_type}_clip_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")

    print(f"\n{'='*70}")
    print("Pipeline execution complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildMatch-CLIP Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="serengeti",
        choices=["serengeti", "wcs", "caltech"],
        help="Dataset to use (serengeti, wcs, or caltech)",
    )
    parser.add_argument(
        "--image_type",
        type=str,
        default="full",
        choices=["full", "cropped"],
        help="Image type to use (full or cropped)",
    )

    args = parser.parse_args()
    main(dataset=args.dataset, image_type=args.image_type)
