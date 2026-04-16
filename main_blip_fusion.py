"""
Main entry point for WildMatch-BLIP-LLM-Fusion pipeline.
Interactive demo of the BLIP-LLM fusion approach.
"""

import os
import argparse
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

from src.utils import load_json
from pipelines.wildmatch_blip_llm_fusion import WildMatchBLIPLLMFusion


def main(dataset="serengeti", image_type="full"):
    """Main execution function."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("=" * 70)
    print("WildMatch-BLIP-LLM-Fusion Pipeline")
    print("Zero-shot fusion of visual BLIP and textual LLM scores")
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
    alpha = 0.7
    blip_model = "Salesforce/blip-itm-base-coco"
    n_captions = 3

    print(f"\nConfiguration:")
    print(f"  Alpha (visual weight): {alpha}")
    print(f"  BLIP model: {blip_model}")
    print(f"  N captions: {n_captions}")

    # Initialize pipeline
    print("\n[3] Initializing Pipeline...")
    print("  Loading BLIP model (this may take a moment)...")

    pipeline = WildMatchBLIPLLMFusion(
        openai_api_key=api_key,
        blip_model=blip_model,
        alpha=alpha,
        normalize_scores=True,
    )
    print("✓ Pipeline ready")

    # Load dataset
    print("\n[4] Loading Dataset...")
    csv_suffix = "_cropped.csv" if image_type == "cropped" else ".csv"
    dataset_csv = f"data/{dataset}/dataset{csv_suffix}"
    df = pd.read_csv(dataset_csv).sample(100).reset_index(drop=True)
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
            n_captions=n_captions,
            vlm_model="gpt-4o-mini",
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
                "vote_counts": str(result.get("vote_counts", {})),
                "correct": is_correct,
            }
        )

    accuracy = correct / len(df_test)
    print(f"\n✓ Batch accuracy: {accuracy:.2%}")

    # Save predictions to CSV
    os.makedirs("results/predictions", exist_ok=True)
    predictions_df = pd.DataFrame(predictions_list)
    output_path = (
        f"results/predictions/{dataset}_{image_type}_blip_fusion_predictions.csv"
    )
    predictions_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")

    print(f"\n{'='*70}")
    print("Pipeline execution complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildMatch-BLIP-LLM-Fusion Pipeline")
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
