"""
Main entry point for WildMatch pipeline.
Example usage of the WildMatch species classification system.
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from src.utils import load_json
from src.knowledge_base import KnowledgeBaseBuilder
from src.pipeline import WildMatchPredictor
from src.batch_predict import BatchPredictor


def main(dataset="serengeti", image_type="full"):
    """Main execution function."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("=" * 70)
    print("WildMatch Species Classification Pipeline")
    print(f"Dataset: {dataset} | Image Type: {image_type}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Build Knowledge Base (if needed)
    # =========================================================================
    print("\n[1] Loading/Building Knowledge Base...")

    kb_path = f"data/{dataset}/knowledge_base.json"

    if os.path.exists(kb_path):
        print(f"✓ Loading existing knowledge base from {kb_path}")
        knowledge_base = load_json(kb_path)
    else:
        print("Building new knowledge base from Wikipedia...")

        # Determine dataset CSV path
        csv_suffix = "_cropped.csv" if image_type == "cropped" else ".csv"
        dataset_csv = f"data/{dataset}/dataset{csv_suffix}"

        # Load dataset to get species list
        df = pd.read_csv(dataset_csv)
        unique_species = df["species_name"].dropna().unique().tolist()

        # Build knowledge base
        kb_builder = KnowledgeBaseBuilder(api_key)
        knowledge_base = kb_builder.build_knowledge_base(
            species_list=unique_species, output_path=kb_path, skip_existing=True
        )

    print(f"✓ Knowledge base loaded: {len(knowledge_base)} species")

    # Load dataset
    csv_suffix = "_cropped.csv" if image_type == "cropped" else ".csv"
    dataset_csv = f"data/{dataset}/dataset{csv_suffix}"
    df = pd.read_csv(dataset_csv)

    # Create predictor
    predictor = WildMatchPredictor(api_key)
    batch_predictor = BatchPredictor(predictor)

    # =========================================================================
    # Step 2: Batch Prediction
    # =========================================================================
    output_file = f"results/predictions/{dataset}_{image_type}_predictions.csv"
    predictions_df = batch_predictor.predict_dataset(
        df=df,
        knowledge_base=knowledge_base,
        n_captions=3,
        vlm_model="gpt-4o-mini",
        llm_model="gpt-4o-mini",
        output_path=output_file,
    )

    print(f"\n✓ Batch prediction complete!")
    print(f"Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("Pipeline execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WildMatch Species Classification Pipeline"
    )
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
