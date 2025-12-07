"""
Main entry point for WildMatch pipeline.
Example usage of the WildMatch species classification system.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from src.utils import load_json
from src.knowledge_base import KnowledgeBaseBuilder
from src.pipeline import WildMatchPredictor
from src.batch_predict import BatchPredictor


def main():
    """Main execution function."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("=" * 70)
    print("WildMatch Species Classification Pipeline")
    print("=" * 70)

    # =========================================================================
    # Step 1: Build Knowledge Base (if needed)
    # =========================================================================
    print("\n[1] Loading/Building Knowledge Base...")

    kb_path = "data/knowledge_base.json"

    if os.path.exists(kb_path):
        print(f"✓ Loading existing knowledge base from {kb_path}")
        knowledge_base = load_json(kb_path)
    else:
        print("Building new knowledge base from Wikipedia...")

        # Load dataset to get species list
        df = pd.read_csv("data/serengeti/dataset.csv")
        unique_species = df["species_name"].dropna().unique().tolist()

        # Build knowledge base
        kb_builder = KnowledgeBaseBuilder(api_key)
        knowledge_base = kb_builder.build_knowledge_base(
            species_list=unique_species, output_path=kb_path, skip_existing=True
        )

    print(f"✓ Knowledge base loaded: {len(knowledge_base)} species")

    # =========================================================================
    # Step 2: Single Image Prediction (Example)
    # =========================================================================
    print("\n[2] Single Image Prediction Example...")

    # Load dataset
    df = pd.read_csv("data/serengeti/dataset.csv")

    # Select a random sample
    sample = df.sample(n=1, random_state=42).iloc[0]
    test_image = sample["full_path"]
    true_species = sample["species_name"]

    print(f"Image: {test_image}")
    print(f"True species: {true_species}")

    # Create predictor
    predictor = WildMatchPredictor(api_key)

    # Make prediction
    result = predictor.predict(
        image_path=test_image,
        knowledge_base=knowledge_base,
        n_captions=5,
        vlm_model="gpt-4o-mini",
        llm_model="gpt-4o-mini",
        verbose=True,
    )

    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Correct: {result['prediction'] == true_species}")

    # =========================================================================
    # Step 3: Batch Prediction (Optional)
    # =========================================================================
    print("\n[3] Batch Prediction (Optional)...")

    run_batch = input("Run batch prediction on test set? (y/n): ").lower().strip()

    if run_batch == "y":
        # Select test subset
        n_samples = int(input("Number of samples to test (e.g., 10): "))
        df_test = df.sample(n=n_samples, random_state=42)

        # Create batch predictor
        batch_predictor = BatchPredictor(predictor)

        # Run predictions
        predictions_df = batch_predictor.predict_dataset(
            df=df_test,
            knowledge_base=knowledge_base,
            n_captions=5,
            vlm_model="gpt-4o-mini",
            llm_model="gpt-4o-mini",
            output_path="results/predictions.csv",
        )

        print(f"\n✓ Batch prediction complete!")
        print(f"Results saved to: results/predictions.csv")
    else:
        print("Skipping batch prediction.")

    print("\n" + "=" * 70)
    print("Pipeline execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
