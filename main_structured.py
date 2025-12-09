"""
Main entry point for Structured WildMatch pipeline.
Example usage of the attribute-based species classification system.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from src.utils import load_json
from src.structured_knowledge_base import StructuredKnowledgeBaseBuilder
from src.structured_pipeline import StructuredWildMatchPredictor
from src.batch_predict import BatchPredictor


def main():
    """Main execution function."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("=" * 70)
    print("Structured WildMatch Species Classification Pipeline")
    print("=" * 70)

    # =========================================================================
    # Step 1: Build Structured Knowledge Base (if needed)
    # =========================================================================
    print("\n[1] Loading/Building Structured Knowledge Base...")

    kb_path = "data/structured_knowledge_base.json"

    if os.path.exists(kb_path):
        print(f"✓ Loading existing structured knowledge base from {kb_path}")
        knowledge_base = load_json(kb_path)
    else:
        raise "Structured knowledge base not found. Please build it first."

    print(f"✓ Structured knowledge base loaded: {len(knowledge_base)} species")

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
    predictor = StructuredWildMatchPredictor(api_key)

    # Make prediction
    result = predictor.predict(
        image_path=test_image,
        knowledge_base=knowledge_base,
        n_samples=5,
        vlm_model="gpt-4o-mini",
        verbose=True,
    )

    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Correct: {result['prediction'] == true_species}")

    # =========================================================================
    # Step 3: Batch Prediction
    # =========================================================================
    print("\n[3] Batch Prediction...")

    # Create batch predictor wrapper
    # We need to adapt the StructuredWildMatchPredictor to work with BatchPredictor
    class StructuredBatchWrapper:
        """Wrapper to make StructuredWildMatchPredictor compatible with BatchPredictor."""

        def __init__(self, predictor):
            self.predictor = predictor

        def predict(
            self,
            image_path,
            knowledge_base,
            n_captions=5,
            vlm_model="gpt-4o-mini",
            llm_model=None,
            verbose=False,
        ):
            """Predict using structured pipeline (n_captions maps to n_samples)."""
            return self.predictor.predict(
                image_path=image_path,
                knowledge_base=knowledge_base,
                n_samples=n_captions,  # Map n_captions to n_samples
                vlm_model=vlm_model,
                verbose=verbose,
            )

    wrapped_predictor = StructuredBatchWrapper(predictor)
    batch_predictor = BatchPredictor(wrapped_predictor)

    # Run predictions
    predictions_df = batch_predictor.predict_dataset(
        df=df,
        knowledge_base=knowledge_base,
        n_captions=5,
        vlm_model="gpt-4o-mini",
        llm_model="gpt-4o-mini",  # Ignored but needed for interface compatibility
        output_path="results/structured_predictions.csv",
    )

    print(f"\n✓ Batch prediction complete!")
    print(f"Results saved to: results/structured_predictions.csv")

    print("\n" + "=" * 70)
    print("Pipeline execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
