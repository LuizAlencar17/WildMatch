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

    # Load dataset
    df = pd.read_csv("data/serengeti/dataset.csv")

    # Create predictor
    predictor = WildMatchPredictor(api_key)

    # =========================================================================
    # Step 2: Batch Prediction
    # =========================================================================
    print("\n[2] Batch Prediction...")

    # Create batch predictor
    batch_predictor = BatchPredictor(predictor)

    # Run predictions
    predictions_df = batch_predictor.predict_dataset(
        # df=df,
        df=df.sample(n=100, random_state=42),
        knowledge_base=knowledge_base,
        n_captions=3,
        vlm_model="gpt-4o-mini",
        llm_model="gpt-4o-mini",
        output_path="results/predictions.csv",
    )

    print(f"\n✓ Batch prediction complete!")
    print(f"Results saved to: results/predictions.csv")

    print("\n" + "=" * 70)
    print("Pipeline execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
