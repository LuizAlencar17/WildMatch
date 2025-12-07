"""
Example script for running predictions on a dataset.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from src.utils import load_json
from src.pipeline import WildMatchPredictor
from src.batch_predict import BatchPredictor


def run_predictions():
    """Run predictions on a dataset."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Load knowledge base
    print("Loading knowledge base...")
    knowledge_base = load_json("data/knowledge_base.json")
    print(f"✓ Loaded {len(knowledge_base)} species")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("data/serengeti/dataset.csv")

    # Configuration
    n_samples = 10  # Change this to run on more samples
    n_captions = 5  # Number of captions per image (for self-consistency)

    print(f"\nRunning predictions on {n_samples} samples...")
    print(f"Using {n_captions} captions per image")

    # Select test subset
    df_test = df.sample(n=n_samples, random_state=42)

    # Create predictor and batch predictor
    predictor = WildMatchPredictor(api_key)
    batch_predictor = BatchPredictor(predictor)

    # Run predictions
    predictions_df = batch_predictor.predict_dataset(
        df=df_test,
        knowledge_base=knowledge_base,
        n_captions=n_captions,
        vlm_model="gpt-4o-mini",
        llm_model="gpt-4o-mini",
        output_path="results/predictions.csv",
    )

    print(f"\n✓ Predictions complete!")
    print(f"Results saved to: results/predictions.csv")

    # Print summary
    print("\nSummary:")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Correct: {predictions_df['correct'].sum()}")
    print(f"Accuracy: {predictions_df['correct'].mean():.2%}")


if __name__ == "__main__":
    run_predictions()
