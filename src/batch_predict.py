"""
Batch prediction and evaluation utilities.
"""

import pandas as pd
from typing import Dict
from tqdm import tqdm

from .pipeline import WildMatchPredictor


class BatchPredictor:
    """Run WildMatch predictions on entire datasets."""

    def __init__(self, predictor: WildMatchPredictor):
        """
        Initialize batch predictor.

        Args:
            predictor: WildMatchPredictor instance
        """
        self.predictor = predictor

    def predict_dataset(
        self,
        df: pd.DataFrame,
        knowledge_base: Dict[str, Dict],
        n_captions: int = 5,
        vlm_model: str = "gpt-4o-mini",
        llm_model: str = "gpt-4o-mini",
        output_path: str = "../results/predictions.csv",
        image_col: str = "full_path",
        label_col: str = "species_name",
    ) -> pd.DataFrame:
        """
        Run WildMatch prediction pipeline on entire dataset.

        Args:
            df: DataFrame with image paths and species labels
            knowledge_base: Dict mapping species to descriptions
            n_captions: Number of captions to generate for self-consistency
            vlm_model: Model for visual description generation
            llm_model: Model for species matching
            output_path: Path to save predictions CSV
            image_col: Name of column containing image paths
            label_col: Name of column containing species labels

        Returns:
            DataFrame with predictions and ground truth
        """
        results = []

        print(f"Running predictions on {len(df)} images...")
        print(f"Using {n_captions} captions per image for self-consistency")
        print(f"VLM Model: {vlm_model}, LLM Model: {llm_model}\n")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            image_path = row[image_col]
            true_species = row[label_col]

            try:
                # Run prediction
                prediction_result = self.predictor.predict(
                    image_path=image_path,
                    knowledge_base=knowledge_base,
                    n_captions=n_captions,
                    vlm_model=vlm_model,
                    llm_model=llm_model,
                    verbose=False,
                )

                # Store results
                results.append(
                    {
                        "image_path": image_path,
                        "image_id": row.get("id", idx),
                        "true_species": true_species,
                        "predicted_species": prediction_result["prediction"],
                        "confidence": prediction_result["confidence"],
                        "vote_counts": str(prediction_result["vote_counts"]),
                        "correct": prediction_result["prediction"] == true_species,
                    }
                )

                # Save periodically (every 10 predictions)
                if (idx + 1) % 10 == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(output_path, index=False)
                    print(f"\n✓ Progress saved ({idx + 1}/{len(df)} images)")

            except Exception as e:
                print(f"\n❌ Error predicting image {image_path}: {e}")
                results.append(
                    {
                        "image_path": image_path,
                        "image_id": row.get("id", idx),
                        "true_species": true_species,
                        "predicted_species": None,
                        "confidence": 0.0,
                        "vote_counts": None,
                        "correct": False,
                    }
                )

        # Create DataFrame and save
        predictions_df = pd.DataFrame(results)
        predictions_df.to_csv(output_path, index=False)

        # Print summary
        accuracy = predictions_df["correct"].mean()
        print(f"\n✓ Predictions complete!")
        print(f"✓ Saved to: {output_path}")
        print(f"✓ Overall Accuracy: {accuracy:.2%}")

        return predictions_df
