"""
Comparison script for WildMatch Original vs Structured versions.
Runs both approaches and compares results.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List

from src.utils import load_json, save_json
from src.pipeline import WildMatchPredictor
from src.structured_pipeline import StructuredWildMatchPredictor
from src.batch_predict import BatchPredictor


class WildMatchComparison:
    """Compare Original WildMatch vs Structured WildMatch."""

    def __init__(self, openai_api_key: str):
        """
        Initialize comparison framework.

        Args:
            openai_api_key: OpenAI API key
        """
        self.api_key = openai_api_key
        self.original_predictor = WildMatchPredictor(openai_api_key)
        self.structured_predictor = StructuredWildMatchPredictor(openai_api_key)

    def predict_single_image(
        self,
        image_path: str,
        true_species: str,
        original_kb: Dict,
        structured_kb: Dict,
        n_captions: int = 5,
    ) -> Dict:
        """
        Run both predictors on a single image and compare.

        Args:
            image_path: Path to image
            true_species: Ground truth species
            original_kb: Original knowledge base (text-based)
            structured_kb: Structured knowledge base (attributes)
            n_captions: Number of captions/samples

        Returns:
            Dictionary with both predictions and comparison
        """
        print(f"\n{'='*70}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"True species: {true_species}")
        print(f"{'='*70}")

        # Original WildMatch
        print("\n[1] Running Original WildMatch...")
        original_result = self.original_predictor.predict(
            image_path=image_path,
            knowledge_base=original_kb,
            n_captions=n_captions,
            vlm_model="gpt-4o-mini",
            llm_model="gpt-4o-mini",
            verbose=False,
        )

        print(f"  Prediction: {original_result['prediction']}")
        print(f"  Confidence: {original_result['confidence']:.2%}")
        print(f"  Correct: {original_result['prediction'] == true_species}")

        # Structured WildMatch
        print("\n[2] Running Structured WildMatch...")
        structured_result = self.structured_predictor.predict(
            image_path=image_path,
            knowledge_base=structured_kb,
            n_samples=n_captions,
            vlm_model="gpt-4o-mini",
            verbose=False,
        )

        print(f"  Prediction: {structured_result['prediction']}")
        print(f"  Confidence: {structured_result['confidence']:.2%}")
        print(f"  Correct: {structured_result['prediction'] == true_species}")

        # Comparison
        comparison = {
            "image_path": image_path,
            "true_species": true_species,
            "original": {
                "prediction": original_result["prediction"],
                "confidence": original_result["confidence"],
                "correct": original_result["prediction"] == true_species,
                "vote_counts": original_result.get("vote_counts", {}),
            },
            "structured": {
                "prediction": structured_result["prediction"],
                "confidence": structured_result["confidence"],
                "correct": structured_result["prediction"] == true_species,
                "vote_counts": structured_result.get("vote_counts", {}),
            },
            "agreement": original_result["prediction"]
            == structured_result["prediction"],
        }

        print(f"\n  Agreement: {comparison['agreement']}")

        return comparison

    def run_comparison_experiment(
        self,
        df: pd.DataFrame,
        original_kb_path: str,
        structured_kb_path: str,
        n_samples: int = 10,
        n_captions: int = 5,
        output_path: str = "results/comparison_results.json",
        image_col: str = "full_path",
        label_col: str = "species_name",
    ) -> Dict:
        """
        Run comparison experiment on a dataset.

        Args:
            df: DataFrame with images and labels
            original_kb_path: Path to original knowledge base
            structured_kb_path: Path to structured knowledge base
            n_samples: Number of images to test
            n_captions: Number of captions/samples per image
            output_path: Where to save results
            image_col: Column name for image paths
            label_col: Column name for species labels

        Returns:
            Dictionary with aggregated results
        """
        # Load knowledge bases
        print("Loading knowledge bases...")
        original_kb = load_json(original_kb_path)
        structured_kb = load_json(structured_kb_path)

        print(f"Original KB: {len(original_kb)} species")
        print(f"Structured KB: {len(structured_kb)} species")

        # Select test samples
        df_test = df.sample(n=n_samples, random_state=42)

        print(f"\nRunning comparison on {n_samples} images...")
        print(f"Using {n_captions} captions/samples per image\n")

        # Run comparisons
        results = []
        for idx, row in df_test.iterrows():
            image_path = row[image_col]
            true_species = row[label_col]

            try:
                result = self.predict_single_image(
                    image_path=image_path,
                    true_species=true_species,
                    original_kb=original_kb,
                    structured_kb=structured_kb,
                    n_captions=n_captions,
                )
                results.append(result)

            except Exception as e:
                print(f"\n❌ Error processing {image_path}: {e}")
                continue

        # Aggregate results
        summary = self.aggregate_results(results)

        # Save results
        save_data = {
            "summary": summary,
            "individual_results": results,
            "parameters": {
                "n_samples": n_samples,
                "n_captions": n_captions,
                "original_kb": original_kb_path,
                "structured_kb": structured_kb_path,
            },
        }
        save_json(save_data, output_path)

        # Print summary
        self.print_summary(summary)

        return save_data

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate comparison results."""
        if not results:
            return {}

        n_total = len(results)

        # Original WildMatch stats
        original_correct = sum(1 for r in results if r["original"]["correct"])
        original_accuracy = original_correct / n_total

        # Structured WildMatch stats
        structured_correct = sum(1 for r in results if r["structured"]["correct"])
        structured_accuracy = structured_correct / n_total

        # Agreement stats
        n_agreement = sum(1 for r in results if r["agreement"])
        agreement_rate = n_agreement / n_total

        # Both correct
        both_correct = sum(
            1
            for r in results
            if r["original"]["correct"] and r["structured"]["correct"]
        )

        # Only one correct
        only_original = sum(
            1
            for r in results
            if r["original"]["correct"] and not r["structured"]["correct"]
        )
        only_structured = sum(
            1
            for r in results
            if r["structured"]["correct"] and not r["original"]["correct"]
        )

        return {
            "n_samples": n_total,
            "original_accuracy": original_accuracy,
            "structured_accuracy": structured_accuracy,
            "accuracy_difference": structured_accuracy - original_accuracy,
            "agreement_rate": agreement_rate,
            "both_correct": both_correct,
            "only_original_correct": only_original,
            "only_structured_correct": only_structured,
            "both_incorrect": n_total - both_correct - only_original - only_structured,
        }

    def print_summary(self, summary: Dict):
        """Print comparison summary."""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\nTotal samples: {summary['n_samples']}")
        print(f"\nAccuracy:")
        print(f"  Original WildMatch:    {summary['original_accuracy']:.2%}")
        print(f"  Structured WildMatch:  {summary['structured_accuracy']:.2%}")
        print(f"  Difference:            {summary['accuracy_difference']:+.2%}")
        print(f"\nAgreement rate: {summary['agreement_rate']:.2%}")
        print(f"\nBreakdown:")
        print(f"  Both correct:           {summary['both_correct']}")
        print(f"  Only original correct:  {summary['only_original_correct']}")
        print(f"  Only structured correct: {summary['only_structured_correct']}")
        print(f"  Both incorrect:         {summary['both_incorrect']}")
        print("=" * 70)


def main():
    """Main execution function."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("=" * 70)
    print("WildMatch: Original vs Structured Comparison")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv("data/serengeti/dataset.csv")
    print(f"\nDataset loaded: {len(df)} images")

    # Configuration
    n_samples = int(input("Number of samples to test (e.g., 10): "))
    n_captions = int(input("Number of captions/samples per image (e.g., 5): "))

    # Initialize comparison
    comparison = WildMatchComparison(api_key)

    # Run experiment
    results = comparison.run_comparison_experiment(
        df=df,
        original_kb_path="data/knowledge_base.json",
        structured_kb_path="data/structured_knowledge_base.json",
        n_samples=n_samples,
        n_captions=n_captions,
        output_path="results/comparison_results.json",
    )

    print("\n✓ Comparison complete!")
    print("Results saved to: results/comparison_results.json")


if __name__ == "__main__":
    main()
