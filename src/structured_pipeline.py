"""
Structured WildMatch prediction pipeline.
Attribute-based version of the WildMatch approach.
"""

from typing import Dict, Optional
from .structured_vlm import StructuredVisualDescriptionGenerator
from .structured_matcher import StructuredAttributeMatcher


class StructuredWildMatchPredictor:
    """Structured attribute-based WildMatch prediction pipeline."""

    def __init__(
        self, openai_api_key: str, attribute_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the structured WildMatch predictor.

        Args:
            openai_api_key: OpenAI API key
            attribute_weights: Custom weights for attributes (optional)
        """
        self.vlm = StructuredVisualDescriptionGenerator(openai_api_key)
        self.matcher = StructuredAttributeMatcher(openai_api_key, attribute_weights)

    def predict(
        self,
        image_path: str,
        knowledge_base: Dict[str, Dict],
        n_samples: int = 1,
        vlm_model: str = "gpt-4o-mini",
        matcher_model: str = "gpt-4o-mini",
        verbose: bool = False,
    ) -> Dict:
        """
        Structured WildMatch prediction pipeline.

        Args:
            image_path: Path to the image
            knowledge_base: Dictionary mapping species to structured attributes
            n_samples: Number of attribute extractions (for voting)
            vlm_model: Model for visual description generation
            matcher_model: Model for attribute matching
            verbose: If True, print progress messages

        Returns:
            Dictionary with prediction, confidence, and attribute info
        """
        # 1. Generate structured attribute descriptions
        if verbose:
            print(f"Extracting {n_samples} structured attribute description(s)...")

        attributes_list = self.vlm.generate_structured_description(
            image_path, model=vlm_model, num_samples=n_samples
        )

        if not attributes_list or all(a is None for a in attributes_list):
            return {
                "prediction": None,
                "confidence": 0.0,
                "attributes": [],
                "error": "Failed to extract attributes",
            }

        # 2. Match to species
        if verbose:
            print(f"Matching attributes to species using {matcher_model}...")

        if n_samples == 1:
            # Single prediction
            result = self.matcher.predict_species(
                attributes_list[0], knowledge_base, model=matcher_model
            )
            if verbose:
                print(f"\n✓ Prediction: {result['prediction']}")
                print(f"  Similarity score: {result['similarity_score']:.3f}")
                print(f"  Top 3 matches: {result['top_matches'][:3]}")
        else:
            # Multiple predictions with voting
            result = self.matcher.predict_with_voting(
                attributes_list, knowledge_base, model=matcher_model
            )
            if verbose:
                print(f"\n✓ Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Vote counts: {result['vote_counts']}")
                print(f"  Avg similarity: {result['avg_similarity_score']:.3f}")

        return result
