"""
WildMatch prediction pipeline.
Complete implementation of the WildMatch species classification approach.
"""

from typing import Dict, Optional
from collections import Counter

from .vlm import VisualDescriptionGenerator
from .matcher import SpeciesMatcher


class WildMatchPredictor:
    """Complete WildMatch prediction pipeline with self-consistency."""

    def __init__(self, openai_api_key: str):
        """
        Initialize the WildMatch predictor.

        Args:
            openai_api_key: OpenAI API key
        """
        self.vlm = VisualDescriptionGenerator(openai_api_key)
        self.matcher = SpeciesMatcher(openai_api_key)

    def predict(
        self,
        image_path: str,
        knowledge_base: Dict[str, Dict],
        n_captions: int = 5,
        vlm_model: str = "gpt-4o-mini",
        llm_model: str = "gpt-4o-mini",
        verbose: bool = False,
    ) -> Dict:
        """
        Complete WildMatch prediction pipeline with self-consistency.
        Implements Section 7 of the paper.

        Args:
            image_path: Path to the image
            knowledge_base: Dict mapping species to descriptions
            n_captions: Number of captions to generate for self-consistency
            vlm_model: Model for visual description generation
            llm_model: Model for species matching
            verbose: If True, print progress messages

        Returns:
            Dict with prediction, confidence, and intermediate results
        """
        # 1. Generate N visual descriptions
        if verbose:
            print(f"Generating {n_captions} visual descriptions...")

        captions = self.vlm.generate_visual_description(
            image_path, model=vlm_model, num_samples=n_captions
        )

        if not captions or all(c is None for c in captions):
            return {
                "prediction": None,
                "confidence": 0.0,
                "captions": [],
                "species_votes": [],
                "vote_counts": {},
                "error": "Failed to generate captions",
            }

        # 2. Match each caption to species
        if verbose:
            print(f"Matching captions to species...")

        species_predictions = []
        for i, caption in enumerate(captions):
            if caption:
                pred = self.matcher.match_caption_to_species(
                    caption, knowledge_base, model=llm_model
                )
                species_predictions.append(pred)
                if verbose:
                    print(f"  Caption {i+1}: {pred}")

        # 3. Majority vote (self-consistency)
        vote_counts = Counter(species_predictions)
        best_species, best_count = vote_counts.most_common(1)[0]
        confidence = best_count / n_captions

        result = {
            "prediction": best_species,
            "confidence": confidence,
            "captions": captions,
            "species_votes": species_predictions,
            "vote_counts": dict(vote_counts),
        }

        if verbose:
            print(f"\n✓ Prediction: {best_species} (confidence: {confidence:.2%})")
            print(f"  Vote distribution: {dict(vote_counts)}")

        return result
