"""
BLIP-only Matcher.
Uses raw BLIP visual similarity for species matching without LLM fusion.
"""

import numpy as np
from typing import Dict, Optional
from .blip_interface import BLIPInterface


class BLIPMatcher:
    """Match species using pure BLIP visual embeddings."""

    def __init__(
        self,
        blip_model: str = "Salesforce/blip-itm-base-coco",
        normalize_scores: bool = True,
        blip_prefix: str = "a camera trap image of an animal. ",
    ):
        """
        Initialize BLIP Matcher.

        Args:
            blip_model: BLIP model variant
            normalize_scores: If True, normalize scores to [0, 1] range
            blip_prefix: Prefix for BLIP text encoding
        """
        self.blip = BLIPInterface(model_name=blip_model)
        self.normalize_scores = normalize_scores
        self.blip_prefix = blip_prefix

    def normalize_score_dict(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Dictionary of species -> score

        Returns:
            Normalized scores
        """
        if not scores:
            return scores

        values = np.array(list(scores.values()))
        min_val = values.min()
        max_val = values.max()

        if max_val - min_val == 0:
            return {k: 1.0 for k in scores.keys()}

        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def compute_scores(
        self, image_path: str, knowledge_base: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Compute visual similarity scores using BLIP.

        Args:
            image_path: Path to image
            knowledge_base: Knowledge base dictionary with species info

        Returns:
            Dictionary of species -> similarity score
        """
        # Encode image with BLIP
        image_embedding = self.blip.encode_image(image_path, normalize=True)

        # Prepare species descriptions
        species_texts = []
        species_names = []

        for species_name, species_info in knowledge_base.items():
            species_names.append(species_name)

            # Create descriptive text from KB
            if isinstance(species_info, dict):
                description = species_info.get("description", species_name)
                summary = species_info.get("summary", "")
                appearance = species_info.get("appearance", "")

                # Combine available info
                full_text = f"{self.blip_prefix}{species_name}. "
                if appearance:
                    full_text += f"Appearance: {appearance} "
                if description:
                    full_text += f"{description}"
            else:
                full_text = f"{self.blip_prefix}{species_name}"

            species_texts.append(full_text)

        # Encode all species descriptions
        species_embeddings = self.blip.encode_text(
            species_texts, normalize=True
        )

        # Handle both single and multiple embeddings
        if species_embeddings.ndim == 1:
            species_embeddings = species_embeddings.reshape(1, -1)

        # Compute similarities
        scores = {}
        for species_name, species_embedding in zip(species_names, species_embeddings):
            similarity = self.blip.compute_similarity(image_embedding, species_embedding)
            scores[species_name] = float(similarity)

        # Normalize if requested
        if self.normalize_scores:
            scores = self.normalize_score_dict(scores)

        return scores

    def match(
        self, image_path: str, knowledge_base: Dict[str, Dict]
    ) -> Dict:
        """
        Perform species matching using BLIP similarity only.

        Args:
            image_path: Path to image
            knowledge_base: Knowledge base dictionary

        Returns:
            Dictionary with prediction, confidence, and scores
        """
        # Compute scores
        scores = self.compute_scores(image_path, knowledge_base)

        # Get top prediction
        if not scores:
            return {
                "prediction": None,
                "confidence": 0.0,
                "visual_scores": {},
            }

        best_species = max(scores, key=scores.get)
        best_score = scores[best_species]

        return {
            "prediction": best_species,
            "confidence": best_score,
            "visual_scores": scores,
        }
