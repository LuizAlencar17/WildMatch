"""
WildMatch-BLIP Pipeline.
Zero-shot classification using pure BLIP visual embeddings.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Optional
from src.matcher_blip import BLIPMatcher


class WildMatchBLIP:
    """WildMatch pipeline with pure BLIP visual matching."""

    def __init__(
        self,
        blip_model: str = "Salesforce/blip-itm-base-coco",
        normalize_scores: bool = True,
        blip_prefix: str = "a camera trap image of an animal. ",
    ):
        """
        Initialize WildMatch-BLIP pipeline.

        Args:
            blip_model: BLIP model variant
            normalize_scores: If True, normalize scores to [0, 1] range
            blip_prefix: Prefix for BLIP text encoding from KB
        """
        self.matcher = BLIPMatcher(
            blip_model=blip_model,
            normalize_scores=normalize_scores,
            blip_prefix=blip_prefix,
        )

    def predict(
        self,
        image_path: str,
        knowledge_base: Dict[str, Dict],
        verbose: bool = False,
    ) -> Dict:
        """
        Run WildMatch-BLIP prediction.

        Args:
            image_path: Path to image
            knowledge_base: Knowledge base dictionary
            verbose: If True, print progress

        Returns:
            Prediction result dictionary with keys:
            - prediction: Predicted species name
            - confidence: Confidence score
            - visual_scores: Dictionary of all species scores
        """
        if verbose:
            print(f"[WildMatch-BLIP]")
            print(f"  BLIP model: {self.matcher.blip.model_name}")
            print(f"  Computing visual similarity scores...")

        # Run pure BLIP matching
        result = self.matcher.match(image_path, knowledge_base)

        if verbose:
            print(f"\n  ✓ Prediction: {result['prediction']}")
            print(f"    Confidence: {result['confidence']:.2%}")

            # Show top 3 predictions
            if result["visual_scores"]:
                sorted_scores = sorted(
                    result["visual_scores"].items(), key=lambda x: x[1], reverse=True
                )
                print(f"    Top 3 matches:")
                for i, (species, score) in enumerate(sorted_scores[:3], 1):
                    print(f"      {i}. {species}: {score:.3f}")

        return result

    def clear_cache(self):
        """No-op for interface compatibility."""
        pass
