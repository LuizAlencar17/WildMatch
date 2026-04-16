"""
WildMatch-CLIP Pipeline.
Zero-shot classification using pure CLIP visual embeddings.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Optional
from src.matcher_clip import CLIPMatcher


class WildMatchCLIP:
    """WildMatch pipeline with pure CLIP visual matching."""

    def __init__(
        self,
        clip_model: str = "ViT-L/14",
        normalize_scores: bool = True,
        clip_prefix: str = "camera trap image of an animal. ",
    ):
        """
        Initialize WildMatch-CLIP pipeline.

        Args:
            clip_model: CLIP model variant
            normalize_scores: If True, normalize scores to [0, 1] range
            clip_prefix: Prefix for CLIP text encoding from KB
        """
        self.matcher = CLIPMatcher(
            clip_model=clip_model,
            normalize_scores=normalize_scores,
            clip_prefix=clip_prefix,
        )

    def predict(
        self,
        image_path: str,
        knowledge_base: Dict[str, Dict],
        verbose: bool = False,
    ) -> Dict:
        """
        Run WildMatch-CLIP prediction.

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
            print(f"[WildMatch-CLIP]")
            print(f"  CLIP model: {self.matcher.clip.model}")
            print(f"  Computing visual similarity scores...")

        # Run pure CLIP matching
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
