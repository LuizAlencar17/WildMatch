"""
WildMatch-CLIP-LLM-Fusion Pipeline.
Zero-shot fusion of CLIP visual scores and LLM textual scores.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Optional
from src.vlm import VisualDescriptionGenerator
from src.matcher_clip_llm_fusion import CLIPLLMFusionMatcher


class WildMatchCLIPLLMFusion:
    """WildMatch pipeline with CLIP-LLM fusion matching."""

    def __init__(
        self,
        openai_api_key: str,
        clip_model: str = "ViT-L/14",
        alpha: float = 0.4,
        normalize_scores: bool = True,
        clip_prefix: str = "camera trap image of an animal. ",
    ):
        """
        Initialize WildMatch-CLIP-LLM-Fusion pipeline.

        Args:
            openai_api_key: OpenAI API key
            clip_model: CLIP model variant
            alpha: Weight for visual score (visual_score * alpha + textual_score * (1-alpha))
            normalize_scores: If True, normalize scores before fusion
            clip_prefix: Prefix for CLIP text encoding from KB
        """
        self.vlm = VisualDescriptionGenerator(openai_api_key)
        self.matcher = CLIPLLMFusionMatcher(
            openai_api_key=openai_api_key,
            clip_model=clip_model,
            alpha=alpha,
            normalize_scores=normalize_scores,
            clip_prefix=clip_prefix,
        )

    def predict(
        self,
        image_path: str,
        knowledge_base: Dict[str, Dict],
        n_captions: int = 1,
        vlm_model: str = "gpt-4o-mini",
        verbose: bool = False,
    ) -> Dict:
        """
        Run WildMatch-CLIP-LLM-Fusion prediction.

        Args:
            image_path: Path to image
            knowledge_base: Knowledge base dictionary
            n_captions: Number of captions for self-consistency
            vlm_model: VLM model for description generation
            verbose: If True, print progress

        Returns:
            Prediction result dictionary
        """
        if verbose:
            print(f"[WildMatch-CLIP-LLM-Fusion]")
            print(f"  Alpha (visual weight): {self.matcher.alpha}")
            print(f"  CLIP model: {self.matcher.clip.model}")
            print(f"  Generating {n_captions} caption(s)...")

        # 1. Generate image descriptions with VLM
        descriptions = self.vlm.generate_visual_description(
            image_path, model=vlm_model, num_samples=n_captions
        )

        if not descriptions or len(descriptions) == 0:
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": "Failed to generate visual descriptions",
            }

        if verbose and n_captions > 1:
            for i, desc in enumerate(descriptions, 1):
                print(f"    Caption {i}: {desc[:100]}...")

        # 2. Match using CLIP-LLM fusion
        if verbose:
            print(f"  Computing CLIP visual scores...")
            print(f"  Computing textual embedding scores...")
            print(f"  Fusing scores...")

        if n_captions == 1:
            # Single prediction
            result = self.matcher.predict_species(
                image_path=image_path,
                image_description=descriptions[0],
                knowledge_base=knowledge_base,
            )
        else:
            # Self-consistency with voting
            result = self.matcher.predict_with_voting(
                image_path=image_path,
                image_descriptions=descriptions,
                knowledge_base=knowledge_base,
            )

        if verbose:
            print(f"\n  ✓ Prediction: {result['prediction']}")
            print(f"    Confidence: {result['confidence']:.2%}")
            if "final_score" in result:
                print(f"    Final score: {result['final_score']:.3f}")
            if "top_matches" in result:
                print(f"    Top 3: {result['top_matches'][:3]}")

        return result

    def clear_cache(self):
        """Clear matcher's embedding caches."""
        self.matcher.clear_cache()
