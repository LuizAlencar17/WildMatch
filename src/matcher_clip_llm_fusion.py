"""
CLIP-LLM Fusion Matcher.
Combines visual CLIP scores with textual LLM scores for species matching.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from openai import OpenAI

from .clip_interface import CLIPInterface


class CLIPLLMFusionMatcher:
    """Match species using fusion of CLIP visual and LLM textual scores."""

    def __init__(
        self,
        openai_api_key: str,
        clip_model: str = "ViT-L/14",
        alpha: float = 0.4,
        normalize_scores: bool = True,
        clip_prefix: str = "camera trap image of an animal. ",
    ):
        """
        Initialize CLIP-LLM Fusion Matcher.

        Args:
            openai_api_key: OpenAI API key for embeddings
            clip_model: CLIP model variant
            alpha: Weight for visual score (1-alpha for textual)
            normalize_scores: If True, normalize scores before fusion
            clip_prefix: Prefix for CLIP text encoding
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.clip = CLIPInterface(model_name=clip_model)
        self.alpha = alpha
        self.normalize_scores = normalize_scores
        self.clip_prefix = clip_prefix

        # Cache for embeddings
        self.clip_kb_cache = None
        self.text_kb_cache = None

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

    def get_text_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """
        Get OpenAI text embedding.

        Args:
            text: Input text
            model: Embedding model

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(input=text, model=model)
        return np.array(response.data[0].embedding)

    def compute_visual_scores(
        self, image_path: str, clip_kb: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute visual similarity scores using CLIP.

        Args:
            image_path: Path to image
            clip_kb: Dictionary of species -> CLIP text embeddings

        Returns:
            Dictionary of species -> visual score
        """
        # Encode image with CLIP
        image_embedding = self.clip.encode_image(image_path, normalize=True)

        # Compute similarities
        visual_scores = {}
        for species, text_embedding in clip_kb.items():
            similarity = self.clip.compute_similarity(image_embedding, text_embedding)
            visual_scores[species] = float(similarity)

        return visual_scores

    def compute_textual_scores(
        self, image_description: str, text_kb: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute textual similarity scores using embeddings.

        Args:
            image_description: LLM description of image
            text_kb: Dictionary of species -> text embeddings

        Returns:
            Dictionary of species -> textual score
        """
        # Embed image description
        desc_embedding = self.get_text_embedding(image_description)

        # Compute cosine similarities
        textual_scores = {}
        for species, kb_embedding in text_kb.items():
            # Cosine similarity
            similarity = np.dot(desc_embedding, kb_embedding) / (
                np.linalg.norm(desc_embedding) * np.linalg.norm(kb_embedding)
            )
            textual_scores[species] = float(similarity)

        return textual_scores

    def fuse_scores(
        self,
        visual_scores: Dict[str, float],
        textual_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Fuse visual and textual scores.

        Args:
            visual_scores: CLIP-based visual scores
            textual_scores: Embedding-based textual scores

        Returns:
            Fused scores
        """
        # Normalize if requested
        if self.normalize_scores:
            visual_scores = self.normalize_score_dict(visual_scores)
            textual_scores = self.normalize_score_dict(textual_scores)

        # Combine scores
        fused_scores = {}
        all_species = set(visual_scores.keys()) | set(textual_scores.keys())

        for species in all_species:
            v_score = visual_scores.get(species, 0.0)
            t_score = textual_scores.get(species, 0.0)
            fused_scores[species] = self.alpha * v_score + (1 - self.alpha) * t_score

        return fused_scores

    def predict_species(
        self,
        image_path: str,
        image_description: str,
        knowledge_base: Dict[str, Dict],
        top_k: int = 5,
    ) -> Dict:
        """
        Predict species using CLIP-LLM fusion.

        Args:
            image_path: Path to image
            image_description: LLM-generated description
            knowledge_base: Knowledge base with species info
            top_k: Number of top matches to return

        Returns:
            Prediction result dictionary
        """
        # Prepare CLIP KB (cache if not done)
        if self.clip_kb_cache is None:
            self.clip_kb_cache = self.clip.encode_knowledge_base(
                knowledge_base, prefix=self.clip_prefix
            )

        # Prepare text KB (cache if not done)
        if self.text_kb_cache is None:
            self.text_kb_cache = {}
            for species, data in knowledge_base.items():
                kb_text = data.get("vrs", data.get("description", ""))
                if kb_text:
                    self.text_kb_cache[species] = self.get_text_embedding(kb_text)

        # Compute visual scores
        visual_scores = self.compute_visual_scores(image_path, self.clip_kb_cache)

        # Compute textual scores
        textual_scores = self.compute_textual_scores(
            image_description, self.text_kb_cache
        )

        # Fuse scores
        final_scores = self.fuse_scores(visual_scores, textual_scores)

        # Get top matches
        sorted_species = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_matches = sorted_species[:top_k]

        # Best prediction
        best_species, best_score = top_matches[0] if top_matches else (None, 0.0)

        # Compute confidence
        if len(top_matches) > 1:
            second_score = top_matches[1][1]
            confidence = (
                best_score / (best_score + second_score)
                if (best_score + second_score) > 0
                else 0.5
            )
        else:
            confidence = best_score

        return {
            "prediction": best_species,
            "confidence": confidence,
            "final_score": best_score,
            "top_matches": top_matches,
            "visual_scores": visual_scores,
            "textual_scores": textual_scores,
            "image_description": image_description,
        }

    def predict_with_voting(
        self,
        image_path: str,
        image_descriptions: List[str],
        knowledge_base: Dict[str, Dict],
    ) -> Dict:
        """
        Predict with self-consistency (multiple descriptions).

        Args:
            image_path: Path to image
            image_descriptions: List of LLM descriptions
            knowledge_base: Knowledge base

        Returns:
            Voting result dictionary
        """
        predictions = []
        all_results = []

        # Predict for each description
        for desc in image_descriptions:
            result = self.predict_species(image_path, desc, knowledge_base)
            predictions.append(result["prediction"])
            all_results.append(result)

        if not predictions:
            return {
                "prediction": None,
                "confidence": 0.0,
                "vote_counts": {},
                "individual_results": [],
            }

        # Majority voting
        vote_counts = Counter(predictions)
        best_species, best_count = vote_counts.most_common(1)[0]
        confidence = best_count / len(predictions)

        # Average final score for winning species
        avg_score = np.mean(
            [r["final_score"] for r in all_results if r["prediction"] == best_species]
        )

        return {
            "prediction": best_species,
            "confidence": confidence,
            "vote_counts": dict(vote_counts),
            "avg_final_score": float(avg_score),
            "individual_results": all_results,
        }

    def clear_cache(self):
        """Clear embedding caches."""
        self.clip_kb_cache = None
        self.text_kb_cache = None
