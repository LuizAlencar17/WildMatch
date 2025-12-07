"""
Structured attribute-based species matching.
Matches images to species based on attribute similarity.
"""

import json
import numpy as np
from typing import Dict, Optional, List
from collections import Counter


class StructuredAttributeMatcher:
    """Match structured attribute descriptions to species using similarity metrics."""

    # Attribute importance weights (can be learned from data later)
    DEFAULT_WEIGHTS = {
        "coat_pattern": 3.0,  # Very distinctive
        "horns_antlers": 3.0,  # Very distinctive
        "coat_color": 2.0,  # Important
        "tail_shape": 2.0,  # Important
        "ear_shape": 2.0,  # Important
        "facial_markings": 2.5,  # Very important
        "body_shape": 1.5,  # Moderately important
        "legs_paws": 2.0,  # Important
        "distinctive_features": 2.5,  # Very important
        "body_size": 1.0,  # Less distinctive (many species similar size)
    }

    def __init__(self, attribute_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the structured attribute matcher.

        Args:
            attribute_weights: Custom weights for each attribute (optional)
        """
        self.weights = attribute_weights if attribute_weights else self.DEFAULT_WEIGHTS

    def compute_attribute_similarity(self, attr1: str, attr2: str) -> float:
        """
        Compute similarity between two attribute values.
        Uses simple string matching (can be enhanced with embeddings).

        Args:
            attr1: First attribute value
            attr2: Second attribute value

        Returns:
            Similarity score between 0 and 1
        """
        if not attr1 or not attr2:
            return 0.0

        # Normalize strings
        attr1 = attr1.lower().strip()
        attr2 = attr2.lower().strip()

        # Exact match
        if attr1 == attr2:
            return 1.0

        # Partial match (either contains the other)
        if attr1 in attr2 or attr2 in attr1:
            return 0.7

        # Check for common words
        words1 = set(attr1.split())
        words2 = set(attr2.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def compute_species_similarity(
        self, image_attributes: Dict[str, str], species_attributes: Dict[str, str]
    ) -> float:
        """
        Compute weighted similarity between image and species attributes.

        Args:
            image_attributes: Attributes extracted from image
            species_attributes: Known attributes of a species

        Returns:
            Weighted similarity score
        """
        total_score = 0.0
        total_weight = 0.0

        for attr_name, weight in self.weights.items():
            if attr_name in image_attributes and attr_name in species_attributes:
                img_val = image_attributes[attr_name]
                species_val = species_attributes[attr_name]

                # Skip if either is unknown
                if (
                    img_val
                    and species_val
                    and img_val.lower() != "unknown"
                    and species_val.lower() != "unknown"
                ):
                    similarity = self.compute_attribute_similarity(img_val, species_val)
                    total_score += similarity * weight
                    total_weight += weight

        # Normalize by total weight
        return total_score / total_weight if total_weight > 0 else 0.0

    def match_to_species(
        self,
        image_attributes: Dict[str, str],
        knowledge_base: Dict[str, Dict],
        top_k: int = 3,
    ) -> List[tuple]:
        """
        Match image attributes to most similar species.

        Args:
            image_attributes: Attributes extracted from image
            knowledge_base: Dictionary mapping species to their attributes
            top_k: Number of top matches to return

        Returns:
            List of (species_name, similarity_score) tuples, sorted by score
        """
        similarities = []

        for species, data in knowledge_base.items():
            if data and data.get("attributes"):
                species_attrs = data["attributes"]
                similarity = self.compute_species_similarity(
                    image_attributes, species_attrs
                )
                similarities.append((species, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def predict_species(
        self, image_attributes: Dict[str, str], knowledge_base: Dict[str, Dict]
    ) -> Dict:
        """
        Predict species from image attributes.

        Args:
            image_attributes: Attributes extracted from image
            knowledge_base: Dictionary mapping species to their attributes

        Returns:
            Dictionary with prediction and confidence info
        """
        matches = self.match_to_species(image_attributes, knowledge_base, top_k=5)

        if not matches:
            return {
                "prediction": None,
                "confidence": 0.0,
                "top_matches": [],
                "similarity_scores": {},
            }

        # Best match
        best_species, best_score = matches[0]

        # Compute confidence based on score gap
        if len(matches) > 1:
            second_score = matches[1][1]
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
            "similarity_score": best_score,
            "top_matches": matches,
            "image_attributes": image_attributes,
        }

    def predict_with_voting(
        self,
        image_attributes_list: List[Dict[str, str]],
        knowledge_base: Dict[str, Dict],
    ) -> Dict:
        """
        Predict species using majority voting from multiple attribute extractions.

        Args:
            image_attributes_list: List of attribute dictionaries from multiple extractions
            knowledge_base: Dictionary mapping species to their attributes

        Returns:
            Dictionary with prediction and voting info
        """
        predictions = []
        all_results = []

        for attrs in image_attributes_list:
            if attrs:
                result = self.predict_species(attrs, knowledge_base)
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

        # Average similarity score for the winning species
        avg_similarity = np.mean(
            [
                r["similarity_score"]
                for r in all_results
                if r["prediction"] == best_species
            ]
        )

        return {
            "prediction": best_species,
            "confidence": confidence,
            "vote_counts": dict(vote_counts),
            "avg_similarity_score": float(avg_similarity),
            "individual_results": all_results,
        }
