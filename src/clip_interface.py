"""
CLIP interface for zero-shot image and text embeddings.
Provides unified access to CLIP models without fine-tuning.
"""

import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Optional


class CLIPInterface:
    """Interface for CLIP model operations."""

    def __init__(self, model_name: str = "ViT-L/14", device: Optional[str] = None):
        """
        Initialize CLIP interface.

        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to run on ("cuda", "cpu", or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_image(self, image_path: str, normalize: bool = True) -> np.ndarray:
        """
        Encode image to CLIP embedding.

        Args:
            image_path: Path to image file
            normalize: If True, normalize embedding to unit length

        Returns:
            Image embedding as numpy array
        """
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

            if normalize:
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

        return image_features.cpu().numpy().squeeze()

    def encode_text(
        self, text: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text to CLIP embedding.

        Args:
            text: Single text string or list of texts
            normalize: If True, normalize embeddings to unit length

        Returns:
            Text embedding(s) as numpy array
        """
        if isinstance(text, str):
            text = [text]

        text_tokens = clip.tokenize(text, truncate=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)

            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embeddings = text_features.cpu().numpy()
        return embeddings.squeeze() if len(text) == 1 else embeddings

    def compute_similarity(
        self, image_embedding: np.ndarray, text_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.

        Args:
            image_embedding: Image embedding (1D array)
            text_embeddings: Text embedding(s) (1D or 2D array)

        Returns:
            Similarity score(s)
        """
        # Ensure correct shapes
        if image_embedding.ndim == 1:
            image_embedding = image_embedding.reshape(1, -1)

        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings.reshape(1, -1)

        # Cosine similarity
        similarity = np.dot(text_embeddings, image_embedding.T).squeeze()
        return similarity

    def encode_knowledge_base(
        self,
        knowledge_base: Dict[str, Dict],
        prefix: str = "camera trap image of an animal. ",
    ) -> Dict[str, np.ndarray]:
        """
        Encode entire knowledge base to CLIP text embeddings.

        Args:
            knowledge_base: Dictionary mapping species to their KB entries
            prefix: Optional prefix for CLIP text encoding

        Returns:
            Dictionary mapping species to CLIP embeddings
        """
        clip_embeddings = {}

        for species, data in knowledge_base.items():
            # Get textual description from KB
            if isinstance(data, dict):
                # Original KB format with VRS
                kb_text = data.get("vrs", data.get("description", ""))
            elif isinstance(data, str):
                kb_text = data
            else:
                print(f"Warning: No text found for {species}")
                continue

            if not kb_text:
                continue

            # Add prefix and encode
            full_text = prefix + kb_text
            embedding = self.encode_text(full_text, normalize=True)
            clip_embeddings[species] = embedding

        print(f"Encoded {len(clip_embeddings)} species to CLIP embeddings")
        return clip_embeddings

    def batch_encode_images(
        self, image_paths: List[str], normalize: bool = True
    ) -> np.ndarray:
        """
        Batch encode multiple images.

        Args:
            image_paths: List of image file paths
            normalize: If True, normalize embeddings

        Returns:
            Array of image embeddings
        """
        embeddings = []
        for img_path in image_paths:
            emb = self.encode_image(img_path, normalize=normalize)
            embeddings.append(emb)

        return np.array(embeddings)
