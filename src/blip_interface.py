"""
BLIP interface for zero-shot image and text embeddings.
Provides unified access to BLIP models for visual-textual similarity.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Optional
from transformers import BlipProcessor, BlipForImageTextRetrieval


class BLIPInterface:
    """Interface for BLIP model operations."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-base-coco",
        device: Optional[str] = None,
    ):
        """
        Initialize BLIP interface.

        Args:
            model_name: BLIP model variant for image-text retrieval
                       (e.g., "Salesforce/blip-itm-base-coco", "Salesforce/blip-itm-large-coco")
            device: Device to run on ("cuda", "cpu", or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading BLIP model: {model_name} on {self.device}")
        self.model_name = model_name
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def encode_image(self, image_path: str, normalize: bool = True) -> np.ndarray:
        """
        Encode image to BLIP embedding.

        Args:
            image_path: Path to image file
            normalize: If True, normalize embedding to unit length

        Returns:
            Image embedding as numpy array
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use vision model to get image embeddings
            vision_outputs = self.model.vision_model(
                pixel_values=inputs.pixel_values, return_dict=True
            )
            # Get the [CLS] token representation
            image_features = vision_outputs[0][:, 0, :]

            if normalize:
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

        return image_features.cpu().numpy().squeeze()

    def encode_text(
        self, text: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text to BLIP embedding.

        Args:
            text: Single text string or list of texts
            normalize: If True, normalize embeddings to unit length

        Returns:
            Text embedding(s) as numpy array
        """
        if isinstance(text, str):
            text = [text]

        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            # Use text encoder to get text embeddings
            text_outputs = self.model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )
            # Get the [CLS] token representation
            text_features = text_outputs[0][:, 0, :]

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
        similarities = np.dot(image_embedding, text_embeddings.T)

        return similarities.squeeze()

    def encode_knowledge_base(
        self, knowledge_base: Dict[str, Dict], prefix: str = ""
    ) -> Dict[str, np.ndarray]:
        """
        Encode knowledge base entries with BLIP.

        Args:
            knowledge_base: Dictionary mapping species to their data
            prefix: Optional prefix for text encoding

        Returns:
            Dictionary mapping species to text embeddings
        """
        blip_kb = {}

        for species, data in knowledge_base.items():
            # Get VRS (Visual Recognition String) or description
            text = data.get("vrs", data.get("description", ""))

            if text:
                # Add prefix if provided
                full_text = prefix + text if prefix else text

                # Encode with BLIP
                embedding = self.encode_text(full_text, normalize=True)
                blip_kb[species] = embedding

        return blip_kb

    def get_image_text_similarity(
        self, image_path: str, texts: List[str]
    ) -> np.ndarray:
        """
        Compute similarity between an image and multiple texts.

        Args:
            image_path: Path to image
            texts: List of text descriptions

        Returns:
            Array of similarity scores
        """
        image_embedding = self.encode_image(image_path, normalize=True)
        text_embeddings = self.encode_text(texts, normalize=True)

        return self.compute_similarity(image_embedding, text_embeddings)
