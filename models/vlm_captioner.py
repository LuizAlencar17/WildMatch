# models/vlm_captioner.py

from typing import List
from PIL import Image

import torch
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoImageProcessor,
)


class VLMCaptioner:
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning"):
        # Force CPU to avoid CUDA / driver issues
        self.device = "cpu"

        # Load model + processor + tokenizer
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Generation config
        self.max_length = 64
        self.num_beams = 4

    def _build_prompt(self) -> str:
        # This captioner is not instruction-following; it just generates a caption.
        return ""

    def caption(self, image: Image.Image, num_samples: int = 1) -> List[str]:
        """
        Generate one or more captions for the given image.
        """
        captions: List[str] = []

        # Preprocess image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        for _ in range(num_samples):
            output_ids = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.num_beams,
            )

            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            caption = caption.strip()
            captions.append(caption)

        return captions
