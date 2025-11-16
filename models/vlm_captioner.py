# models/vlm_captioner.py

from typing import List
from PIL import Image

import os
import torch

# 🔍 Debug: show python + torch info
print("Python executable:", os.sys.executable)
print("torch version:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())

# 🔧 Make sure transformers is allowed to use torch
# Some tools / shells set these and they silently disable the backend.
for var in ["TRANSFORMERS_NO_TORCH", "USE_TF", "USE_FLAX"]:
    if var in os.environ:
        print(f"Unsetting {var} =", os.environ[var])
        os.environ.pop(var)

# If USE_TORCH was set to something weird like "0" or "false"
if os.environ.get("USE_TORCH") not in (None, "", "1", "True", "true"):
    print("Fixing USE_TORCH from", os.environ["USE_TORCH"], "to '1'")
    os.environ["USE_TORCH"] = "1"

from transformers.utils import is_torch_available
print("transformers.utils.is_torch_available():", is_torch_available())

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoFeatureExtractor,
)


class VLMCaptioner:
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning"):
        # To simplify debugging, force CPU (you can switch back to cuda later)
        self.device = "cpu"
        # If you really want GPU:
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model + processor + tokenizer
        print("Loading model:", model_name, "on device:", self.device)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
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
            captions.append(caption.strip())

        return captions
