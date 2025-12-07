"""
Visual description generation using Vision-Language Models.
"""

import base64
import random
from typing import List, Optional
from openai import OpenAI


class VisualDescriptionGenerator:
    """Generate visual descriptions of animals using GPT-4 Vision."""

    # Instruction prompts from the WildMatch paper (Section 5.2)
    INSTRUCTIONS = [
        "Give a very detailed visual description of the animal in the photo.",
        "Describe in detail the visible body parts of the animal in the photo.",
        "What are the visual characteristics of the animal in the photo?",
        "Describe the appearance of the animal in the photo.",
        "What are the identifying characteristics of the animal visible in the photo?",
        "How would you describe the animal in the photo?",
        "What does the animal in the photo look like?",
    ]

    def __init__(self, openai_api_key: str):
        """
        Initialize the visual description generator.

        Args:
            openai_api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=openai_api_key)

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encode image to base64 for OpenAI API.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_visual_description(
        self, image_path: str, model: str = "gpt-4o-mini", num_samples: int = 1
    ) -> Optional[List[str]]:
        """
        Generate detailed visual description of an animal in an image.
        Uses prompts similar to those in the WildMatch paper (Section 5.2).

        Args:
            image_path: Path to the image
            model: OpenAI model to use (gpt-4o, gpt-4o-mini)
            num_samples: Number of caption samples to generate

        Returns:
            List of caption strings
        """
        captions = []

        for i in range(num_samples):
            # Randomly select an instruction
            instruction = random.choice(self.INSTRUCTIONS)

            try:
                # For gpt-4o models, we can use vision
                if "gpt-4o" in model:
                    # Encode image
                    base64_image = self.encode_image_to_base64(image_path)

                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": instruction},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=500,
                        temperature=0.7 if num_samples > 1 else 0.0,
                    )

                    caption = response.choices[0].message.content.strip()
                    captions.append(caption)
                else:
                    print(
                        f"⚠ Model {model} doesn't support vision. Use gpt-4o or gpt-4o-mini"
                    )
                    return None

            except Exception as e:
                print(f"❌ Error generating caption: {e}")
                captions.append(None)

        return captions
