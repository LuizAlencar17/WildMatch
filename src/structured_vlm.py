"""
Structured visual description generation using Vision-Language Models.
Generates attribute-based descriptions instead of free-form text.
"""

import base64
import json
from typing import Dict, Optional, List
from openai import OpenAI


class StructuredVisualDescriptionGenerator:
    """Generate structured attribute-based visual descriptions of animals using GPT-4 Vision."""

    # Structured prompt for attribute extraction
    STRUCTURED_PROMPT = """Describe the animal in the photo by filling in the following fields with ONLY what is visible in the image. Be precise and concise.

Required attributes:
- body_size: small, medium, or large
- coat_color: dominant color(s) of fur/skin (e.g., brown, gray, black and white)
- coat_pattern: smooth, spotted, striped, patched, solid, mixed
- ear_shape: pointed, rounded, long, short, large, small
- horns_antlers: none, short horns, long horns, curved horns, straight horns, antlers
- tail_shape: long, short, bushy, thin, tufted
- facial_markings: stripes, patches, mask-like, spots, plain, distinctive features
- legs_paws: hooves, paws, long legs, short legs, slim, thick
- body_shape: stocky, slender, robust, compact, elongated
- distinctive_features: any other very distinctive visible features

Return ONLY a valid JSON object with these exact keys. Do not include explanations or additional text."""

    def __init__(self, openai_api_key: str):
        """
        Initialize the structured visual description generator.

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

    def generate_structured_description(
        self, image_path: str, model: str = "gpt-4o-mini", num_samples: int = 1
    ) -> Optional[List[Dict[str, str]]]:
        """
        Generate structured attribute-based description of an animal in an image.

        Args:
            image_path: Path to the image
            model: OpenAI model to use (gpt-4o, gpt-4o-mini)
            num_samples: Number of structured descriptions to generate

        Returns:
            List of attribute dictionaries
        """
        descriptions = []

        for i in range(num_samples):
            try:
                if "gpt-4o" not in model:
                    print(
                        f"⚠ Model {model} doesn't support vision. Use gpt-4o or gpt-4o-mini"
                    )
                    return None

                # Encode image
                base64_image = self.encode_image_to_base64(image_path)

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.STRUCTURED_PROMPT},
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
                    temperature=0.3 if num_samples > 1 else 0.0,
                    response_format={"type": "json_object"},  # Force JSON output
                )

                # Parse JSON response
                content = response.choices[0].message.content.strip()
                attributes = json.loads(content)
                descriptions.append(attributes)

            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON response: {e}")
                print(f"Response was: {content}")
                descriptions.append(None)
            except Exception as e:
                print(f"❌ Error generating structured description: {e}")
                descriptions.append(None)

        return descriptions
