"""
Example script for single image prediction.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from src.utils import load_json
from src.pipeline import WildMatchPredictor


def predict_single_image(image_path: str = None):
    """Predict species for a single image."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Load knowledge base
    print("Loading knowledge base...")
    knowledge_base = load_json("data/knowledge_base.json")
    print(f"✓ Loaded {len(knowledge_base)} species")

    # Get image path
    if image_path is None:
        # Use a random sample from dataset
        df = pd.read_csv("data/serengeti/dataset.csv")
        sample = df.sample(n=1, random_state=42).iloc[0]
        image_path = sample["full_path"]
        true_species = sample["species_name"]
        print(f"\nUsing random sample from dataset:")
        print(f"Image: {image_path}")
        print(f"True species: {true_species}")
    else:
        print(f"\nImage: {image_path}")
        true_species = None

    # Display image (optional)
    try:
        img = Image.open(image_path)
        print(f"Image size: {img.size}")
    except Exception as e:
        print(f"Could not display image: {e}")

    # Create predictor
    predictor = WildMatchPredictor(api_key)

    # Make prediction
    print("\nGenerating prediction...")
    result = predictor.predict(
        image_path=image_path,
        knowledge_base=knowledge_base,
        n_captions=5,
        vlm_model="gpt-4o-mini",
        llm_model="gpt-4o-mini",
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Predicted species: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Vote distribution: {result['vote_counts']}")

    if true_species:
        print(f"True species: {true_species}")
        print(f"Correct: {result['prediction'] == true_species}")

    print("\nGenerated captions:")
    for i, caption in enumerate(result["captions"], 1):
        if caption:
            print(f"{i}. {caption[:100]}...")


if __name__ == "__main__":
    # You can provide a specific image path or let it choose randomly
    predict_single_image()

    # To predict a specific image:
    # predict_single_image("path/to/your/image.jpg")
