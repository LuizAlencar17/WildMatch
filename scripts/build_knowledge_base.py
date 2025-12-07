"""
Example script for building a knowledge base.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from src.knowledge_base import KnowledgeBaseBuilder


def build_kb():
    """Build knowledge base for all species in the dataset."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("data/serengeti/dataset.csv")
    unique_species = df["species_name"].dropna().unique().tolist()

    print(f"Found {len(unique_species)} unique species:")
    print(unique_species)

    # Initialize builder
    kb_builder = KnowledgeBaseBuilder(api_key)

    # Build knowledge base
    knowledge_base = kb_builder.build_knowledge_base(
        species_list=unique_species,
        output_path="data/knowledge_base.json",
        skip_existing=True,
    )

    print("\n✓ Knowledge base building complete!")


if __name__ == "__main__":
    build_kb()
