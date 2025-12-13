"""
Main entry point for WildMatch-CLIP-LLM-Fusion pipeline.
Interactive demo of the CLIP-LLM fusion approach.
"""

import os
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

from src.utils import load_json
from pipelines.wildmatch_clip_llm_fusion import WildMatchCLIPLLMFusion


def main():
    """Main execution function."""

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("=" * 70)
    print("WildMatch-CLIP-LLM-Fusion Pipeline")
    print("Zero-shot fusion of visual CLIP and textual LLM scores")
    print("=" * 70)

    # Load knowledge base
    print("\n[1] Loading Knowledge Base...")
    kb_path = "data/knowledge_base.json"

    if not os.path.exists(kb_path):
        print(f"Error: Knowledge base not found at {kb_path}")
        print("Please run the original WildMatch pipeline first to build the KB.")
        return

    knowledge_base = load_json(kb_path)
    print(f"✓ Knowledge base loaded: {len(knowledge_base)} species")

    # Configuration
    print("\n[2] Pipeline Configuration...")
    alpha = 0.4
    clip_model = "ViT-L/14"
    n_captions = 5

    print(f"\nConfiguration:")
    print(f"  Alpha (visual weight): {alpha}")
    print(f"  CLIP model: {clip_model}")
    print(f"  N captions: {n_captions}")

    # Initialize pipeline
    print("\n[3] Initializing Pipeline...")
    print("  Loading CLIP model (this may take a moment)...")

    pipeline = WildMatchCLIPLLMFusion(
        openai_api_key=api_key,
        clip_model=clip_model,
        alpha=alpha,
        normalize_scores=True,
    )
    print("✓ Pipeline ready")

    # Load dataset
    print("\n[4] Loading Dataset...")
    df = pd.read_csv("data/serengeti/dataset.csv")
    print(f"✓ Dataset loaded: {len(df)} images")

    print(f"\n{'='*70}")

    df_test = df.sample(n=2, random_state=42)

    correct = 0
    predictions_list = []

    for idx, row in tqdm(df_test.iterrows()):
        result = pipeline.predict(
            image_path=row["full_path"],
            knowledge_base=knowledge_base,
            n_captions=n_captions,
            vlm_model="gpt-4o-mini",
            verbose=False,
        )

        is_correct = result["prediction"] == row["species_name"]
        if is_correct:
            correct += 1

        # Collect prediction data
        predictions_list.append(
            {
                "image_path": row["full_path"],
                "image_id": os.path.basename(row["full_path"]),
                "true_species": row["species_name"],
                "predicted_species": result["prediction"],
                "confidence": result["confidence"],
                "vote_counts": str(result.get("vote_counts", {})),
                "correct": is_correct,
            }
        )

    accuracy = correct / len(df_test)
    print(f"\n✓ Batch accuracy: {accuracy:.2%}")

    # Save predictions to CSV
    os.makedirs("results", exist_ok=True)
    predictions_df = pd.DataFrame(predictions_list)
    output_path = "results/clip_fusion_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")

    print(f"\n{'='*70}")
    print("Pipeline execution complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
