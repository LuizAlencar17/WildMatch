# %%
import os
import json
import random
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.utils import resample
from uuid import uuid4
from tqdm import tqdm
from typing import Dict, List, Optional
from datetime import datetime

# %% [markdown]
# # 1. Dataset builder

# %%
# Define paths
METADATA_SERENGETI_PATH = (
    "/data/luiz/dataset/serengeti/SnapshotSerengeti_S1-11_v2.1.json"
)
DATA_SERENGETI_PATH = "/data/luiz/dataset/serengeti_images/"
DATA_SERENGETI_SSD_PATH = "/ssd/luiz/dataset/serengeti_images/"
SPECIES_CSV_SERENGETI_PATH = (
    "/data/luiz/dataset/partitions/species-classifier/serengeti"
)
ANIMAL_CSV_SERENGETI_PATH = "/data/luiz/dataset/partitions/species-classifier/serengeti"

# METADATA_SERENGETI_PATH = "C:\\Users\\fabio\\Documents\\Workspace\\my-repos\\WildMatch\\data\\serengeti\\metadata.json"
# DATA_SERENGETI_PATH = "C:\\Users\\fabio\\Documents\\Workspace\\my-repos\\WildMatch\\"

# %% [markdown]
# Utils functions


# %%
def load_json(json_file_path):
    print(f"Loading JSON file from {json_file_path}")
    with open(json_file_path, "r") as f:
        return json.load(f)


def merge_annotations_and_images(data, image_base_path):
    """
    Merge annotations with images data and filter out images that don't exist.

    Args:
        data: Dictionary containing 'annotations' and 'images' lists
        image_base_path: Base path where images are stored

    Returns:
        List of dictionaries with merged data (only for existing images)
    """
    # Create a mapping from image_id to image metadata
    images_dict = {img["id"]: img for img in data["images"]}

    # Create a mapping from image_id to annotation
    annotations_dict = {ann["image_id"]: ann for ann in data["annotations"]}

    merged_data = []
    missing_images = []

    print(f"Total images in metadata: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"\nChecking which images exist on disk...")

    for img in tqdm(data["images"]):
        image_id = img["id"]
        file_name = img["file_name"]

        # Construct full image path
        image_path = os.path.join(image_base_path, file_name)

        # Check if image exists
        if os.path.exists(image_path):
            # Get corresponding annotation if it exists
            annotation = annotations_dict.get(image_id, None)

            # Merge image and annotation data
            merged_item = {
                **img,  # Include all image metadata
                "full_path": image_path,
            }

            # Add annotation data if available
            if annotation:
                merged_item.update(
                    {
                        "annotation_id": annotation["id"],
                        "category_id": annotation["category_id"],
                        "seq_id": annotation["seq_id"],
                        "season": annotation["season"],
                        "subject_id": annotation["subject_id"],
                        "count": annotation["count"],
                        "standing": annotation["standing"],
                        "resting": annotation["resting"],
                        "moving": annotation["moving"],
                        "interacting": annotation["interacting"],
                        "young_present": annotation["young_present"],
                    }
                )
            else:
                merged_item["annotation_id"] = None
                merged_item["category_id"] = None

            merged_data.append(merged_item)
        else:
            missing_images.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "expected_path": image_path,
                }
            )

    print(f"\n✓ Found {len(merged_data)} existing images")
    print(f"✗ Missing {len(missing_images)} images")

    return merged_data, missing_images


# Function to balance the dataset
def balance_dataset(df, category_col, min_samples=100):
    # Count samples per class
    class_counts = df[category_col].value_counts()

    # Target size = smallest class count
    min_size = class_counts.min()
    if min_size > min_samples:
        min_size = min_samples

    balanced_list = []

    # Downsample each class to min_size
    for class_value in class_counts.index:
        df_class = df[df[category_col] == class_value]
        df_downsampled = resample(
            df_class, replace=False, n_samples=min_size, random_state=123
        )
        balanced_list.append(df_downsampled)

    # Combine all classes
    df_balanced = pd.concat(balanced_list).reset_index(drop=True)

    return df_balanced


# Function to copy images to a specified directory
def copy_images_to_directory(df, source_col, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    paths = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Copying images"):
        src = row[source_col]
        paths.append(os.path.join(dest_dir, os.path.basename(src)))
        if os.path.exists(src):
            shutil.copy(src, dest_dir)

    df[source_col] = paths
    return df


# %%
CATEGORIES_TO_REMOVE = [0, 1, 23]

SPECIES_TO_INCLUDE = [
    "elephant",
    "ostrich",
    "zebra",
    "cheetah",
    "hippopotamus",
    "baboon",
    "buffalo",
    "giraffe",
    "warthog",
    "guineafowl",
    "hyenaspotted",
    "impala",
]

CSV_PATH = "../data/serengeti/dataset.csv"

# %%
df_balanced = pd.read_csv(CSV_PATH)
df_balanced

# %% [markdown]
# ## 1.1 Analyze sequence-level data (needed for Section 11 of paper)

# %%
# Group by sequence to see how many frames per sequence
sequence_stats = (
    df_balanced.groupby("seq_id")
    .agg(
        {
            "id": "count",  # Number of frames
            "species_name": "first",  # Species (should be same across sequence)
            "datetime": ["min", "max"],  # Time range
        }
    )
    .round(2)
)

sequence_stats.columns = ["num_frames", "species", "start_time", "end_time"]
sequence_stats = sequence_stats.reset_index()

print("Sequence-level statistics:")
print(f"Total sequences: {len(sequence_stats)}")
print(f"\nFrames per sequence distribution:")
print(sequence_stats["num_frames"].describe())
print(f"\nSequences with multiple frames: {(sequence_stats['num_frames'] > 1).sum()}")

# %%
sequence_stats

# %%


# %% [markdown]
# # 2. Task 1: Build Knowledge Base from Wikipedia
#
# We'll use Wikipedia API to fetch articles and GPT-4 to generate Visually Relevant Summaries (VRS)

# %%
import wikipediaapi
from openai import OpenAI
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="WildMatch/1.0 (fabio@example.com)", language="en"
)


# %%
def fetch_wikipedia_article(species_name):
    """
    Fetch Wikipedia article for a species.
    Extract summary and relevant sections about appearance.

    Args:
        species_name: Name of the species (e.g., 'zebra', 'elephant')

    Returns:
        Dictionary with summary and appearance sections
    """
    # Try to get the page
    page = wiki_wiki.page(species_name)

    if not page.exists():
        # Try with capital first letter
        page = wiki_wiki.page(species_name.capitalize())

    if not page.exists():
        print(f"⚠ Wikipedia page not found for: {species_name}")
        return None

    # Get the summary
    summary = page.summary

    # Extract relevant sections
    appearance_text = []
    relevant_keywords = [
        "description",
        "characteristics",
        "appearance",
        "anatomy",
        "morphology",
        "physical",
    ]

    def extract_sections(sections, level=0):
        """Recursively extract sections with relevant keywords"""
        for section in sections:
            # Check if section title contains relevant keywords
            if any(keyword in section.title.lower() for keyword in relevant_keywords):
                appearance_text.append(f"\n## {section.title}\n{section.text}")

            # Recursively check subsections
            if hasattr(section, "sections") and section.sections:
                extract_sections(section.sections, level + 1)

    # Extract from all sections
    if hasattr(page, "sections") and page.sections:
        extract_sections(page.sections)

    return {
        "species": species_name,
        "page_title": page.title,
        "summary": summary,
        "appearance_sections": (
            "\n".join(appearance_text) if appearance_text else summary
        ),
        "url": page.fullurl,
    }


# %%
def generate_visually_relevant_summary(wiki_text, species_name, model="gpt-4o-mini"):
    """
    Use GPT-4 to generate Visually Relevant Summary (VRS) from Wikipedia text.
    This follows the prompt from Appendix A of the WildMatch paper.

    Args:
        wiki_text: Wikipedia article text
        species_name: Name of the species
        model: OpenAI model to use

    Returns:
        Visually relevant description string
    """
    system_msg = (
        "You are an AI assistant specialized in biology and providing accurate and "
        "detailed descriptions of animal species."
    )

    user_msg = f"""You are given the description of an animal species. Provide a very detailed description of the appearance of the species and describe each body part of the animal in detail. Only include details that can be directly visible in a photograph of the animal. Only include information related to the appearance of the animal and nothing else. Make sure to only include information that is present in the species description and is certainly true for the given species. Do not include any information related to the sound or smell of the animal. Do not include any numerical information related to measurements in the text in units: m, cm, in, inches, ft, feet, km/h, kg, lb, lbs. Remove any special characters such as unicode tags from the text. Return the answer as a single paragraph.

Species description: {wiki_text}

Answer:"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )

        vrs = completion.choices[0].message.content.strip()
        return vrs

    except Exception as e:
        print(f"❌ Error generating VRS for {species_name}: {e}")
        return None


# %%
def build_knowledge_base(
    species_list, output_path="data/knowledge_base.json", skip_existing=True
):
    """
    Build the complete knowledge base for all species.

    Args:
        species_list: List of species names
        output_path: Path to save the knowledge base
        skip_existing: If True, skip species that already exist in the KB

    Returns:
        Dictionary mapping species names to visual descriptions
    """
    # Manual mapping for known discrepancies
    mapper = {"hyenaspotted": "Spotted_hyena"}
    # Load existing KB if it exists
    knowledge_base = {}
    if skip_existing and os.path.exists(output_path):
        print(f"Loading existing knowledge base from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        print(f"Found {len(knowledge_base)} existing species")

    # Filter out species that already exist
    species_to_process = [
        s for s in species_list if s not in knowledge_base or not knowledge_base.get(s)
    ]

    print(
        f"\nProcessing {len(species_to_process)} species (out of {len(species_list)} total)"
    )
    print("This may take a while due to API rate limits...\n")

    for i, species in enumerate(tqdm(species_to_process, desc="Building KB")):
        # Skip empty species
        if not species or species == "empty":
            continue

        try:
            # Fetch Wikipedia article
            wiki_data = fetch_wikipedia_article(mapper.get(species, species))

            if not wiki_data:
                knowledge_base[species] = None
                continue

            # Generate VRS
            vrs = generate_visually_relevant_summary(
                wiki_data["appearance_sections"], species
            )

            # Store in KB
            knowledge_base[species] = {
                "description": vrs,
                "wikipedia_title": wiki_data["page_title"],
                "wikipedia_url": wiki_data["url"],
                "raw_summary": wiki_data["summary"][:500]
                + "...",  # Store truncated version
            }

            # Save periodically (every 5 species)
            if (i + 1) % 5 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\n❌ Error processing {species}: {e}")
            knowledge_base[species] = None

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Knowledge base saved to {output_path}")
    print(
        f"✓ Successfully processed {sum(1 for v in knowledge_base.values() if v)} species"
    )
    print(
        f"✗ Failed to process {sum(1 for v in knowledge_base.values() if not v)} species"
    )

    return knowledge_base


# Get unique species from your dataset
unique_species = df_balanced["species_name"].dropna().unique().tolist()
print(f"Found {len(unique_species)} unique species in dataset:")
print(unique_species, "..." if len(unique_species) > 10 else "")

# %%
# Build the knowledge base for all species
kb = build_knowledge_base(unique_species, output_path="../data/knowledge_base.json")

# %% [markdown]
# # 3. Task 2: Collect Taxonomic Hierarchy
#
# We'll use an API to get taxonomic information (Class → Order → Family → Genus → Species)

# %%
taxonomies = load_json("../data/taxonomy.json")
taxonomies

# %% [markdown]
# # 4. Task 3: Visual Description Generation (VLM Captioning)
#
# For this task, you have several options:
# 1. **Use OpenAI's GPT-4 Vision API** (easier, but costs per image)
# 2. **Fine-tune LLaVA-7B** (complex, requires GPU training as in the paper)
# 3. **Use other VLM APIs** (Google Gemini, Anthropic Claude with vision, etc.)
#
# Let's implement option 1 (GPT-4 Vision) as a starting point:

# %%
import base64


def encode_image_to_base64(image_path):
    """Encode image to base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_visual_description(image_path, model="gpt-4o-mini", num_samples=1):
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
    # Instruction prompts from the paper (Section 5.2)
    instructions = [
        "Give a very detailed visual description of the animal in the photo.",
        "Describe in detail the visible body parts of the animal in the photo.",
        "What are the visual characteristics of the animal in the photo?",
        "Describe the appearance of the animal in the photo.",
        "What are the identifying characteristics of the animal visible in the photo?",
        "How would you describe the animal in the photo?",
        "What does the animal in the photo look like?",
    ]

    captions = []

    for i in range(num_samples):
        # Randomly select an instruction
        instruction = random.choice(instructions)

        try:
            # For gpt-4o models, we can use vision
            if "gpt-4o" in model:
                # Encode image
                base64_image = encode_image_to_base64(image_path)

                response = client.chat.completions.create(
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


def choose_sample():
    # Select a random sample from the balanced dataset
    sample = df_balanced.sample(n=1).iloc[0]
    test_image_path = sample["full_path"]
    return test_image_path, sample["species_name"]


# %%
def match_caption_to_species(caption, knowledge_base, model="gpt-4o-mini"):
    """
    Use LLM to match a visual description to the most likely species.
    Follows the prompt from Appendix G of the WildMatch paper.

    Args:
        caption: Visual description of the animal
        knowledge_base: Dict mapping species to descriptions
        model: OpenAI model to use

    Returns:
        Predicted species name
    """
    # Format knowledge base
    kb_text_lines = []
    species_names = []

    for species, data in knowledge_base.items():
        if data and data.get("description"):
            species_names.append(species)
            kb_text_lines.append(f"{species}: {data['description']}")

    kb_text = "\n\n".join(kb_text_lines)
    species_list_str = ", ".join(species_names)

    system_msg = (
        "You are an AI expert in biology specialized in animal species identification."
    )

    user_msg = f"""You are given a knowledge base of animal species and their visual appearance:

{kb_text}

Now you are given the following detailed description of an animal seen in a photograph:

\"\"\"{caption}\"\"\"

Task:
- Choose the single most likely species from this list: [{species_list_str}]
- Answer with exactly one of these names, nothing else.
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )

        answer = completion.choices[0].message.content.strip()

        # Post-process to ensure it's one of the species names
        if answer not in species_names:
            # Try fuzzy matching
            answer_lower = answer.lower()
            for sp in species_names:
                if sp.lower() in answer_lower or answer_lower in sp.lower():
                    return sp
            # Default to first species if no match
            print(f"⚠ Warning: LLM returned '{answer}' which is not in species list")
            return species_names[0] if species_names else None

        return answer

    except Exception as e:
        print(f"❌ Error in LLM matching: {e}")
        return None


print("✓ LLM matching function defined")


# %%
def wildmatch_predict(
    image_path,
    knowledge_base,
    n_captions=5,
    vlm_model="gpt-4o-mini",
    llm_model="gpt-4o-mini",
):
    """
    Complete WildMatch prediction pipeline with self-consistency.
    Implements Section 7 of the paper.

    Args:
        image_path: Path to the image
        knowledge_base: Dict mapping species to descriptions
        n_captions: Number of captions to generate for self-consistency
        vlm_model: Model for visual description generation
        llm_model: Model for species matching

    Returns:
        Dict with prediction, confidence, and intermediate results
    """
    # 1. Generate N visual descriptions
    captions = generate_visual_description(
        image_path, model=vlm_model, num_samples=n_captions
    )

    if not captions or all(c is None for c in captions):
        return {
            "prediction": None,
            "confidence": 0.0,
            "error": "Failed to generate captions",
        }

    # 2. Match each caption to species
    species_predictions = []
    for i, caption in enumerate(captions):
        if caption:
            pred = match_caption_to_species(caption, knowledge_base, model=llm_model)
            species_predictions.append(pred)

    # 3. Majority vote (self-consistency)
    from collections import Counter

    vote_counts = Counter(species_predictions)
    best_species, best_count = vote_counts.most_common(1)[0]
    confidence = best_count / n_captions

    result = {
        "prediction": best_species,
        "confidence": confidence,
        "captions": captions,
        "species_votes": species_predictions,
        "vote_counts": dict(vote_counts),
    }

    print(f"\n✓ Prediction: {best_species} (confidence: {confidence:.2%})")
    print(f"  Vote distribution: {dict(vote_counts)}")

    return result


# %%
def wildmatch_predict_dataset(
    df,
    knowledge_base,
    n_captions=5,
    vlm_model="gpt-4o-mini",
    llm_model="gpt-4o-mini",
    output_path="../data/predictions.csv",
):
    """
    Run WildMatch prediction pipeline on entire dataset.

    Args:
        df: DataFrame with 'full_path' and 'species_name' columns
        knowledge_base: Dict mapping species to descriptions
        n_captions: Number of captions to generate for self-consistency
        vlm_model: Model for visual description generation
        llm_model: Model for species matching
        output_path: Path to save predictions CSV

    Returns:
        DataFrame with predictions and ground truth
    """
    results = []

    print(f"Running predictions on {len(df)} images...")
    print(f"Using {n_captions} captions per image for self-consistency")
    print(f"VLM Model: {vlm_model}, LLM Model: {llm_model}\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        image_path = row["full_path"]
        true_species = row["species_name"]

        try:
            # Run prediction
            prediction_result = wildmatch_predict(
                image_path=image_path,
                knowledge_base=knowledge_base,
                n_captions=n_captions,
                vlm_model=vlm_model,
                llm_model=llm_model,
            )

            # Store results
            results.append(
                {
                    "image_path": image_path,
                    "image_id": row.get("id", idx),
                    "true_species": true_species,
                    "predicted_species": prediction_result["prediction"],
                    "confidence": prediction_result["confidence"],
                    "vote_counts": str(prediction_result["vote_counts"]),
                    "correct": prediction_result["prediction"] == true_species,
                }
            )

            # Save periodically (every 10 predictions)
            if (idx + 1) % 10 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_path, index=False)
                print(f"\n✓ Progress saved ({idx + 1}/{len(df)} images)")

        except Exception as e:
            print(f"\n❌ Error predicting image {image_path}: {e}")
            results.append(
                {
                    "image_path": image_path,
                    "image_id": row.get("id", idx),
                    "true_species": true_species,
                    "predicted_species": None,
                    "confidence": 0.0,
                    "vote_counts": None,
                    "correct": False,
                }
            )

    # Create DataFrame and save
    predictions_df = pd.DataFrame(results)
    predictions_df.to_csv(output_path, index=False)

    # Print summary
    accuracy = predictions_df["correct"].mean()
    print(f"\n✓ Predictions complete!")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Overall Accuracy: {accuracy:.2%}")

    return predictions_df


print("✓ Batch prediction function defined")

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns


def calculate_metrics(predictions_df, save_confusion_matrix=True, output_dir="../data"):
    """
    Calculate precision, accuracy, recall, F1-score, and confusion matrix.

    Args:
        predictions_df: DataFrame with 'true_species' and 'predicted_species' columns
        save_confusion_matrix: Whether to save confusion matrix plot
        output_dir: Directory to save confusion matrix plot

    Returns:
        Dictionary with all metrics
    """
    # Remove rows with failed predictions
    df_valid = predictions_df[predictions_df["predicted_species"].notna()].copy()

    if len(df_valid) == 0:
        print("❌ No valid predictions found!")
        return None

    y_true = df_valid["true_species"]
    y_pred = df_valid["predicted_species"]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Get unique classes
    classes = sorted(y_true.unique())

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Print results
    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f} ({precision:.2%})")
    print(f"Recall:    {recall:.4f} ({recall:.2%})")
    print(f"F1-Score:  {f1:.4f} ({f1:.2%})")
    print(f"\nTotal samples: {len(df_valid)}")
    print(f"Failed predictions: {len(predictions_df) - len(df_valid)}")

    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)

    # Create per-class metrics DataFrame
    per_class_metrics = pd.DataFrame(
        {
            "Species": classes,
            "Precision": precision_per_class,
            "Recall": recall_per_class,
            "F1-Score": f1_per_class,
            "Support": [sum(y_true == c) for c in classes],
        }
    )
    print(per_class_metrics.to_string(index=False))

    # Plot confusion matrix
    if save_confusion_matrix:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix", fontsize=16, pad=20)
        plt.xlabel("Predicted Species", fontsize=12)
        plt.ylabel("True Species", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Confusion matrix saved to: {cm_path}")
        plt.show()

    # Return metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "classes": classes,
        "n_samples": len(df_valid),
        "n_failed": len(predictions_df) - len(df_valid),
    }

    return metrics


print("✓ Metrics calculation function defined")

# %% [markdown]
# ### 6.1 Run Predictions on Dataset
#
# **Warning:** This will take a considerable amount of time and use OpenAI API credits. Consider starting with a small subset for testing.

# %%
# Option 1: Test on a small subset first (recommended)

# Run predictions on test subset
predictions_df = wildmatch_predict_dataset(
    df=df_balanced,
    knowledge_base=kb,
    n_captions=5,  # Use 5 captions for self-consistency
    vlm_model="gpt-4o-mini",
    llm_model="gpt-4o-mini",
    output_path="../results/predictions_test.csv",
)
