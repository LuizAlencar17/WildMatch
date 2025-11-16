from dotenv import load_dotenv
import os
import json
from collections import Counter
from typing import Dict, List
from openai import OpenAI

from PIL import Image
from models.vlm_captioner import VLMCaptioner

load_dotenv()

# This uses OPENAI_API_KEY from your .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_match_species(
    caption: str,
    kb: Dict[str, str],
    model: str = "gpt-4o-mini",
) -> str:
    """
    One LLM call: pick the best matching species from the KB.
    Returns a single species name.
    """
    # Format KB as SPECIES: DESCRIPTION, as in Appendix G :contentReference[oaicite:9]{index=9}
    kb_text_lines = []
    species_names = []
    for sp, desc in kb.items():
        species_names.append(sp)
        kb_text_lines.append(f"{sp}: {desc}")
    kb_text = "\n\n".join(kb_text_lines)

    species_list_str = ", ".join(species_names)

    system_msg = (
        "You are an AI expert in biology specialized in animal species identification."
    )

    user_msg = f"""
    You are given a knowledge base of animal species and their visual appearance:

    {kb_text}

    Now you are given the following detailed description of an animal seen in a photograph:

    \"\"\"{caption}\"\"\"

    Task:
    - Choose the single most likely species from this list:
    [{species_list_str}]
    - Answer with exactly one of these names, nothing else.
    """

    completion = client.chat.completions.create(
        model=model,  # e.g. "gpt-4o-mini"
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    answer = completion.choices[0].message.content.strip()



    # You might want to post-process to ensure it’s one of the species_names
    # e.g. by fuzzy-matching or simple exact matching
    if answer not in species_names:
        # naive fallback: pick first species that appears as substring
        for sp in species_names:
            if sp.lower() in answer.lower():
                return sp
        # or default to something
        return species_names[0]
    return answer

def wildmatch_predict(
    image_path: str,
    kb_path: str = "data/knowledge_base.json",
    n_captions: int = 5,
) -> Dict:
    """
    Run the WildMatch-style pipeline on a single image.
    Returns prediction + confidence + all intermediate info.
    """
    # 1) Load KB
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    # 2) Load image
    img = Image.open(image_path).convert("RGB")

    # 3) Caption with VLM (N samples)
    captioner = VLMCaptioner()
    captions = captioner.caption(img, num_samples=n_captions)

    # 4) For each caption, run LLM matching
    species_preds: List[str] = []
    for c in captions:
        sp = llm_match_species(c, kb)
        species_preds.append(sp)

    # 5) Majority vote + confidence (Section 3.6) :contentReference[oaicite:11]{index=11}
    counts = Counter(species_preds)
    best_species, best_count = counts.most_common(1)[0]
    confidence = best_count / n_captions

    result = {
        "prediction": best_species,
        "confidence": confidence,
        "caption_samples": captions,
        "species_votes": species_preds,
        "vote_counts": dict(counts),
    }
    print(f"Predicted species: {result['prediction']} with confidence {result['confidence']:.2f} from votes {result['vote_counts']}")
    return result

if __name__ == "__main__":
    wildmatch_predict("images/zebra.jpeg")
    wildmatch_predict("images/puma.jpeg")