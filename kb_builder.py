# kb_builder.py
import json
import requests
from typing import List, Dict

import openai  # or another LLM client

openai.api_key = "YOUR_API_KEY"

WIKI_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

def fetch_wiki_summary(title: str) -> str:
    resp = requests.get(WIKI_API_URL.format(title.replace(" ", "_")))
    resp.raise_for_status()
    data = resp.json()
    # You can also combine 'extract' with other sections if needed
    return data.get("extract", "")

def summarize_visual_appearance(raw_text: str) -> str:
    system_msg = (
        "You are an AI assistant specialized in biology and providing accurate "
        "and detailed descriptions of animal species."
    )

    user_msg = f"""
You are given the description of an animal species.

Provide a very detailed description of the appearance of the species and describe
each body part of the animal in detail.

Rules:
- Only include details that can be directly visible in a photograph.
- Only include information related to appearance, nothing about behavior, sound or smell.
- Do not include numerical information with units like m, cm, in, ft, kg, lb, km/h, etc.
- Only include information that is explicitly present in the text below.
- Return the answer as a single paragraph.

Species description:
\"\"\"{raw_text}\"\"\"
"""

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or another model you have
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message["content"].strip()

def build_knowledge_base(species_list: List[str]) -> Dict[str, str]:
    kb = {}
    for species in species_list:
        raw = fetch_wiki_summary(species)
        if not raw:
            print(f"[WARN] No Wikipedia text for {species}")
            continue
        vis = summarize_visual_appearance(raw)
        kb[species] = vis
    return kb

def save_kb(kb: Dict[str, str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # example: species at genus level etc.
    species_list = [
        "Puma concolor",
        "Panthera onca",
        "Odocoileus virginianus",
        # ...
    ]
    kb = build_knowledge_base(species_list)
    save_kb(kb, "data/knowledge_base.json")
