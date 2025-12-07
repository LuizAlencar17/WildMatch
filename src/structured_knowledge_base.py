"""
Structured knowledge base builder.
Extracts attribute-based descriptions from Wikipedia articles.
"""

import json
from typing import Dict, List, Optional
from tqdm import tqdm
import wikipediaapi
from openai import OpenAI

from .utils import save_json


class StructuredKnowledgeBaseBuilder:
    """Build structured attribute-based knowledge base from Wikipedia articles."""

    # Structured prompt for extracting attributes from Wikipedia text
    ATTRIBUTE_EXTRACTION_PROMPT = """You are given a description of an animal species from Wikipedia. Extract ONLY visually observable attributes that would be visible in a photograph.

Fill in the following fields using ONLY information that is present in the text and certainly true for this species:

Required attributes:
- body_size: small, medium, or large (relative to other mammals)
- coat_color: dominant color(s) of fur/skin
- coat_pattern: smooth, spotted, striped, patched, solid, mixed
- ear_shape: pointed, rounded, long, short, large, small
- horns_antlers: none, short horns, long horns, curved horns, straight horns, antlers, tusks
- tail_shape: long, short, bushy, thin, tufted
- facial_markings: stripes, patches, mask-like, spots, plain, distinctive features
- legs_paws: hooves, paws, long legs, short legs, slim, thick
- body_shape: stocky, slender, robust, compact, elongated
- distinctive_features: any other very distinctive visible features unique to this species

Do NOT include:
- Numerical measurements (height, weight, speed)
- Behavioral information
- Sound or smell descriptions
- Geographic or habitat information (unless it's about physical adaptations)

Return ONLY a valid JSON object with these exact keys. Use "unknown" if the information is not available in the text."""

    def __init__(self, openai_api_key: str, user_agent: str = "WildMatch/1.0"):
        """
        Initialize the structured knowledge base builder.

        Args:
            openai_api_key: OpenAI API key
            user_agent: User agent for Wikipedia API
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")

        # Manual mapping for species with different Wikipedia page names
        self.species_mapper = {"hyenaspotted": "Spotted_hyena"}

    def fetch_wikipedia_article(self, species_name: str) -> Optional[Dict[str, str]]:
        """
        Fetch Wikipedia article for a species.

        Args:
            species_name: Name of the species

        Returns:
            Dictionary with article content, or None if not found
        """
        # Try to get the page
        page = self.wiki.page(species_name)

        if not page.exists():
            # Try with capital first letter
            page = self.wiki.page(species_name.capitalize())

        if not page.exists():
            print(f"⚠ Wikipedia page not found for: {species_name}")
            return None

        # Get the summary and relevant sections
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
                if any(
                    keyword in section.title.lower() for keyword in relevant_keywords
                ):
                    appearance_text.append(f"\n## {section.title}\n{section.text}")

                if hasattr(section, "sections") and section.sections:
                    extract_sections(section.sections, level + 1)

        if hasattr(page, "sections") and page.sections:
            extract_sections(page.sections)

        combined_text = "\n".join(appearance_text) if appearance_text else summary

        return {
            "species": species_name,
            "page_title": page.title,
            "text": combined_text[:4000],  # Limit to avoid token limits
            "url": page.fullurl,
        }

    def extract_structured_attributes(
        self, wiki_text: str, species_name: str, model: str = "gpt-4o-mini"
    ) -> Optional[Dict[str, str]]:
        """
        Extract structured attributes from Wikipedia text using GPT-4.

        Args:
            wiki_text: Wikipedia article text
            species_name: Name of the species
            model: OpenAI model to use

        Returns:
            Dictionary with extracted attributes
        """
        user_msg = f"""{self.ATTRIBUTE_EXTRACTION_PROMPT}

Species: {species_name}

Wikipedia text:
{wiki_text}

Return the JSON object:"""

        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in animal biology and visual description.",
                    },
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = completion.choices[0].message.content.strip()
            attributes = json.loads(content)
            return attributes

        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON for {species_name}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error extracting attributes for {species_name}: {e}")
            return None

    def build_structured_knowledge_base(
        self,
        species_list: List[str],
        output_path: str = "data/structured_knowledge_base.json",
        skip_existing: bool = True,
    ) -> Dict[str, Dict]:
        """
        Build structured knowledge base for all species.

        Args:
            species_list: List of species names
            output_path: Path to save the knowledge base
            skip_existing: If True, skip species that already exist

        Returns:
            Dictionary mapping species names to structured attributes
        """
        import os

        # Load existing KB if it exists
        knowledge_base = {}
        if skip_existing and os.path.exists(output_path):
            print(f"Loading existing structured knowledge base from {output_path}")
            from .utils import load_json

            knowledge_base = load_json(output_path)
            print(f"Found {len(knowledge_base)} existing species")

        # Filter out species that already exist
        species_to_process = [
            s
            for s in species_list
            if s not in knowledge_base or not knowledge_base.get(s)
        ]

        print(
            f"\nProcessing {len(species_to_process)} species (out of {len(species_list)} total)"
        )
        print("Extracting structured attributes from Wikipedia...\n")

        for i, species in enumerate(
            tqdm(species_to_process, desc="Building Structured KB")
        ):
            # Skip empty species
            if not species or species == "empty":
                continue

            try:
                # Fetch Wikipedia article
                mapped_name = self.species_mapper.get(species, species)
                wiki_data = self.fetch_wikipedia_article(mapped_name)

                if not wiki_data:
                    knowledge_base[species] = None
                    continue

                # Extract structured attributes
                attributes = self.extract_structured_attributes(
                    wiki_data["text"], species
                )

                # Store in KB
                knowledge_base[species] = {
                    "attributes": attributes,
                    "wikipedia_title": wiki_data["page_title"],
                    "wikipedia_url": wiki_data["url"],
                }

                # Save periodically (every 5 species)
                if (i + 1) % 5 == 0:
                    save_json(knowledge_base, output_path)

            except Exception as e:
                print(f"\n❌ Error processing {species}: {e}")
                knowledge_base[species] = None

        # Final save
        save_json(knowledge_base, output_path)

        print(f"\n✓ Structured knowledge base saved to {output_path}")
        print(
            f"✓ Successfully processed {sum(1 for v in knowledge_base.values() if v)} species"
        )
        print(
            f"✗ Failed to process {sum(1 for v in knowledge_base.values() if not v)} species"
        )

        return knowledge_base
