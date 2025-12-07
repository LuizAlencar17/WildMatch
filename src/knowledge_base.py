"""
Knowledge base building module for WildMatch.
Fetches Wikipedia articles and generates visually relevant summaries using GPT-4.
"""

import os
from typing import Dict, List, Optional
from tqdm import tqdm
import wikipediaapi
from openai import OpenAI

from .utils import save_json


class KnowledgeBaseBuilder:
    """Build knowledge base from Wikipedia articles with GPT-4 generated summaries."""

    def __init__(self, openai_api_key: str, user_agent: str = "WildMatch/1.0"):
        """
        Initialize the knowledge base builder.

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
        Extract summary and relevant sections about appearance.

        Args:
            species_name: Name of the species (e.g., 'zebra', 'elephant')

        Returns:
            Dictionary with summary and appearance sections, or None if not found
        """
        # Try to get the page
        page = self.wiki.page(species_name)

        if not page.exists():
            # Try with capital first letter
            page = self.wiki.page(species_name.capitalize())

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
                if any(
                    keyword in section.title.lower() for keyword in relevant_keywords
                ):
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

    def generate_visually_relevant_summary(
        self, wiki_text: str, species_name: str, model: str = "gpt-4o-mini"
    ) -> Optional[str]:
        """
        Use GPT-4 to generate Visually Relevant Summary (VRS) from Wikipedia text.
        Follows the prompt from Appendix A of the WildMatch paper.

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
            completion = self.client.chat.completions.create(
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

    def build_knowledge_base(
        self,
        species_list: List[str],
        output_path: str = "data/knowledge_base.json",
        skip_existing: bool = True,
    ) -> Dict[str, Dict]:
        """
        Build the complete knowledge base for all species.

        Args:
            species_list: List of species names
            output_path: Path to save the knowledge base
            skip_existing: If True, skip species that already exist in the KB

        Returns:
            Dictionary mapping species names to visual descriptions
        """
        # Load existing KB if it exists
        knowledge_base = {}
        if skip_existing and os.path.exists(output_path):
            print(f"Loading existing knowledge base from {output_path}")
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
        print("This may take a while due to API rate limits...\n")

        for i, species in enumerate(tqdm(species_to_process, desc="Building KB")):
            # Skip empty species
            if not species or species == "empty":
                continue

            try:
                # Fetch Wikipedia article (use mapper if available)
                mapped_name = self.species_mapper.get(species, species)
                wiki_data = self.fetch_wikipedia_article(mapped_name)

                if not wiki_data:
                    knowledge_base[species] = None
                    continue

                # Generate VRS
                vrs = self.generate_visually_relevant_summary(
                    wiki_data["appearance_sections"], species
                )

                # Store in KB
                knowledge_base[species] = {
                    "description": vrs,
                    "wikipedia_title": wiki_data["page_title"],
                    "wikipedia_url": wiki_data["url"],
                    "raw_summary": wiki_data["summary"][:500] + "...",
                }

                # Save periodically (every 5 species)
                if (i + 1) % 5 == 0:
                    save_json(knowledge_base, output_path)

            except Exception as e:
                print(f"\n❌ Error processing {species}: {e}")
                knowledge_base[species] = None

        # Final save
        save_json(knowledge_base, output_path)

        print(f"\n✓ Knowledge base saved to {output_path}")
        print(
            f"✓ Successfully processed {sum(1 for v in knowledge_base.values() if v)} species"
        )
        print(
            f"✗ Failed to process {sum(1 for v in knowledge_base.values() if not v)} species"
        )

        return knowledge_base
