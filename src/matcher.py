"""
LLM matching module for species identification.
"""

from typing import Dict, Optional
from openai import OpenAI


class SpeciesMatcher:
    """Match visual descriptions to species using LLM."""

    def __init__(self, openai_api_key: str):
        """
        Initialize the species matcher.

        Args:
            openai_api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=openai_api_key)

    def match_caption_to_species(
        self, caption: str, knowledge_base: Dict[str, Dict], model: str = "gpt-4o-mini"
    ) -> Optional[str]:
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

        system_msg = "You are an AI expert in biology specialized in animal species identification."

        user_msg = f"""You are given a knowledge base of animal species and their visual appearance:

{kb_text}

Now you are given the following detailed description of an animal seen in a photograph:

\"\"\"{caption}\"\"\"

Task:
- Choose the single most likely species from this list: [{species_list_str}]
- Answer with exactly one of these names, nothing else.
"""

        try:
            completion = self.client.chat.completions.create(
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
                print(
                    f"⚠ Warning: LLM returned '{answer}' which is not in species list"
                )
                return species_names[0] if species_names else None

            return answer

        except Exception as e:
            print(f"❌ Error in LLM matching: {e}")
            return None
