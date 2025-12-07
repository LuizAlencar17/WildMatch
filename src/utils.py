"""
Utility functions for WildMatch project.
"""

import json
from typing import Dict, Any


def load_json(json_file_path: str) -> Dict[str, Any]:
    """
    Load JSON file from disk.

    Args:
        json_file_path: Path to JSON file

    Returns:
        Dictionary with JSON contents
    """
    print(f"Loading JSON file from {json_file_path}")
    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        output_path: Path where to save the JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {output_path}")
