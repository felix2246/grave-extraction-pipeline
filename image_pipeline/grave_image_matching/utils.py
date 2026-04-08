import ast
import base64
import unicodedata
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from grave_image_matching.constants import GRAVE_NAME_COL, TAFEL_ID_EXTRACTOR


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = " ".join(text.split())
    text = text.replace('"', "")
    text = text.replace(" ", "")
    text = text.replace("„", "").replace("“", "")
    text = text.replace("–", "-")

    return text


def encode_image(image_path: Path) -> str:
    """Encodes an image to base64 for the API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_references(row_content: Any) -> list[str]:
    """Helper to safely parse the stringified list of references."""
    if pd.isna(row_content):
        return []

    row_str = str(row_content).replace("\u2013", "-")
    refs = []
    try:
        parsed = ast.literal_eval(row_str)
        if isinstance(parsed, list):
            refs = parsed
        elif isinstance(parsed, str):
            refs = [parsed]
    except (ValueError, SyntaxError):
        return []

    return [r.replace(" ", "").strip() for r in refs if isinstance(r, str)]


def get_tafel_id(text: str) -> Optional[str]:
    """
    Extracts the normalized Tafel number from a string.
    Returns '18' for 'Taf. 18-B/3' or 'Tafel 18'.
    """
    match = TAFEL_ID_EXTRACTOR.search(text)
    if match:
        return match.group(1)  # Returns just the digits, e.g. "18"
    return None


def build_candidate_dict(
    candidate_graves: pd.DataFrame, tafel_id: str
) -> dict[Any, tuple[str, list[str]]]:
    candidate_dict = {}  # type: ignore
    for idx, row in candidate_graves.iterrows():
        refs = parse_references(row["referenzierte_abbildungen"])
        relevant_refs = [r for r in refs if get_tafel_id(r) == tafel_id]

        if relevant_refs:
            if idx not in candidate_dict:
                grave_name = str(row[GRAVE_NAME_COL])
                candidate_dict[idx] = (grave_name, [])
            candidate_dict[idx][1].extend(relevant_refs)
    return candidate_dict
