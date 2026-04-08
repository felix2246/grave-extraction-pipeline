import re
from pathlib import Path

TAFEL_PATTERN = re.compile(r"(?i)^Taf(?:el)?\.?\s")
ABB_PATTERN = re.compile(r"^(Abb\.\s*\d+[A-Z]?)")

GRAVE_NAME_COL = "grab_identifikation"
TAFEL_ID_EXTRACTOR = re.compile(r"(?:Taf(?:\.|el)?)\s*(\d+)", re.IGNORECASE)

DEFAULT_IMAGE_DIR = Path("grave_image_matching/output/images")
DEFAULT_TAFEL_SPLIT_DIR = DEFAULT_IMAGE_DIR / "tafel-splits"


def get_image_dirs(base_output_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Return (image_dir, tafel_split_dir) using either a provided base directory
    or the default locations.
    """
    if base_output_dir is None:
        image_dir = DEFAULT_IMAGE_DIR
    else:
        image_dir = base_output_dir

    tafel_split_dir = image_dir / "tafel-splits"
    return image_dir, tafel_split_dir
