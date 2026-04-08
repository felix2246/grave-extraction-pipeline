"""Match grave images to grave records based on referenced image patterns."""

import json
import os
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
from dotenv import load_dotenv
from grave_image_matching.constants import (
    ABB_PATTERN,
    DEFAULT_IMAGE_DIR,
    DEFAULT_TAFEL_SPLIT_DIR,
)
from grave_image_matching.logger import logger
from grave_image_matching.tafel_matching_strategies.base import TafelBoxToGraveMatcher
from grave_image_matching.tafel_matching_strategies.extract_ids_first_strategy import (
    TafelBoxToGraveWithIdExtractionAndLLMMatcher,
)
from grave_image_matching.utils import get_tafel_id, parse_references
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
from tqdm import tqdm  # type: ignore[import-untyped]

load_dotenv()

warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")


def process_tafel_matches(
    df: pd.DataFrame,
    tafel_id_map: Dict[str, str],
    image_caption_map: Dict[str, str],
    matcher: TafelBoxToGraveMatcher,
    image_dir: Path,
    tafel_split_dir: Path,
    output_csv_path: Path,
) -> None:
    """
    Orchestrates the matching process using the provided strategy (matcher).
    """
    logger.debug("Starting Tafel processing", tafel_id_count=len(tafel_id_map))

    #  Tafel ID -> List of Grave Indices
    tafel_id_to_grave_indices = defaultdict(list)

    for idx, row in df.iterrows():
        refs = parse_references(row["referenzierte_abbildungen"])
        for ref in refs:
            # We check if it looks like a Tafel reference
            tid = get_tafel_id(ref)
            if tid:
                tafel_id_to_grave_indices[tid].append(idx)

    logger.debug(
        "Found unique Tafel IDs in CSV references",
        count=len(tafel_id_to_grave_indices),
    )

    # Pre-filter to only IDs that can actually be processed (have extracted subimages)
    processable_tafeln: list[tuple[str, list[int], str, str, Path, list[Path]]] = []
    for tid, grave_indices in tafel_id_to_grave_indices.items():
        if not tid:
            raise Exception("tid should not be null")

        if tid not in tafel_id_map:
            logger.debug(
                "Tafel ID found in CSV but not in captions.json map", tafel_id=tid
            )
            continue

        filename = tafel_id_map[tid]
        tafel_caption = image_caption_map.get(filename, "")
        tafel_image_path = image_dir / filename

        folder_name = Path(filename).stem
        tafel_folder_path = tafel_split_dir / folder_name
        if not tafel_folder_path.exists() or not tafel_folder_path.is_dir():
            logger.debug("Folder missing", path=str(tafel_folder_path), tafel_id=tid)
            continue

        subimages = list(tafel_folder_path.glob("*"))
        subimages = [
            p
            for p in subimages
            if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
        if not subimages:
            logger.debug("Folder exists but is empty", path=str(tafel_folder_path))
            continue

        processable_tafeln.append(
            (tid, grave_indices, filename, tafel_caption, tafel_image_path, subimages)
        )

    logger.debug("Processable Tafel IDs with extracted images", count=len(processable_tafeln))

    # Iterate only over actually processable Tafel IDs
    for tid, grave_indices, filename, tafel_caption, tafel_image_path, subimages in tqdm(
        processable_tafeln
    ):

        # Get the subset of graves that point to this Tafel
        candidate_graves_df = df.loc[grave_indices]

        # match each subimage to a grave
        matches_count = 0
        for subimage in subimages:
            matched_grave_idx = matcher.match(
                subimage_path=subimage,
                candidate_graves=candidate_graves_df,
                tafel_id=tid,
                tafel_image_path=tafel_image_path,
                tafel_caption=tafel_caption,
            )

            if matched_grave_idx is not None and matched_grave_idx in grave_indices:
                # Update DataFrame
                current_list = df.at[matched_grave_idx, "matched_filenames"]
                current_list.append(str(subimage))  # type: ignore
                df.at[matched_grave_idx, "matched_filenames"] = current_list
                matches_count += 1

                # Save intermediate result
                df.to_csv(output_csv_path, encoding="utf-8")

        if matches_count == 0:
            logger.warning("No matches made for Tafel", tafel_id=tid)


def process_abb_matches(
    df: pd.DataFrame, image_lookup_map: Dict[str, Any], image_dir: Path
) -> pd.Series:
    """Standard logic: Iterate rows -> find 'Abb.' references -> look up files in map."""
    matched_series = df.apply(lambda x: [], axis=1)

    for idx, row in df.iterrows():
        refs = parse_references(row["referenzierte_abbildungen"])
        found_files = []

        for ref in refs:
            abb_match = ABB_PATTERN.match(ref)
            if abb_match:
                try:
                    base_ref = abb_match.group(1)
                except IndexError:
                    base_ref = abb_match.group(0)

                lookup_key = base_ref.replace(" ", "")

                if lookup_key in image_lookup_map:
                    fnames = [
                        str(image_dir / item["filename"])
                        for item in image_lookup_map[lookup_key]
                    ]
                    found_files.extend(fnames)

        matched_series.at[idx] = found_files  # type: ignore

    return matched_series


_PHOENIX_CHECK_URL = os.environ.get("PHOENIX_CHECK_URL", "http://127.0.0.1:6006")


def _phoenix_reachable(url: str = _PHOENIX_CHECK_URL, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(url.rstrip("/") + "/", timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def init_tracing(model_name: str) -> None:
    try:
        if not _phoenix_reachable():
            logger.warning(
                "Tracing disabled – Phoenix not reachable",
                url=_PHOENIX_CHECK_URL,
            )
            return

        tracer_provider = register(project_name=f"mllm-tafel-matching/{model_name}")
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Initialized tracing with Phoenix")
    except Exception as e:
        logger.warning("Tracing disabled – Phoenix not available", error=str(e))


def run_matching(
    graves_df: pd.DataFrame,
    captions_json_path: Path,
    output_csv_path: Path,
    model_name: str,
    id_model_dir: str,
    image_dir: Path | None = None,
    tafel_split_dir: Path | None = None,
) -> None:
    """
    Run the full matching process given graves data, captions, and model configuration.
    """
    with open(captions_json_path, "r", encoding="utf-8") as f:
        image_caption_map = json.load(f)

    if image_dir is None:
        image_dir = DEFAULT_IMAGE_DIR
    if tafel_split_dir is None:
        tafel_split_dir = DEFAULT_TAFEL_SPLIT_DIR

    tafel_subbox_matcher = TafelBoxToGraveWithIdExtractionAndLLMMatcher(
        client=OpenAI(
            api_key=os.environ["GWDG_API_KEY"],
            base_url="https://chat-ai.academiccloud.de/v1",
        ),
        model_name=model_name,
        graves_df=graves_df,
        id_model_config_dir=id_model_dir,
    )

    init_tracing(f"{tafel_subbox_matcher.__class__.__name__}/{model_name}")

    # Build Lookup Maps
    abb_lookup_map: Dict[str, Any] = defaultdict(list)
    tafel_id_map: Dict[str, str] = {}

    logger.debug("Building lookup maps")
    for filename, full_caption in image_caption_map.items():
        if not full_caption:
            continue

        clean_caption = full_caption.strip()

        # Check Abb
        if match := ABB_PATTERN.match(clean_caption):
            try:
                key = match.group(1).replace(" ", "")
            except IndexError:
                key = match.group(0).replace(" ", "")
            abb_lookup_map[key].append({"filename": filename, "caption": full_caption})

        # Check Tafel
        if tid := get_tafel_id(clean_caption):
            tafel_id_map[tid] = filename

    if len(tafel_id_map) > 0:
        logger.debug("Successfully mapped Tafel IDs", count=len(tafel_id_map))

    logger.info("Processing Abb matches")
    graves_df["matched_filenames"] = process_abb_matches(
        graves_df, abb_lookup_map, image_dir
    )

    logger.info("Processing Tafel matches")
    process_tafel_matches(
        df=graves_df,
        tafel_id_map=tafel_id_map,
        image_caption_map=image_caption_map,
        matcher=tafel_subbox_matcher,
        image_dir=image_dir,
        tafel_split_dir=tafel_split_dir,
        output_csv_path=output_csv_path,
    )

    # Save Final Results
    graves_df.to_csv(output_csv_path, encoding="utf-8")
    logger.info("Done – results written", path=str(output_csv_path))


if __name__ == "__main__":
    MODEL_NAME = "mistral-large-instruct"
    OUTPUT_CSV_PATH = Path(
        f"grave_image_matching/output/grave_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    graves_df = pd.read_csv("grave_image_matching/data/grave_extract.csv")
    captions_path = Path("grave_image_matching/output/images/captions.json")

    run_matching(
        graves_df=graves_df,
        captions_json_path=captions_path,
        output_csv_path=OUTPUT_CSV_PATH,
        model_name=MODEL_NAME,
        id_model_dir="tafel_subbox_id_extraction_model/output",
    )
