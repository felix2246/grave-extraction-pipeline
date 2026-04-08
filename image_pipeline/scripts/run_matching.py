from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from grave_image_matching.main import run_matching


app = typer.Typer()


@app.command()
def main(
    graves_csv: Path = typer.Option(
        Path("grave_image_matching/data/grave_extract.csv"),
        "--graves-csv",
        help="Path to the CSV produced by the text pipeline (grave_extract.csv).",
    ),
    captions_json: Path = typer.Option(
        Path("grave_image_matching/output/images/captions.json"),
        "--captions-json",
        help="Path to captions.json produced by run_extract_images.",
    ),
    output_dir: Path = typer.Option(
        Path("grave_image_matching/output"),
        "--output-dir",
        help="Directory where the grave_matches_<timestamp>.csv will be written.",
    ),
    model: str = typer.Option(
        "mistral-large-instruct",
        "--model",
        help="Model name for the MLLM/LLM provider.",
    ),
    id_model_dir: Path = typer.Option(
        Path("tafel_subbox_id_extraction_model/output"),
        "--id-model-dir",
        help="Directory containing config.yaml and model_final.pth for the ID extraction model (A2-OOL).",
    ),
) -> None:
    """
    Run stage 2 of the image pipeline: grave image matching.
    """
    graves_df = pd.read_csv(graves_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = output_dir / f"grave_matches_{timestamp}.csv"

    # Image dir and tafel-splits are next to captions.json
    image_dir = captions_json.parent
    tafel_split_dir = captions_json.parent / "tafel-splits"

    run_matching(
        graves_df=graves_df,
        captions_json_path=captions_json,
        output_csv_path=output_csv_path,
        model_name=model,
        id_model_dir=str(id_model_dir),
        image_dir=image_dir,
        tafel_split_dir=tafel_split_dir,
    )


if __name__ == "__main__":
    app()
