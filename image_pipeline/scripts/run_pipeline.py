import sys
from pathlib import Path

# Ensure image_pipeline root is on path when running as scripts/run_pipeline.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from scripts.run_extract_images import main as run_extract_images_main
from scripts.run_matching import main as run_matching_main

app = typer.Typer()


@app.command()
def main(
    pdf_path: Path = typer.Option(
        Path("grave_image_matching/data/katalog2_tafeln.pdf"),
        "--pdf-path",
        help="Path to the input PDF containing plates and figures.",
    ),
    graves_csv: Path = typer.Option(
        Path("grave_image_matching/data/grave_extract.csv"),
        "--graves-csv",
        help="Path to the CSV produced by the text pipeline (grave_extract.csv).",
    ),
    output_dir: Path = typer.Option(
        Path("grave_image_matching/output"),
        "--output-dir",
        help="Base output directory for images, captions, and final matches.",
    ),
    model: str = typer.Option(
        "mistral-large-instruct",
        "--model",
        help="Model name for the MLLM/LLM provider.",
    ),
    seg_model_dir: Path = typer.Option(
        Path("tafel_segmentation_model/output"),
        "--seg-model-dir",
        help="Directory containing config.yaml and model_final.pth for the plate segmentation model.",
    ),
    id_model_dir: Path = typer.Option(
        Path("tafel_subbox_id_extraction_model/output"),
        "--id-model-dir",
        help="Directory containing config.yaml and model_final.pth for the ID extraction model (A2-OOL).",
    ),
) -> None:
    """
    Run the full image pipeline: image extraction + caption assignment + plate segmentation
    followed by grave image matching.
    """
    images_output_dir = output_dir / "images"

    # extract images + captions + plate segmentation
    run_extract_images_main(
        pdf_path=pdf_path,
        output_dir=images_output_dir,
        seg_model_dir=seg_model_dir,
    )

    # matching
    captions_json = images_output_dir / "captions.json"
    run_matching_main(
        graves_csv=graves_csv,
        captions_json=captions_json,
        output_dir=output_dir,
        model=model,
        id_model_dir=id_model_dir,
    )


if __name__ == "__main__":
    app()
