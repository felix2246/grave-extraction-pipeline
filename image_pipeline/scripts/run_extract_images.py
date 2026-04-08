from pathlib import Path

import typer
from grave_image_matching.constants import TAFEL_PATTERN
from grave_image_matching.extract_boxes import extract_from_image
from grave_image_matching.extract_images_with_captions import (
    extract_images_and_save_json,
)

app = typer.Typer()


@app.command()
def main(
    pdf_path: Path = typer.Option(
        Path("grave_image_matching/data/katalog2_tafeln.pdf"),
        "--pdf-path",
        help="Path to the input PDF containing plates and figures.",
    ),
    output_dir: Path = typer.Option(
        Path("grave_image_matching/output/images"),
        "--output-dir",
        help="Directory where extracted images and captions.json will be written.",
    ),
    seg_model_dir: Path = typer.Option(
        Path("tafel_segmentation_model/output"),
        "--seg-model-dir",
        help="Directory containing config.yaml and model_final.pth for the plate segmentation model.",
    ),
) -> None:
    """
    Run stage 1 of the image pipeline: image extraction + caption assignment + plate segmentation.
    """
    # Extract images and captions
    captions_map = extract_images_and_save_json(str(pdf_path), str(output_dir))

    # Post-process Tafeln: split plates into sub-images using the segmentation model
    tafel_splits_base = output_dir / "tafel-splits"
    tafel_splits_base.mkdir(parents=True, exist_ok=True)

    for filename, caption in captions_map.items():
        if caption and TAFEL_PATTERN.search(caption):
            full_image_path = output_dir / filename
            image_base_name = Path(filename).stem
            image_output_dir = tafel_splits_base / image_base_name
            image_output_dir.mkdir(parents=True, exist_ok=True)

            extract_from_image(
                image_path=str(full_image_path),
                output_base_dir=str(image_output_dir),
                seg_model_dir=str(seg_model_dir),
            )


if __name__ == "__main__":
    app()
