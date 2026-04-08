"""
Extracts headers from a PDF and saves them as JSON.

Supported strategies:
  kmeans  – K-Means font-style clustering (default, no ML models required)
  marker  – Marker-PDF table-of-contents extraction (requires marker models)
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from grave_extraction.pdf_processing.headers import (
    HeaderExtractionWithKMeansClusteringStrategy,
    HeadersExtractionStrategy,
    HeadersExtractionWithMarkerStrategy,
)

STRATEGIES = ("kmeans", "marker")

app = typer.Typer()


@app.command()
def main(
    pdf_path: Path = typer.Option(
        Path("data/Grabfunde_text_only.pdf"),
        "--pdf-path",
        help="Path to the input PDF file.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/experiments/header-extraction"),
        "--output-dir",
        help="Directory to write output JSON and TXT files.",
    ),
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help=f"Extraction strategy to use. One of: {', '.join(STRATEGIES)}.",
        ),
    ] = "kmeans",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor: HeadersExtractionStrategy
    if strategy == "kmeans":
        extractor = HeaderExtractionWithKMeansClusteringStrategy(
            str(pdf_path), plot_output_path=str(output_dir / "cluster_plot.png")
        )
        stem = "style_clustering_k_means"
    elif strategy == "marker":
        extractor = HeadersExtractionWithMarkerStrategy(str(pdf_path))
        stem = "marker"
    else:
        typer.echo(
            f"Unknown strategy '{strategy}'. Choose one of: {', '.join(STRATEGIES)}."
        )
        raise typer.Exit(1)

    headers = extractor.extract_headers()

    json_path = output_dir / f"{stem}.json"
    txt_path = output_dir / f"{stem}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(headers, f, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        for h in headers:
            f.write(h["header_text"] + "\n")

    typer.echo(f"Extracted {len(headers)} headers -> {json_path}")


if __name__ == "__main__":
    app()
