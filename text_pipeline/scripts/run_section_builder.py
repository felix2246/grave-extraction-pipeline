"""
Builds document sections from a PDF using a pre-extracted hierarchy JSON.
"""

import json
from pathlib import Path

import typer

from grave_extraction.pdf_processing.hierarchy.extractors import (
    HierarchyExtractionFromSavedFileStrategy,
)
from grave_extraction.pdf_processing.sections import HybridSectionsBuilder

app = typer.Typer()


@app.command()
def main(
    pdf_path: Path = typer.Option(
        ...,
        "--pdf-path",
        help="Path to the input PDF file.",
    ),
    hierarchy_file: Path = typer.Option(
        ...,
        "--hierarchy-file",
        help="Path to the JSON file containing the header hierarchy.",
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        help="Path for the output sections JSON file.",
    ),
):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    headers = HierarchyExtractionFromSavedFileStrategy(
        str(hierarchy_file)
    ).extract_hierarchy([])
    sections = HybridSectionsBuilder().build_sections(str(pdf_path), headers)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    typer.echo(f"Built {len(sections)} sections -> {output_file}")


if __name__ == "__main__":
    app()
