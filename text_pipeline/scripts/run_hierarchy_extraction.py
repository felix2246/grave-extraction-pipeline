"""
Extracts a heading hierarchy from a flat list of headers saved as JSON.
"""

import json
from datetime import datetime
from pathlib import Path

import typer

from grave_extraction.llm_factory import get_llm
from grave_extraction.models import ModelProvider
from grave_extraction.pdf_processing.hierarchy.extractors import (
    NumberedPatternParsingHierarchyExtractionStrategy,
)
from grave_extraction.pdf_processing.hierarchy.processors import (
    LLMRecursiveProcessor,
)
from grave_extraction.prompt_strategy import Jinja2FilePromptStrategy
from grave_extraction.tracing import init_tracing

app = typer.Typer()


@app.command()
def main(
    headers_file: Path = typer.Option(
        ...,
        "--headers-file",
        help="Path to the JSON file containing the flat list of headers.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/experiments/hierarchy-extraction"),
        "--output-dir",
        help="Directory for output files.",
    ),
    prompt_file: str = typer.Option(
        "extract_hierarchy_hybrid_few_shot",
        "--prompt-file",
        help="Prompt template name (without .jinja2 extension) inside prompts/hierarchy-extraction/.",
    ),
    provider: ModelProvider = typer.Option("gwdg", "--provider", help="LLM provider."),
    model: str = typer.Option(
        "deepseek-r1", "--model", help="Model name for the LLM provider."
    ),
    run_name: str = typer.Option(
        "",
        "--run-name",
        help="Name prefix for output files. Defaults to a timestamped name.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(headers_file, "r", encoding="utf-8") as f:
        gt_headers = json.load(f)

    name = (
        run_name or f"numbering_pattern_with_llm_{model}_{datetime.now().isoformat()}"
    )

    init_tracing(f"extract_hierarchy_{name}")

    extractor = NumberedPatternParsingHierarchyExtractionStrategy(
        unstructured_processor=LLMRecursiveProcessor(
            get_llm(provider, model, temperature=0),
            Jinja2FilePromptStrategy(
                Path(f"prompts/hierarchy-extraction/{prompt_file}.jinja2")
            ),
        )
    )

    hierarchy = extractor.extract_hierarchy(gt_headers)

    json_out = output_dir / f"{name}.json"
    txt_out = output_dir / f"{name}.txt"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False)

    with open(txt_out, "w", encoding="utf-8") as f:
        for h in hierarchy:
            f.write(f"{'    ' * (h['heading_level'] or 0)}{h['header_text']}\n")

    typer.echo(f"Hierarchy extracted -> {json_out}")


if __name__ == "__main__":
    app()
