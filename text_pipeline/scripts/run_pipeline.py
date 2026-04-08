"""
Runs the full pipeline: header extraction -> hierarchy -> sections -> grave extraction.
"""

from datetime import datetime
from pathlib import Path

import typer
from grave_extraction.extraction.agent import (
    execute_agent,
    transform_df_for_evaluation,
)
from grave_extraction.llm_factory import get_llm
from grave_extraction.models import ModelProvider
from grave_extraction.pdf_processing.headers import (
    HeaderExtractionWithKMeansClusteringStrategy,
)
from grave_extraction.pdf_processing.hierarchy.extractors import (
    NumberedPatternParsingHierarchyExtractionStrategy,
)
from grave_extraction.pdf_processing.hierarchy.processors import (
    LLMRecursiveProcessor,
)
from grave_extraction.pdf_processing.sections import HybridSectionsBuilder
from grave_extraction.prompt_strategy import Jinja2FilePromptStrategy
from grave_extraction.tracing import init_tracing

app = typer.Typer()


@app.command()
def main(
    pdf_path: Path = typer.Option(
        Path("data/Grabfunde_Teil 1.pdf"),
        "--pdf-path",
        help="Path to the input PDF file.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/experiments/grave-extraction/first-catalogue"),
        "--output-dir",
        help="Directory for the pipeline output CSV.",
    ),
    provider: ModelProvider = typer.Option("gwdg", "--provider", help="LLM provider."),
    model: str = typer.Option(
        "mistral-large-instruct", "--model", help="Model name for the LLM provider."
    ),
    hierarchy_model: str | None = typer.Option(
        None,
        "--hierarchy-model",
        help=(
            "Model name used for hierarchy extraction. "
            "Defaults to --model when omitted."
        ),
    ),
    prompt_template: Path = typer.Option(
        Path(
            "prompts/grave-extraction/extract_graves_few_shot_2_grave_examples_improved_and_images.jinja2"
        ),
        "--prompt-template",
        help="Path to the Jinja2 grave extraction prompt template.",
    ),
    hierarchy_prompt: str = typer.Option(
        "extract_hierarchy_hybrid_few_shot",
        "--hierarchy-prompt",
        help="Prompt template name (without .jinja2) for hierarchy extraction.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    selected_hierarchy_model = hierarchy_model or model

    headers = HeaderExtractionWithKMeansClusteringStrategy(
        str(pdf_path)
    ).extract_headers()

    hierarchy = NumberedPatternParsingHierarchyExtractionStrategy(
        unstructured_processor=LLMRecursiveProcessor(
            get_llm(provider, selected_hierarchy_model, temperature=0),
            Jinja2FilePromptStrategy(
                Path(f"prompts/hierarchy-extraction/{hierarchy_prompt}.jinja2")
            ),
        )
    ).extract_hierarchy(headers)

    sections = HybridSectionsBuilder().build_sections(str(pdf_path), hierarchy)

    init_tracing(f"full_pipeline-{timestamp}")

    extraction_df = execute_agent(
        sections,
        llm=get_llm(provider, model, temperature=0),
        grave_extraction_prompt_strategy=Jinja2FilePromptStrategy(prompt_template),
    )

    eval_df = transform_df_for_evaluation(extraction_df)
    output_path = output_dir / f"full_pipeline_{timestamp}.csv"
    eval_df.to_csv(output_path)

    typer.echo(f"Pipeline complete -> {output_path}")


if __name__ == "__main__":
    app()
