"""
Extracts grave records from a sections JSON file using an LLM agent.
"""

import json
import random
from pathlib import Path

import typer

from grave_extraction.extraction.agent import (
    execute_agent,
    transform_df_for_evaluation,
)
from grave_extraction.llm_factory import get_llm
from grave_extraction.models import ModelProvider
from grave_extraction.prompt_strategy import Jinja2FilePromptStrategy
from grave_extraction.tracing import init_tracing
from grave_extraction.utils import read_file

app = typer.Typer()


@app.command()
def main(
    sections_file: Path = typer.Option(
        ...,
        "--sections-file",
        help="Path to the JSON file containing document sections.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/experiments/grave-extraction"),
        "--output-dir",
        help="Directory for the output CSV file.",
    ),
    provider: ModelProvider = typer.Option("gwdg", "--provider", help="LLM provider."),
    model: str = typer.Option(
        "mistral-large-instruct", "--model", help="Model name for the LLM provider."
    ),
    prompt_template: Path = typer.Option(
        Path(
            "prompts/grave-extraction/extract_graves_few_shot_2_grave_examples_improved_and_images.jinja2"
        ),
        "--prompt-template",
        help="Path to the Jinja2 prompt template.",
    ),
):

    output_dir.mkdir(parents=True, exist_ok=True)

    llm = get_llm(provider, model, temperature=0)
    prompt_strategy = Jinja2FilePromptStrategy(prompt_template)

    try:
        init_tracing(
            f"{sections_file.stem}/{model} - {prompt_strategy.file_name} ({random.randint(0, 99999):05d})",
        )

        sections = json.loads(read_file(str(sections_file)))

        extraction_df = execute_agent(
            sections,
            llm=llm,
            grave_extraction_prompt_strategy=prompt_strategy,
        )

        eval_df = transform_df_for_evaluation(extraction_df)

        output_path = output_dir / f"{model} - {prompt_strategy.file_name}.csv"
        eval_df.to_csv(output_path)
        typer.echo(f"Grave extraction complete -> {output_path}")

    except KeyboardInterrupt:
        typer.echo("Shutting down process")


if __name__ == "__main__":
    app()
