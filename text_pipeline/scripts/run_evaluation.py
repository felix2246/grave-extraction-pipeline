"""
Evaluates extracted grave records against a ground truth CSV.
"""

from pathlib import Path

import typer

from grave_extraction.evaluation.grave_metrics import compare_graves

app = typer.Typer()


@app.command()
def main(
    extracted_path: Path = typer.Option(
        ...,
        "--extracted-path",
        help="Path to the extracted records CSV (must be in eval format).",
    ),
    gt_path: Path = typer.Option(
        ...,
        "--gt-path",
        help="Path to the ground truth CSV.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/evaluations"),
        "--output-dir",
        help="Directory for the output matrix CSV and summary JSON.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = extracted_path.stem
    matrix_path = output_dir / f"matrix_{stem}.csv"
    summary_path = output_dir / f"summary_{stem}.json"

    compare_graves(
        str(gt_path),
        str(extracted_path),
        str(matrix_path),
        str(summary_path),
    )

    typer.echo(f"Evaluation complete -> {matrix_path} / {summary_path}")


if __name__ == "__main__":
    app()
