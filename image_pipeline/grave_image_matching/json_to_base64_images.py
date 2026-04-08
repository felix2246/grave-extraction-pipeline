"""
Convert grave_matches JSON so that matched_images contain base64-encoded image data
instead of file paths.
"""

import base64
import json
from pathlib import Path

import typer

app = typer.Typer(help="Convert matched_images from file paths to base64 strings.")

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def image_path_to_base64(path: str, base_dir: Path) -> str | None:
    """Read image file and return data URL (data:image/...;base64,...), or None if file not found."""
    full_path = (base_dir / path).resolve()
    if not full_path.exists():
        return None
    try:
        data = full_path.read_bytes()
        b64 = base64.standard_b64encode(data).decode("ascii")
        mime = MIME_BY_EXT.get(full_path.suffix.lower(), "image/png")
        return f"data:{mime};base64,{b64}"
    except (OSError, ValueError):
        return None


def convert_matched_images_to_base64(
    data: list[dict],
    base_dir: Path,
) -> list[dict]:
    """Replace matched_images paths with base64 strings in a copy of data."""
    result = []
    for item in data:
        entry = dict(item)
        paths = entry.get("matched_images", [])
        if not paths:
            entry["matched_images"] = []
            result.append(entry)
            continue
        b64_list = []
        for path in paths:
            b64 = image_path_to_base64(path, base_dir)
            b64_list.append(b64 if b64 is not None else None)
        entry["matched_images"] = b64_list
        result.append(entry)
    return result


@app.command()
def main(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Input JSON file with matched_images paths.",
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Output JSON file (matched_images will be base64 strings).",
    ),
    base_dir: Path | None = typer.Option(
        None,
        "--base-dir",
        "-b",
        help="Base directory for resolving image paths (default: project root).",
    ),
) -> None:
    input_path = input_file.resolve()
    base_dir_resolved = (
        base_dir.resolve()
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    output_path = output_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        typer.echo("Expected JSON root to be a list of grave entries.", err=True)
        raise typer.Exit(1)

    converted = convert_matched_images_to_base64(data, base_dir_resolved)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    typer.echo(f"Wrote {len(converted)} entries to {output_path}")


if __name__ == "__main__":
    app()
