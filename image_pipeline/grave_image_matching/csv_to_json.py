"""Transform grave_matches CSV (with filepaths) to structured JSON."""

import ast
import json
import re
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Transform grave_matches CSV to structured JSON.")

BESTATTUNG_COLS = [
    "bestattung_geschlecht",
    "bestattung_alter",
    "bestattung_ritus",
    "bestattung_lage",
    "bestattung_orientierung",
    "bestattung_störung",
]
BESTATTUNG_KEYS = [c.replace("bestattung_", "") for c in BESTATTUNG_COLS]
BEIGABEN_PATTERN = re.compile(r"^(\d+)x(.+)$")


def _safe_literal_eval_list(s):
    """Parse Python list literal string to list; return [] on failure or empty."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if not s or s == "[]":
        return []
    try:
        out = ast.literal_eval(s)
        return list(out) if isinstance(out, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _to_json_val(x):
    """Convert scalar for JSON: NaN/None -> None, keep numbers and str."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str) and x.strip() in ("", "None", "nan"):
        return None
    if isinstance(x, (int, float)):
        return x
    # pandas/numpy numeric types
    try:
        if hasattr(x, "item"):
            return x.item()
    except (ValueError, AttributeError):
        pass
    s = str(x).strip()
    if s in ("None", "nan"):
        return None
    return s


def _to_numeric(x):
    """Coerce to int or float for JSON; None/NaN -> None. Whole numbers become int."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        if hasattr(x, "item"):
            v = x.item()
        else:
            v = x
        if isinstance(v, float):
            if pd.isna(v):
                return None
            return int(v) if v == int(v) else v
        return int(v)
    except (ValueError, TypeError, AttributeError):
        return None


def _parse_bestattung_cell(val):
    """Return either a single value (for one burial) or a tuple of values (for multiple)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return (None,)
    s = str(val).strip()
    if not s or s == "None":
        return (None,)
    if s.startswith("(") and ")" in s:
        try:
            t = ast.literal_eval(s)
            if isinstance(t, tuple):
                return tuple(None if v is None else str(v) for v in t)
            return (str(t),)
        except (ValueError, SyntaxError):
            return (s,)
    return (s,)


def _build_bestattungen(row):
    """Build list of bestattung objects from the six bestattung_ columns."""
    parsed = [_parse_bestattung_cell(row[c]) for c in BESTATTUNG_COLS]
    n = max(len(p) for p in parsed)
    # Normalize so all have length n (pad with None)
    normalized = [tuple(list(p) + [None] * (n - len(p))) for p in parsed]
    bestattungen = []
    for i in range(n):
        obj = {}
        for j, key in enumerate(BESTATTUNG_KEYS):
            v = normalized[j][i]
            obj[key] = None if v is None or v == "None" else v
        bestattungen.append(obj)
    return bestattungen


def _parse_beigaben_liste(s):
    """Parse beigaben_liste string to list of {name, amount}."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    items = [x.strip() for x in s.split(",") if x.strip()]
    result = []
    for item in items:
        m = BEIGABEN_PATTERN.match(item)
        if m:
            result.append({"name": m.group(2).strip(), "amount": int(m.group(1))})
        else:
            result.append({"name": item, "amount": 1})
    return result


def row_to_grave_obj(row):
    """Convert one CSV row to the grave JSON object."""
    bestattungen = _build_bestattungen(row)

    refs = _safe_literal_eval_list(row.get("referenzierte_abbildungen"))
    matched = _safe_literal_eval_list(row.get("matched_filenames"))
    # Ensure matched_images are plain strings
    matched_images = [str(p) for p in matched]

    return {
        "page_id": _to_numeric(row.get("page_id")),
        "section_title": _to_json_val(row.get("section_title")),
        "fundort": _to_json_val(row.get("fundort")),
        "bezirk": _to_json_val(row.get("bezirk")),
        "grab_identifikation": _to_json_val(row.get("grab_identifikation")),
        "bestattungen": bestattungen,
        "grabeinbauten": _to_json_val(row.get("grabeinbauten")),
        "grube_form": _to_json_val(row.get("grube_form")),
        "grube_länge": _to_numeric(row.get("grube_länge")),
        "grube_breite": _to_numeric(row.get("grube_breite")),
        "grube_tiefe": _to_numeric(row.get("grube_tiefe")),
        "beigaben_liste": _parse_beigaben_liste(row.get("beigaben_liste")),
        "referenzierte_abbildungen": refs,
        "matched_images": matched_images,
    }


def csv_to_json(csv_path: Path, json_path: Path) -> None:
    """Read CSV and write JSON array of grave objects."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    out = [row_to_grave_obj(df.loc[i]) for i in df.index]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    typer.echo(f"Wrote {len(out)} graves to {json_path}")


@app.command()
def main(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Input CSV file (grave_matches with filepaths).",
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Output JSON file.",
    ),
) -> None:
    csv_path = input_file.resolve()
    json_path = output_file.resolve()
    csv_to_json(csv_path, json_path)


if __name__ == "__main__":
    app()
