import json
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field


# carries general grave information for the next graves, often mentioned in introductory texts. Gets injected into the prompt
class StructuredGraveContext(BaseModel):
    fundort: Optional[str] = Field(
        description="Fundort oder Ort des Grabes", default=None
    )

    bezirk: Optional[str] = Field(description="Bezirk des Grabes", default=None)

    grabeinbauten: Optional[str] = Field(
        description="Ob das Grab innen mit bestimmten Materialien verkleidet ist. Wenn ja, gib das Material an.",
        default=None,
    )

    bestattung_ritus: Optional[str] = Field(
        description="Art der Bestattung z.B. Körperbestattung oder Brandbestattung.",
        default=None,
    )

    bestattung_störung: Optional[Literal["ja", "nein", "unsicher"]] = Field(
        description='Wurde die Bestattung gestört? "unsicher", wenn im Text als unsicher klassifiziert.',
        default=None,
    )


@dataclass
class ContextLayer:
    structured: StructuredGraveContext
    interpretive: list[str] = field(default_factory=list)
    level: int = 0  # Heading level at which this context applies


class ContextManager:
    """Manages hierarchical structured + interpretive context for extraction."""

    def __init__(self):
        self.stack: list[ContextLayer] = []

    def push(
        self, structured: StructuredGraveContext, interpretive: list[str] = [], level=0
    ):
        self.stack.append(ContextLayer(structured, interpretive, level))

    def pop_to_level(self, current_level: int):
        """
        Pops all context layers from the stack that are at or deeper than the given hierarchy level.
        """
        self.stack = [c for c in self.stack if c.level < current_level]

    def merged_structured(self) -> StructuredGraveContext:
        merged = {}
        for layer in self.stack:
            merged.update(
                layer.structured.model_dump(exclude_none=True, exclude_unset=True)
            )
        return StructuredGraveContext(**merged)

    def merged_interpretive(self) -> Optional[str]:
        notes = [n for layer in self.stack for n in layer.interpretive]
        return " ".join(set[str](notes)) if notes else None


def print_context_stack(cm: ContextManager, title: str):
    if not cm.stack:
        print(f"Context before '{title}': <empty>")
        return
    rows = []
    for idx, layer in enumerate[ContextLayer](cm.stack):
        s = layer.structured.model_dump(exclude_none=True, exclude_unset=True)
        rows.append(
            f"  level={layer.level} idx={idx} structured={json.dumps(s, ensure_ascii=False)} interpretive={layer.interpretive}"
        )
    print(f"Context before '{title}':\n" + "\n".join(rows))
