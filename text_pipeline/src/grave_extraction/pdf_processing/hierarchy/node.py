from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HierarchyNode:
    """
    A unified tree node representation used internally by strategies to build the tree before flattening it back to the Header list.
    """

    id: int
    header_text: str
    level: int
    children: list["HierarchyNode"] = field(default_factory=list)
    numbering: Optional[str] = None

    def add_child(self, node: "HierarchyNode"):
        self.children.append(node)
