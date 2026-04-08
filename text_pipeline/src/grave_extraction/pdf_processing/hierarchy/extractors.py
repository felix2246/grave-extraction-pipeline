import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from json_repair import repair_json
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from grave_extraction.logger import logger
from grave_extraction.models import Header
from grave_extraction.prompt_strategy import PromptStrategy
from grave_extraction.tracing import init_tracing
from grave_extraction.utils import timed

from .node import HierarchyNode
from .processors import SimpleIndentProcessor, UnstructuredBlockProcessor
from .utils import NumberingParser


class HierarchyExtractionStrategy(ABC):
    """
    The main interface for any logic that takes a list of flat headers and assigns them hierarchy levels.
    """

    @abstractmethod
    def extract_hierarchy(self, headers: list[Header]) -> list[Header]:
        raise NotImplementedError()


class HierarchyExtractionFromSavedFileStrategy(HierarchyExtractionStrategy):
    def __init__(
        self,
        saved_hierarchy_file_path: Optional[
            str
        ] = "outputs/logs/extracted_headers_with_hierarchy.json",
    ) -> None:
        self.saved_hierarchy_file_path = saved_hierarchy_file_path

    def extract_hierarchy(self, headers: list[Header] = []) -> list[Header]:
        logger.info("Using %s", self.saved_hierarchy_file_path)

        with open(self.saved_hierarchy_file_path, "r", encoding="utf-8") as f:  # type: ignore
            saved_data = json.load(f)
        return saved_data


class NumberedPatternParsingHierarchyExtractionStrategy(HierarchyExtractionStrategy):
    """
    1. Identifies explicit Numbering (1.1, 1.2) as the 'Backbone'.
    2. Uses a Tuple-Key Map to strictly enforce lineage (fixing the 3.1 under 2. issue).
    3. Delegates unnumbered 'blocks' to a configurable UnstructuredBlockProcessor.
    """

    def __init__(
        self,
        unstructured_processor: Optional[
            UnstructuredBlockProcessor
        ] = SimpleIndentProcessor(),
    ) -> None:
        self.unstructured_processor = unstructured_processor

    @timed
    def extract_hierarchy(self, headers: list[Header]) -> list[Header]:
        root_headings: list[HierarchyNode] = []
        node_map: dict[tuple[int, ...], HierarchyNode] = {}

        unstructured_buffer: list[Header] = []
        headers_by_id = {h["id"]: h for h in headers}
        current_active_node: Optional[HierarchyNode] = None

        def _get_numbering_key(number_str: str) -> tuple[int, ...]:
            clean = number_str.strip().rstrip(".")
            try:
                return tuple(int(x) for x in clean.split("."))
            except ValueError:
                return tuple(clean.split("."))  # type: ignore

        def _flush_buffer():
            if not unstructured_buffer:
                return
            # Determine parent level for the processor using the structural level
            parent_level = current_active_node.level if current_active_node else -1

            processed_nodes = self.unstructured_processor.process(
                unstructured_buffer, parent_level
            )

            if current_active_node:
                current_active_node.children.extend(processed_nodes)
            else:
                root_headings.extend(processed_nodes)

            unstructured_buffer.clear()

        for header in headers:
            parsed = NumberingParser.parse(header)

            if parsed:
                _flush_buffer()

                number_str, level = parsed
                current_key = _get_numbering_key(number_str)

                new_node = HierarchyNode(
                    id=header["id"],
                    header_text=header["header_text"],
                    # We store the parsed level for reference,
                    # but we won't use it for the final structure assignment.
                    level=level,
                )

                node_map[current_key] = new_node

                parent_key = current_key[:-1]
                parent_node = node_map.get(parent_key)

                if parent_node:
                    parent_node.add_child(new_node)
                else:
                    # If 3. is missing, 3.1 goes to Root.
                    root_headings.append(new_node)

                current_active_node = new_node

            else:
                unstructured_buffer.append(header)

        _flush_buffer()

        # Instead of trusting node.level (which says 3.1 is Level 2),
        # we calculate the level based on where it actually ended up in the tree.
        def _apply_structural_levels(nodes: list[HierarchyNode], depth: int = 1):
            for node in nodes:
                if node.id in headers_by_id:
                    # Force the level to match the Tree Structure
                    headers_by_id[node.id]["heading_level"] = depth

                # Recurse with depth + 1
                _apply_structural_levels(node.children, depth + 1)

        _apply_structural_levels(root_headings, depth=1)

        return headers


class HierarchyExtractionWithLLMStrategy(HierarchyExtractionStrategy):
    """
    Pure LLM strategy. Sends the whole list to LLM to tag levels.
    """

    def __init__(self, llm: BaseChatModel, prompt_strategy: PromptStrategy) -> None:
        self.llm = llm
        self.prompt_strategy = prompt_strategy
        init_tracing(self.__class__.__name__ + datetime.now().isoformat())

    @timed
    def extract_hierarchy(self, headers: list[Header]) -> list[Header]:
        logger.info("Extracting Hierarchy with pure LLM Strategy")

        class HierarchyHeader(BaseModel):
            id: int
            heading_level: int

        prompt = self.prompt_strategy.format(
            header_list=(
                "\n".join([f"[{h['id']}][{h['header_text']}]" for h in headers])
            )
        )

        res = self.llm.with_retry().invoke(prompt)
        json_str = repair_json(str(res.content))
        data = json.loads(json_str)
        validated = [HierarchyHeader(**item) for item in data]

        id_to_level = {h.id: h.heading_level for h in validated}
        for h in headers:
            if h["id"] in id_to_level:
                h["heading_level"] = id_to_level[h["id"]]

        return headers
