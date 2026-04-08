from dataclasses import dataclass, field
from unittest.mock import MagicMock

from grave_extraction.pdf_processing.hierarchy.extractors import (
    NumberedPatternParsingHierarchyExtractionStrategy,
)
from grave_extraction.pdf_processing.hierarchy.processors import LLMRecursiveProcessor


class TestNumberMatchingExtractor:
    def test_hierarchy_number_matching(self) -> None:
        strategy = NumberedPatternParsingHierarchyExtractionStrategy()

        p = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        headers = [
            {
                "id": 1,
                "header_text": "Abstract",
                "page_id": 1,
                "heading_level": None,
                "polygon": p,
            },
            {
                "id": 2,
                "header_text": "1. Intro",
                "page_id": 1,
                "heading_level": None,
                "polygon": p,
            },
            {
                "id": 3,
                "header_text": "1.1 Deep Dive",
                "page_id": 2,
                "heading_level": None,
                "polygon": p,
            },
            {
                "id": 4,
                "header_text": "Just Text",
                "page_id": 2,
                "heading_level": None,
                "polygon": p,
            },
            {
                "id": 5,
                "header_text": "1.1.1. Indent",
                "page_id": 3,
                "heading_level": None,
                "polygon": p,
            },
            {
                "id": 6,
                "header_text": "2. Chapter",
                "page_id": 3,
                "heading_level": None,
                "polygon": p,
            },
        ]

        results = strategy.extract_hierarchy(headers)

        # Abstract
        # 1. Intro
        #     1.1 Deep Dive
        #         Just Text
        #         1.1.1. Indent
        # 2. Chapter
        assert [h["heading_level"] for h in results] == [1, 1, 2, 3, 3, 1]


# A simple helper to mock the Pydantic structure returned by the LLM
@dataclass
class MockSemanticHeading:
    header_id: int
    header_text: str
    children: list["MockSemanticHeading"] = field(default_factory=list)


@dataclass
class MockSemanticHierarchy:
    headings: list[MockSemanticHeading]


class TestHybridLLMExtractor:
    def test_hybrid_hierarchy_extraction(self) -> None:
        p = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        headers = [
            {
                "id": 1,
                "header_text": "1. Introduction",
                "page_id": 1,
                "heading_level": None,
                "polygon": p,
            },
            # Block 2: Unnumbered (Buffer)
            # We want the LLM to structure these.
            {
                "id": 2,
                "header_text": "Context",
                "page_id": 1,
                "heading_level": None,
                "polygon": p,
            },
            {
                "id": 3,
                "header_text": "Problem Statement",
                "page_id": 1,
                "heading_level": None,
                "polygon": p,
            },
            {
                "id": 4,
                "header_text": "2. Methods",
                "page_id": 2,
                "heading_level": None,
                "polygon": p,
            },
        ]

        mock_llm = MagicMock()
        mock_prompt_strategy = MagicMock()

        mock_llm_output = MockSemanticHierarchy(
            headings=[
                MockSemanticHeading(
                    header_id=2,
                    header_text="Context",
                    children=[
                        MockSemanticHeading(
                            header_id=3, header_text="Problem Statement"
                        )
                    ],
                )
            ]
        )

        # Create a mock for the structured LLM chain
        structured_llm_mock = MagicMock()

        structured_llm_mock.with_retry.return_value = structured_llm_mock
        structured_llm_mock.invoke.return_value = mock_llm_output
        mock_llm.with_structured_output.return_value = structured_llm_mock

        strategy = NumberedPatternParsingHierarchyExtractionStrategy(
            unstructured_processor=LLMRecursiveProcessor(
                llm=mock_llm, prompt_strategy=mock_prompt_strategy
            )
        )

        results = strategy.extract_hierarchy(headers)

        # Expected Logic:
        # [ID 1] "1. Introduction" -> Level 0 (via Regex)
        # [ID 2] "Context" -> Level 1 (LLM said it's a child of the buffer parent.
        #                      Buffer parent is ID 1 (Level 0), so ID 2 becomes Level 1).
        # [ID 3] "Problem Statement" -> Level 2 (LLM said it's a child of Context).
        # [ID 4] "2. Methods" -> Level 0 (via Regex)

        assert results[0]["header_text"] == "1. Introduction"
        assert results[0]["heading_level"] == 1

        assert results[1]["header_text"] == "Context"
        assert results[1]["heading_level"] == 2

        assert results[2]["header_text"] == "Problem Statement"
        assert results[2]["heading_level"] == 3

        assert results[3]["header_text"] == "2. Methods"
        assert results[3]["heading_level"] == 1

        # Verify LLM was actually called
        assert mock_llm.with_structured_output.called
