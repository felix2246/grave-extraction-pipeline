"""
Unit tests for the CLI scripts in scripts/.

All heavy operations (PDF processing, LLM calls, business-logic file I/O) are
mocked. Output files are written to pytest's tmp_path and are cleaned up
automatically.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from scripts.run_evaluation import app as evaluation_app
from scripts.run_grave_extraction import app as grave_extraction_app
from scripts.run_header_extraction import STRATEGIES as HEADER_STRATEGIES
from scripts.run_header_extraction import app as header_extraction_app
from scripts.run_hierarchy_extraction import app as hierarchy_extraction_app
from scripts.run_pipeline import app as pipeline_app
from scripts.run_section_builder import app as section_builder_app
from typer.testing import CliRunner

runner = CliRunner()

FAKE_HEADERS = [
    {
        "id": 0,
        "header_text": "1. INTRO",
        "page_id": 0,
        "heading_level": 1,
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
    },
    {
        "id": 1,
        "header_text": "1.1 Sub",
        "page_id": 0,
        "heading_level": 2,
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
    },
]

FAKE_SECTIONS = [
    {"id": 0, "title": "1. INTRO", "heading_level": 1, "page_id": 0, "text": "foo"},
]


@pytest.fixture
def headers_json(tmp_path: Path) -> Path:
    p = tmp_path / "headers.json"
    p.write_text(json.dumps(FAKE_HEADERS), encoding="utf-8")
    return p


@pytest.fixture
def hierarchy_json(tmp_path: Path) -> Path:
    p = tmp_path / "hierarchy.json"
    p.write_text(json.dumps(FAKE_HEADERS), encoding="utf-8")
    return p


@pytest.fixture
def sections_json(tmp_path: Path) -> Path:
    p = tmp_path / "sections.json"
    p.write_text(json.dumps(FAKE_SECTIONS), encoding="utf-8")
    return p


@pytest.fixture
def extracted_csv(tmp_path: Path) -> Path:
    p = tmp_path / "extracted.csv"
    pd.DataFrame({"grave_id": [1], "fundort": ["Testort"]}).to_csv(p)
    return p


@pytest.fixture
def gt_csv(tmp_path: Path) -> Path:
    p = tmp_path / "gt.csv"
    pd.DataFrame({"grave_id": [1], "fundort": ["Testort"]}).to_csv(p)
    return p


class TestRunHeaderExtraction:
    def test_unknown_strategy_exits_with_code_1(self, tmp_path: Path):
        result = runner.invoke(
            header_extraction_app,
            [
                "--strategy",
                "foobar",
                "--pdf-path",
                "dummy.pdf",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "Unknown strategy" in result.output

    def test_known_strategies_are_kmeans_and_marker(self):
        assert set(HEADER_STRATEGIES) == {"kmeans", "marker"}

    def test_kmeans_strategy_calls_correct_extractor_and_writes_files(
        self, tmp_path: Path
    ):
        with patch(
            "scripts.run_header_extraction.HeaderExtractionWithKMeansClusteringStrategy"
        ) as MockExtractor:
            MockExtractor.return_value.extract_headers.return_value = FAKE_HEADERS

            result = runner.invoke(
                header_extraction_app,
                [
                    "--pdf-path",
                    "data/dummy.pdf",
                    "--output-dir",
                    str(tmp_path),
                    "--strategy",
                    "kmeans",
                ],
            )

        assert result.exit_code == 0, result.output
        MockExtractor.assert_called_once_with(
            "data/dummy.pdf",
            plot_output_path=str(tmp_path / "cluster_plot.png"),
        )
        MockExtractor.return_value.extract_headers.assert_called_once()
        assert (tmp_path / "style_clustering_k_means.json").exists()
        assert (tmp_path / "style_clustering_k_means.txt").exists()
        assert f"Extracted {len(FAKE_HEADERS)} headers" in result.output

    def test_marker_strategy_calls_correct_extractor_and_writes_files(
        self, tmp_path: Path
    ):
        with patch(
            "scripts.run_header_extraction.HeadersExtractionWithMarkerStrategy"
        ) as MockExtractor:
            MockExtractor.return_value.extract_headers.return_value = FAKE_HEADERS

            result = runner.invoke(
                header_extraction_app,
                [
                    "--pdf-path",
                    "data/dummy.pdf",
                    "--output-dir",
                    str(tmp_path),
                    "--strategy",
                    "marker",
                ],
            )

        assert result.exit_code == 0, result.output
        MockExtractor.assert_called_once_with("data/dummy.pdf")
        MockExtractor.return_value.extract_headers.assert_called_once()
        assert (tmp_path / "marker.json").exists()
        assert (tmp_path / "marker.txt").exists()
        assert f"Extracted {len(FAKE_HEADERS)} headers" in result.output


class TestRunHierarchyExtraction:
    def test_missing_required_option_fails(self):
        result = runner.invoke(hierarchy_extraction_app, [])
        assert result.exit_code != 0

    def test_valid_invocation_extracts_and_writes_files(
        self, tmp_path: Path, headers_json: Path
    ):
        with (
            patch("scripts.run_hierarchy_extraction.get_llm") as mock_get_llm,
            patch(
                "scripts.run_hierarchy_extraction.Jinja2FilePromptStrategy"
            ) as MockPrompt,
            patch(
                "scripts.run_hierarchy_extraction.LLMRecursiveProcessor"
            ) as MockProcessor,
            patch(
                "scripts.run_hierarchy_extraction.NumberedPatternParsingHierarchyExtractionStrategy"
            ) as MockExtractor,
            patch("scripts.run_hierarchy_extraction.init_tracing"),
        ):
            mock_get_llm.return_value = MagicMock()
            MockPrompt.return_value = MagicMock()
            MockProcessor.return_value = MagicMock()
            MockExtractor.return_value.extract_hierarchy.return_value = FAKE_HEADERS

            result = runner.invoke(
                hierarchy_extraction_app,
                [
                    "--headers-file",
                    str(headers_json),
                    "--output-dir",
                    str(tmp_path),
                    "--run-name",
                    "test_run",
                ],
            )

        assert result.exit_code == 0, result.output
        MockExtractor.return_value.extract_hierarchy.assert_called_once_with(
            FAKE_HEADERS
        )
        assert (tmp_path / "test_run.json").exists()
        assert (tmp_path / "test_run.txt").exists()
        assert "Hierarchy extracted" in result.output


class TestRunSectionBuilder:
    def test_missing_required_options_fails(self):
        result = runner.invoke(section_builder_app, [])
        assert result.exit_code != 0

    def test_valid_invocation_builds_and_writes_sections(
        self, tmp_path: Path, hierarchy_json: Path
    ):
        output_file = tmp_path / "sections.json"

        with (
            patch(
                "scripts.run_section_builder.HierarchyExtractionFromSavedFileStrategy"
            ) as MockHierarchy,
            patch("scripts.run_section_builder.HybridSectionsBuilder") as MockBuilder,
        ):
            MockHierarchy.return_value.extract_hierarchy.return_value = FAKE_HEADERS
            MockBuilder.return_value.build_sections.return_value = FAKE_SECTIONS

            result = runner.invoke(
                section_builder_app,
                [
                    "--pdf-path",
                    "data/dummy.pdf",
                    "--hierarchy-file",
                    str(hierarchy_json),
                    "--output-file",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0, result.output
        MockHierarchy.assert_called_once_with(str(hierarchy_json))
        MockBuilder.return_value.build_sections.assert_called_once_with(
            "data/dummy.pdf", FAKE_HEADERS
        )
        assert output_file.exists()
        assert json.loads(output_file.read_text()) == FAKE_SECTIONS
        assert f"Built {len(FAKE_SECTIONS)} sections" in result.output


class TestRunGraveExtraction:
    def test_missing_required_option_fails(self):
        result = runner.invoke(grave_extraction_app, [])
        assert result.exit_code != 0

    def test_valid_invocation_runs_agent_and_writes_csv(
        self, tmp_path: Path, sections_json: Path
    ):
        eval_df = pd.DataFrame({"grave_id": [1], "fundort": ["Testort"]})

        with (
            patch("scripts.run_grave_extraction.get_llm") as mock_get_llm,
            patch(
                "scripts.run_grave_extraction.Jinja2FilePromptStrategy"
            ) as MockPrompt,
            patch("scripts.run_grave_extraction.execute_agent") as mock_execute,
            patch(
                "scripts.run_grave_extraction.transform_df_for_evaluation"
            ) as mock_transform,
            patch("scripts.run_grave_extraction.init_tracing"),
            patch(
                "scripts.run_grave_extraction.read_file",
                return_value=sections_json.read_text(),
            ),
        ):
            mock_get_llm.return_value = MagicMock()
            MockPrompt.return_value = MagicMock(file_name="test_prompt.jinja2")
            mock_execute.return_value = MagicMock()
            mock_transform.return_value = eval_df

            result = runner.invoke(
                grave_extraction_app,
                [
                    "--sections-file",
                    str(sections_json),
                    "--output-dir",
                    str(tmp_path),
                    "--model",
                    "test-model",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_execute.assert_called_once()
        mock_transform.assert_called_once()
        assert (tmp_path / "test-model - test_prompt.jinja2.csv").exists()
        assert "Grave extraction complete" in result.output


class TestRunEvaluation:
    def test_missing_required_options_fails(self):
        result = runner.invoke(evaluation_app, [])
        assert result.exit_code != 0

    def test_valid_invocation_calls_compare_graves_with_correct_paths(
        self, tmp_path: Path, extracted_csv: Path, gt_csv: Path
    ):
        with patch("scripts.run_evaluation.compare_graves") as mock_compare:
            result = runner.invoke(
                evaluation_app,
                [
                    "--extracted-path",
                    str(extracted_csv),
                    "--gt-path",
                    str(gt_csv),
                    "--output-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        mock_compare.assert_called_once_with(
            str(gt_csv),
            str(extracted_csv),
            str(tmp_path / f"matrix_{extracted_csv.stem}.csv"),
            str(tmp_path / f"summary_{extracted_csv.stem}.json"),
        )
        assert "Evaluation complete" in result.output


class TestRunPipeline:
    def test_valid_invocation_runs_all_stages_and_writes_csv(self, tmp_path: Path):
        eval_df = pd.DataFrame({"grave_id": [1], "fundort": ["Testort"]})

        with (
            patch(
                "scripts.run_pipeline.HeaderExtractionWithKMeansClusteringStrategy"
            ) as MockHeaders,
            patch(
                "scripts.run_pipeline.NumberedPatternParsingHierarchyExtractionStrategy"
            ) as MockHierarchy,
            patch("scripts.run_pipeline.LLMRecursiveProcessor") as MockProcessor,
            patch("scripts.run_pipeline.HybridSectionsBuilder") as MockSections,
            patch("scripts.run_pipeline.get_llm") as mock_get_llm,
            patch("scripts.run_pipeline.Jinja2FilePromptStrategy") as MockPrompt,
            patch("scripts.run_pipeline.execute_agent") as mock_execute,
            patch("scripts.run_pipeline.transform_df_for_evaluation") as mock_transform,
            patch("scripts.run_pipeline.init_tracing"),
        ):
            MockHeaders.return_value.extract_headers.return_value = FAKE_HEADERS
            MockHierarchy.return_value.extract_hierarchy.return_value = FAKE_HEADERS
            MockProcessor.return_value = MagicMock()
            MockSections.return_value.build_sections.return_value = FAKE_SECTIONS
            mock_get_llm.return_value = MagicMock()
            MockPrompt.return_value = MagicMock()
            mock_execute.return_value = MagicMock()
            mock_transform.return_value = eval_df

            result = runner.invoke(
                pipeline_app,
                [
                    "--pdf-path",
                    "data/dummy.pdf",
                    "--output-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        MockHeaders.return_value.extract_headers.assert_called_once()
        MockHierarchy.return_value.extract_hierarchy.assert_called_once_with(
            FAKE_HEADERS
        )
        MockSections.return_value.build_sections.assert_called_once()
        mock_execute.assert_called_once()
        mock_transform.assert_called_once()
        assert "Pipeline complete" in result.output
        csv_files = list(tmp_path.glob("full_pipeline_*.csv"))
        assert len(csv_files) == 1

    def test_pipeline_uses_selected_model_for_hierarchy_by_default(
        self, tmp_path: Path
    ):
        eval_df = pd.DataFrame({"grave_id": [1], "fundort": ["Testort"]})

        with (
            patch(
                "scripts.run_pipeline.HeaderExtractionWithKMeansClusteringStrategy"
            ) as MockHeaders,
            patch(
                "scripts.run_pipeline.NumberedPatternParsingHierarchyExtractionStrategy"
            ) as MockHierarchy,
            patch("scripts.run_pipeline.LLMRecursiveProcessor") as MockProcessor,
            patch("scripts.run_pipeline.HybridSectionsBuilder") as MockSections,
            patch("scripts.run_pipeline.get_llm") as mock_get_llm,
            patch("scripts.run_pipeline.Jinja2FilePromptStrategy") as MockPrompt,
            patch("scripts.run_pipeline.execute_agent") as mock_execute,
            patch("scripts.run_pipeline.transform_df_for_evaluation") as mock_transform,
            patch("scripts.run_pipeline.init_tracing"),
        ):
            MockHeaders.return_value.extract_headers.return_value = FAKE_HEADERS
            MockHierarchy.return_value.extract_hierarchy.return_value = FAKE_HEADERS
            MockProcessor.return_value = MagicMock()
            MockSections.return_value.build_sections.return_value = FAKE_SECTIONS
            mock_get_llm.return_value = MagicMock()
            MockPrompt.return_value = MagicMock()
            mock_execute.return_value = MagicMock()
            mock_transform.return_value = eval_df

            result = runner.invoke(
                pipeline_app,
                [
                    "--pdf-path",
                    "data/dummy.pdf",
                    "--output-dir",
                    str(tmp_path),
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 0, result.output
        assert mock_get_llm.call_count >= 2
        assert mock_get_llm.call_args_list[0].args == ("openai", "gpt-4o")
