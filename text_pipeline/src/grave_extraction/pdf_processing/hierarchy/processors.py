from __future__ import annotations

import json
from abc import ABC, abstractmethod

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from grave_extraction.logger import logger
from grave_extraction.models import Header
from grave_extraction.prompt_strategy import PromptStrategy

from .node import HierarchyNode
from .utils import assign_levels_from_cluster_labels


class UnstructuredBlockProcessor(ABC):
    """
    Interface for processing 'blocks' of headers found in between explicitly numbered headers (the 'gaps').
    """

    @abstractmethod
    def process(self, headers: list[Header], parent_level: int) -> list[HierarchyNode]:
        """
        Args:
            headers: A list of contiguous unnumbered headers.
            parent_level: The level of the immediately preceding numbered header.
                          (-1 if at the start of the doc).
        Returns:
            A list of HierarchyNodes (potentially with children) relative to the parent.
        """
        raise NotImplementedError()


class SimpleIndentProcessor(UnstructuredBlockProcessor):
    """
    Simply indents all unnumbered headers by 1 relative to the parent.
    """

    def process(self, headers: list[Header], parent_level: int) -> list[HierarchyNode]:
        target_level = max(0, parent_level + 1)
        return [
            HierarchyNode(id=h["id"], header_text=h["header_text"], level=target_level)
            for h in headers
        ]


class LLMRecursiveProcessor(UnstructuredBlockProcessor):
    """
    Uses an LLM to determine the sub-hierarchy of the unnumbered block.
    """

    def __init__(self, llm: BaseChatModel, prompt_strategy: PromptStrategy):
        self.llm = llm
        self.prompt_strategy = prompt_strategy

    def process(self, headers: list[Header], parent_level: int) -> list[HierarchyNode]:
        if not headers:
            return []

        logger.info(f"Processing unstructured block of size {len(headers)} with LLM")

        # Internal Pydantic models for Structured Output
        class SemanticHeading(BaseModel):
            header_id: int = Field(
                ..., description="The exact, original id of the header."
            )
            children: list["SemanticHeading"] = Field(default_factory=list)

        class SemanticHierarchy(BaseModel):
            headings: list[SemanticHeading] = Field(
                ..., description="Root list of headings."
            )

        structured_llm = self.llm.with_structured_output(SemanticHierarchy).with_retry()

        header_txt = "\n".join([f"[{h['id']}][{h['header_text']}]" for h in headers])
        prompt = self.prompt_strategy.format(header_list=header_txt)

        try:
            res = structured_llm.invoke(prompt)
        except Exception as e:
            logger.error(f"LLM Hierarchy extraction failed: {e}. Fallback to flat.")
            return SimpleIndentProcessor().process(headers, parent_level)

        headers_map = {h["id"]: h for h in headers}

        def convert(
            semantic_nodes: list[SemanticHeading], lvl: int
        ) -> list[HierarchyNode]:
            nodes = []
            for node in semantic_nodes:
                if node.header_id in headers_map:
                    orig = headers_map[node.header_id]
                    h_node = HierarchyNode(
                        id=orig["id"], header_text=orig["header_text"], level=lvl
                    )
                    h_node.children = convert(node.children, lvl + 1)
                    nodes.append(h_node)
            return nodes

        # The LLM returns roots relative to the block. We attach them at parent_level + 1
        return convert(res.headings, parent_level + 1)  # type: ignore[union-attr]


class AHCProcessor(UnstructuredBlockProcessor):
    """
    Uses Agglomerative Hierarchical Clustering on embeddings to infer levels.
    Automatically determines the optimal number of clusters (levels)
    using Silhouette Score optimization.
    """

    def __init__(self, embeddings_model: Embeddings, max_search_clusters: int = 5):
        """
        Args:
            max_search_clusters: The maximum number of indentation levels
                                 to attempt finding in a single block.
                                 Defaults to 5 (rarely are blocks deeper than that).
        """
        self.embeddings_model = embeddings_model
        self.max_search_clusters = max_search_clusters

    def _find_optimal_clusters(self, embeddings: np.ndarray) -> list[int]:
        """
        Runs AHC with n_clusters ranging from 2 to max_search_clusters.
        Returns the labels of the best configuration based on Silhouette Score.
        """
        embeddings = normalize(embeddings)
        n_samples = embeddings.shape[0]

        # If we have very few headers, we can't meaningfully cluster or calculate silhouette
        if n_samples < 3:
            # Default to 1 cluster (all same level)
            return [0] * n_samples

        best_score = -1.0
        best_labels = [0] * n_samples

        # We try to find between 2 and (Samples-1) clusters, capped by max_search_clusters e.g. if 4 headers, try k=2, k=3.
        possible_k = range(2, min(n_samples, self.max_search_clusters + 1))

        found_optimal = False
        for k in possible_k:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(embeddings)

            try:
                score = silhouette_score(embeddings, labels)

                # We prefer simpler structures (fewer clusters) if scores are very similar, but strictly we take the max.
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    found_optimal = True
            except Exception as e:
                logger.warning(
                    f"Clustering silhouette calculation failed for k={k}: {e}"
                )
                continue

        if not found_optimal:
            # Fallback: If data is too uniform to cluster, return all zeros
            return [0] * n_samples

        return best_labels.tolist()  # type: ignore

    def process(self, headers: list[Header], parent_level: int) -> list[HierarchyNode]:
        if not headers:
            return []

        logger.info("Running AHC for %d headers", len(headers), headers=headers)

        embeddings = np.array(
            self.embeddings_model.embed_documents([h["header_text"] for h in headers])
        )

        labels = self._find_optimal_clusters(embeddings)
        logger.info("Found k=%d as optimal clusters", len(np.unique(labels)))

        relative_levels = assign_levels_from_cluster_labels(labels)

        nodes = []
        base_level = parent_level + 1

        for i, h in enumerate(headers):
            nodes.append(
                HierarchyNode(
                    id=h["id"],
                    header_text=h["header_text"],
                    level=base_level + relative_levels[i],
                )
            )

        logger.info(json.dumps([n.__dict__ for n in nodes], indent=2))

        return nodes
