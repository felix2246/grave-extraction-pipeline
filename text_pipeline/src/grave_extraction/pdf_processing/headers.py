import json
import re
from abc import ABC, abstractmethod
from typing import Optional

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from grave_extraction.logger import logger
from grave_extraction.models import Header
from grave_extraction.utils import timed


class HeadersExtractionStrategy(ABC):
    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path

    @abstractmethod
    def extract_headers(self) -> list[Header]:
        raise NotImplementedError()


class HeadersExtractionWithMarkerStrategy(HeadersExtractionStrategy):
    @timed
    def extract_headers(self) -> list[Header]:
        headers: list[Header] = []

        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        document = converter.build_document(self.pdf_path)
        toc = document.table_of_contents

        i = 0
        for t in toc:
            if t["title"].strip() == "":
                continue

            headers.append(
                Header(
                    id=i,
                    header_text=t["title"].strip().replace("\n", " "),
                    page_id=t["page_id"],
                    polygon=t["polygon"],
                    heading_level=t["heading_level"],
                )
            )
            i += 1

        if headers[0]["page_id"] != 0:
            raise ValueError(
                "Headers start with page_id of 1. Indizes should be 0-based!"
            )

        return headers


class HeadersExtractionFromSavedFileStrategy(HeadersExtractionStrategy):
    def __init__(
        self, saved_headers_file_path: Optional[str] = "logs/extracted_headers.json"
    ) -> None:
        self.saved_headers_file_path = saved_headers_file_path

    def extract_headers(self) -> list[Header]:
        logger.info("Using %s", self.saved_headers_file_path)

        if not self.saved_headers_file_path:
            raise ValueError("No filepath set")

        with open(self.saved_headers_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data[0]["page_id"] != 0:
            raise ValueError(
                "Headers start with page_id of 1. Indizes should be 0-based!"
            )

        return data


class HeaderExtractionWithKMeansClusteringStrategy(HeadersExtractionStrategy):
    def __init__(self, pdf_path: str, plot_output_path: Optional[str] = None) -> None:
        super().__init__(pdf_path)
        self.plot_output_path = plot_output_path

    @timed
    def extract_headers(self) -> list[Header]:
        raw_lines = self._extract_raw_lines()
        if not raw_lines:
            raise Exception("No text lines found")

        logical_blocks = self._reconstruct_logical_blocks(raw_lines)

        # Identifies headers and optionally plots the PCA result
        headers = self._identify_headers_with_baseline(
            logical_blocks,
            debug_plot=self.plot_output_path is not None,
            plot_path=self.plot_output_path,
        )

        if headers and headers[0]["page_id"] != 0:
            pass

        return headers

    def _extract_raw_lines(self) -> list[dict]:
        """Extracts all individual lines with their core style info."""
        doc = fitz.open(self.pdf_path)
        lines = []
        for page_num, page in enumerate(doc):  # type: ignore
            page_height = page.rect.height
            page_blocks_raw = page.get_text("dict")["blocks"]
            for block in page_blocks_raw:
                if block["type"] != 0 or "lines" not in block:
                    continue
                for line in block["lines"]:
                    if not line["spans"]:
                        continue

                    text = " ".join(s["text"].strip() for s in line["spans"]).strip()
                    if not text:
                        continue

                    bbox = line["bbox"]
                    if bbox[3] > page_height * 0.9 and text.isdigit() and len(text) < 5:
                        continue

                    span = line["spans"][0]
                    flags = span.get("flags", 0)
                    is_bold = ((flags & 16) != 0) or ("bold" in span["font"].lower())

                    lines.append(
                        {
                            "text": text,
                            "bbox": bbox,
                            "font_size": span["size"],
                            "is_bold": is_bold,
                            "page_number": page_num,
                        }
                    )
        doc.close()
        return lines

    def _reconstruct_logical_blocks(self, lines: list[dict]) -> list[dict]:
        """Reconstructs logical paragraphs from lines."""
        logical_blocks: list[dict] = []
        if not lines:
            return logical_blocks

        current_block_lines = [lines[0]]

        for i in range(1, len(lines)):
            prev_line = lines[i - 1]
            current_line = lines[i]

            vertical_gap = current_line["bbox"][1] - prev_line["bbox"][3]
            style_changed = (
                abs(current_line["font_size"] - prev_line["font_size"]) > 0.1
                or current_line["is_bold"] != prev_line["is_bold"]
            )
            is_new_page = current_line["page_number"] != prev_line["page_number"]

            if (
                is_new_page
                or vertical_gap > prev_line["font_size"] * 0.8
                or style_changed
            ):
                full_text = " ".join(l["text"] for l in current_block_lines)
                x0 = min(l["bbox"][0] for l in current_block_lines)
                y0 = min(l["bbox"][1] for l in current_block_lines)
                x1 = max(l["bbox"][2] for l in current_block_lines)
                y1 = max(l["bbox"][3] for l in current_block_lines)

                logical_blocks.append(
                    {
                        "text": full_text,
                        "text_length": len(full_text),
                        "page_number": current_block_lines[0]["page_number"],
                        "bbox": (x0, y0, x1, y1),
                        "avg_font_size": np.mean(
                            [l["font_size"] for l in current_block_lines]
                        ),
                        "bold_ratio": np.mean(
                            [l["is_bold"] for l in current_block_lines]
                        ),
                    }
                )
                current_block_lines = [current_line]
            else:
                current_block_lines.append(current_line)

        if current_block_lines:
            full_text = " ".join(l["text"] for l in current_block_lines)
            x0 = min(l["bbox"][0] for l in current_block_lines)
            y0 = min(l["bbox"][1] for l in current_block_lines)
            x1 = max(l["bbox"][2] for l in current_block_lines)
            y1 = max(l["bbox"][3] for l in current_block_lines)
            logical_blocks.append(
                {
                    "text": full_text,
                    "text_length": len(full_text),
                    "page_number": current_block_lines[0]["page_number"],
                    "bbox": (x0, y0, x1, y1),
                    "avg_font_size": np.mean(
                        [l["font_size"] for l in current_block_lines]
                    ),
                    "bold_ratio": np.mean([l["is_bold"] for l in current_block_lines]),
                }
            )

        return logical_blocks

    def _identify_headers_with_baseline(
        self,
        blocks: list[dict],
        max_styles: int = 10,
        debug_plot: bool = False,
        plot_path: Optional[str] = None,
    ) -> list[Header]:
        """
        Identifies headers using K-Means Clustering and visualizes with PCA.
        """
        if len(blocks) < 2:
            return []

        # Feature Engineering
        for block in blocks:
            letters = re.findall(r"[A-Za-z]", block["text"])
            block["uppercase_ratio"] = (
                sum(1 for c in letters if c.isupper()) / len(letters)
                if letters
                else 0.0
            )

        features = np.array(
            [[b["avg_font_size"], b["bold_ratio"], b["text_length"]] for b in blocks]
        )

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # k-Means Optimization
        best_kmeans = None
        best_score = -1
        num_components = min(max_styles, len(blocks))

        if num_components < 2:
            return []

        for n in range(2, num_components + 1):
            try:
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(
                    scaled_features
                )
                labels = kmeans.labels_
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(scaled_features, labels)
                if score > best_score:
                    best_score = score
                    best_kmeans = kmeans
            except ValueError:
                continue

        if best_kmeans is None:
            return []

        labels = best_kmeans.labels_
        n_clusters = best_kmeans.n_clusters

        # calculate Cluster Statistics
        cluster_stats = {}
        for i in range(n_clusters):
            indices = np.where(labels == i)[0]
            if len(indices) == 0:
                continue
            cluster_stats[i] = {
                "count": len(indices),
                "font_size": np.median([blocks[j]["avg_font_size"] for j in indices]),
                "bold_ratio": np.median([blocks[j]["bold_ratio"] for j in indices]),
                "text_length": np.median([blocks[j]["text_length"] for j in indices]),
                "uppercase_ratio": np.median(
                    [blocks[j]["uppercase_ratio"] for j in indices]
                ),
            }

        # identify Body Text
        body_cluster_label = max(
            cluster_stats.items(),
            key=lambda item: (
                item[1]["count"]
                * (item[1]["text_length"] if item[1]["text_length"] > 50 else 0)
            ),
        )[0]
        body_baseline = cluster_stats[body_cluster_label]

        # classify Clusters
        cluster_status = {}
        for cid, stats in cluster_stats.items():
            if cid == body_cluster_label:
                cluster_status[cid] = "Body Text"
                continue

            is_larger = stats["font_size"] > body_baseline["font_size"] + 0.5
            is_bolder = stats["bold_ratio"] > body_baseline["bold_ratio"] + 0.4
            is_all_caps = stats["uppercase_ratio"] > 0.8
            is_short_enough = stats["text_length"] < 300

            if is_short_enough and (is_larger or is_bolder or is_all_caps):
                cluster_status[cid] = "Header Candidate"
            else:
                cluster_status[cid] = "Rejected (Noise/Footer)"

        # visualization
        if debug_plot:
            self._plot_debug_clusters(
                blocks,
                scaled_features,
                labels,
                cluster_status,
                best_kmeans,
                plot_path=plot_path,
            )

        # extract Headers
        header_blocks = []
        for i, block in enumerate(blocks):
            cid = labels[i]
            if cluster_status[cid] == "Header Candidate":
                header_blocks.append(block)

        header_blocks.sort(key=lambda b: (b["page_number"], b["bbox"][1]))

        formatted_headers: list[Header] = []
        for i, block in enumerate(header_blocks):
            x0, y0, x1, y1 = block["bbox"]
            polygon = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            header = Header(
                id=i + 1,
                header_text=block["text"].strip().replace("\n", " "),
                page_id=block["page_number"],
                heading_level=None,
                polygon=polygon,
            )
            formatted_headers.append(header)

        return formatted_headers

    def _plot_debug_clusters(
        self,
        blocks,
        scaled_features,
        labels,
        cluster_status,
        kmeans,
        plot_path: Optional[str] = None,
    ):
        """
        Plots clusters using PCA.
        - Markers: Circles, no outlines.
        - Labels: Placed algorithmically in 'free space' (gaps) to avoid overlapping data.
        """
        print("Calculating PCA projection and searching for label positions...")

        # pca
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(scaled_features)

        # df setup
        df = pd.DataFrame(coords, columns=["x", "y"])
        df["Cluster ID"] = labels
        df["Status"] = df["Cluster ID"].map(cluster_status)

        # find representatives
        dist_matrix = kmeans.transform(scaled_features)
        representatives = {}
        unique_clusters = np.unique(labels)

        for cid in unique_clusters:
            cluster_indices = np.where(labels == cid)[0]
            distances_to_center = dist_matrix[cluster_indices, cid]
            closest_local_indices = np.argsort(distances_to_center)[:3]
            best_global_indices = cluster_indices[closest_local_indices]

            snippets = []
            for idx in best_global_indices:
                text = blocks[idx]["text"].replace("\n", " ")
                if len(text) > 30:
                    text = text[:27] + "..."
                snippets.append(text)
            representatives[cid] = snippets

        plt.figure(figsize=(15, 11))
        sns.set_theme(style="whitegrid")
        size_map = {
            "Header Candidate": 180,
            "Rejected (Noise/Footer)": 70,
            "Body Text": 30,
        }
        df["Size"] = df["Status"].map(size_map).fillna(50)
        unique_cluster_ids = sorted(df["Cluster ID"].unique())
        palette = sns.color_palette("tab10", n_colors=len(unique_cluster_ids))
        cluster_colors = {cid: palette[i] for i, cid in enumerate(unique_cluster_ids)}

        all_points = df[["x", "y"]].values

        for cid in unique_cluster_ids:
            subset = df[df["Cluster ID"] == cid]
            if subset.empty:
                continue

            status = subset["Status"].iloc[0]
            alpha = 0.3 if status == "Body Text" else 0.8

            plt.scatter(
                subset["x"],
                subset["y"],
                color=cluster_colors[cid],
                s=subset["Size"],
                alpha=alpha,
                label=f"Cluster {cid}",
                marker="o",
                edgecolors="none",
                linewidths=0,
                zorder=2 if status != "Body Text" else 1,
            )

        # Define boundaries with slight padding
        x_min, x_max = df["x"].min(), df["x"].max()
        y_min, y_max = df["y"].min(), df["y"].max()

        # Create a grid of candidate positions (e.g., 10x10 grid)
        # We will test these spots to see if they are "empty"
        grid_x = np.linspace(x_min, x_max, 10)
        grid_y = np.linspace(y_min, y_max, 10)
        candidate_positions = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)

        placed_labels: list[np.ndarray] = []  # Stores (x, y) of placed boxes

        for cid in unique_clusters:
            cluster_data = df[df["Cluster ID"] == cid]
            if cluster_data.empty:
                continue

            # Cluster Center
            center_x = cluster_data["x"].mean()
            center_y = cluster_data["y"].mean()
            cluster_center = np.array([center_x, center_y])

            # SCORING SYSTEM
            best_pos = None
            best_score = -np.inf  # We want to maximize score

            for candidate in candidate_positions:
                cand_x, cand_y = candidate

                # Metric 1: Distance to THIS cluster center (Closer is better)
                # We don't want the label to be on the other side of the map
                dist_to_self = np.linalg.norm(candidate - cluster_center)

                # Metric 2: Distance to NEAREST data point (Farther is better - finding holes)
                # Vectorized distance calculation to all points
                dists_to_all_points = np.linalg.norm(all_points - candidate, axis=1)
                dist_to_nearest_point = np.min(dists_to_all_points)

                # Metric 3: Distance to ALREADY PLACED labels (Farther is better)
                dist_to_nearest_label: float = 9999
                if placed_labels:
                    dists_labels = [
                        np.linalg.norm(candidate - np.array(p)) for p in placed_labels
                    ]
                    dist_to_nearest_label = float(min(dists_labels))

                # Heuristic Score calculation:
                # 1. Heavily reward being in a "hole" (dist_to_nearest_point)
                # 2. Heavily reward staying away from other labels
                # 3. Penalize being too far from the cluster center

                # Tuning weights:
                w_hole = 3.0  # Priority: Find empty space
                w_label = 5.0  # Priority: Don't overlap labels
                w_prox = 1.0  # Priority: Stay relatively close to cluster

                score = (
                    (w_hole * dist_to_nearest_point)
                    + (w_label * min(dist_to_nearest_label, 5))
                    - (w_prox * dist_to_self)
                )

                if score > best_score:
                    best_score = score
                    best_pos = candidate

            if best_pos is None:
                continue
            text_x, text_y = best_pos
            placed_labels.append(best_pos)

            # Draw Annotation
            snippet_text = "\n".join([f"- {s}" for s in representatives[cid]])
            label_text = f"Cluster {cid}\n{snippet_text}"

            rad = 0.2
            if text_x > center_x:
                rad = -0.2

            plt.annotate(
                label_text,
                xy=(center_x, center_y),
                xytext=(text_x, text_y),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="white",
                    ec=cluster_colors[cid],
                    alpha=0.9,
                    lw=1.5,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle=f"arc3,rad={rad}",
                    color="gray",
                    lw=1,
                ),
                zorder=10,
            )

        plt.title("Textblöcke in ihren Clustern", fontsize=16)
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        plt.tight_layout()

        plt.savefig(plot_path or "cluster_plot.png")
