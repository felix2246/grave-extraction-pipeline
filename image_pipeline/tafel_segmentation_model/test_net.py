"""
Evaluate and visualize a trained Detectron2 model on the test set.

- Expects test images/annotations in 'data/test/'
- Model weights in 'output/model_final.pth'
- Saves results to 'test_results_visuals/'

Run after training with train_net.py.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils  # type: ignore[import-untyped, import-not-found]
from detectron2 import model_zoo  # type: ignore[import-untyped, import-not-found]
from detectron2.config import get_cfg  # type: ignore[import-untyped, import-not-found]
from detectron2.data import (  # type: ignore[import-untyped, import-not-found]
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.data.datasets import (  # type: ignore[import-untyped, import-not-found]
    convert_to_coco_json,
)
from detectron2.engine import (  # type: ignore[import-untyped, import-not-found]
    DefaultPredictor,
)
from detectron2.evaluation import (  # type: ignore[import-untyped, import-not-found]
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.utils.visualizer import (  # type: ignore[import-untyped, import-not-found]
    ColorMode,
    Visualizer,
)
from pycocotools.coco import COCO  # type: ignore[import-untyped, import-not-found]
from pycocotools.cocoeval import (  # type: ignore[import-untyped, import-not-found]
    COCOeval,
)
from tafel_segmentation_model.lib import (  # type: ignore[import-untyped, import-not-found]
    get_box_dicts,
)

TEST_DATA_PATH = "experiments/segmentation/first-catalogue/data/test"
WEIGHTS_PATH = (
    "tafel_segmentation_model/output_train_only_first_catalogue/model_final.pth"
)
OUTPUT_VIS_DIR = "experiments/segmentation/first-catalogue/output/test_result_visuals"
OUTPUT_DIR = "experiments/segmentation/first-catalogue/output"

RAINBOW_COLORS = [
    (1.0, 0.0, 0.0),  # Red
    (1.0, 0.5, 0.0),  # Orange
    (1.0, 1.0, 0.0),  # Yellow
    (0.5, 1.0, 0.0),  # Lime
    (0.0, 1.0, 0.0),  # Green
    (0.0, 1.0, 1.0),  # Cyan
    (0.0, 0.5, 1.0),  # Azure/Sky Blue
    (0.0, 0.0, 1.0),  # Blue
    (0.5, 0.0, 1.0),  # Violet
    (1.0, 0.0, 1.0),  # Magenta
]


def plot_precision_recall(output_dir, dataset_name):
    """
    Plots PR curve and calculates AR50.
    """
    pred_json = os.path.join(output_dir, "coco_instances_results.json")

    if not os.path.exists(pred_json):
        print(f"Error: Prediction file not found at {pred_json}")
        return

    gt_json = os.path.join(output_dir, "test_ground_truth.json")

    if not os.path.exists(gt_json):
        print(f"Converting dataset '{dataset_name}' to COCO format for evaluation...")
        convert_to_coco_json(dataset_name, output_file=gt_json, allow_cached=True)

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    ar50 = coco_eval.eval["recall"][0, :, 0, 2].mean()
    mean_ar = coco_eval.eval["recall"][:, :, 0, 2].mean()

    print("-" * 40)
    print(f"Calculated AR@50 (IoU=0.50, maxDets=100): {ar50:.4f}")
    print(f"Calculated Mean AR (IoU=0.50:0.95, maxDets=100): {mean_ar:.4f}")
    print("-" * 40)

    # Plotting logic
    precision = coco_eval.eval["precision"]

    # IoU @ 0.50 is index 0
    pr_50 = precision[0, :, 0, 0, 2]
    # IoU @ 0.75 is index 5
    pr_75 = precision[5, :, 0, 0, 2]

    x_recall = np.linspace(0, 1, 101)

    plt.figure(figsize=(10, 7))
    plt.plot(x_recall, pr_50, label="IoU=0.50", linewidth=2)
    plt.plot(x_recall, pr_75, label="IoU=0.75", linewidth=2)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    save_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Robust PR Curve to: {save_path}")


def compute_f1_score(output_dir, dataset_name, iou_threshold=0.5):
    """Compute Precision, Recall, and F1-Score using mask IoU."""
    gt_json = os.path.join(output_dir, "test_ground_truth.json")
    pred_json = os.path.join(output_dir, "coco_instances_results.json")

    if not os.path.exists(pred_json):
        print(f"Error: Prediction file not found at {pred_json}")
        return None
    if not os.path.exists(gt_json):
        convert_to_coco_json(dataset_name, output_file=gt_json, allow_cached=True)

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    tp, fp, fn = 0, 0, 0

    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        pred_anns = sorted(pred_anns, key=lambda x: x["score"], reverse=True)

        if not gt_anns:
            fp += len(pred_anns)
            continue
        if not pred_anns:
            fn += len(gt_anns)
            continue

        gt_rles = [coco_gt.annToRLE(ann) for ann in gt_anns]
        pred_rles = [ann["segmentation"] for ann in pred_anns]
        iscrowd = [int(ann.get("iscrowd", 0)) for ann in gt_anns]

        iou_matrix = mask_utils.iou(pred_rles, gt_rles, iscrowd)

        matched_gt = set()

        for pred_idx in range(len(pred_anns)):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx in range(len(gt_anns)):
                if gt_idx in matched_gt:
                    continue
                if iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn += len(gt_anns) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("-" * 40)
    print(f"F1-Score @ Mask-IoU={iou_threshold:.2f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("-" * 40)

    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    DatasetCatalog.register("boxes_test", lambda: get_box_dicts([TEST_DATA_PATH]))
    MetadataCatalog.get("boxes_test").set(thing_classes=["box"])

    # Load model architecture
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}. Did you train first?")
        return

    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.MODEL.WEIGHTS = WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # We only have 'box'
    # use low threshold for evaluation and high threshold for visualization
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)

    # quantitative evaluation
    print("\n--- Starting COCO Metric Evaluation ---")
    evaluator = COCOEvaluator("boxes_test", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "boxes_test")

    # This prints the AP (Average Precision) table
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print("\nEvaluation Results:")
    print(metrics)

    # visual evaluation
    print(f"\n--- Saving Visual Predictions to '{OUTPUT_VIS_DIR}' ---")
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    dataset_dicts = get_box_dicts([TEST_DATA_PATH])

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])

        outputs = predictor(img)

        # visualize
        v = Visualizer(
            img[:, :, ::-1],  # type: ignore
            metadata=MetadataCatalog.get("boxes_test"),
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,
        )

        instances = outputs["instances"].to("cpu")
        # Use overlay_instances to draw only the masks
        if instances.has("pred_masks"):
            num_instances = len(instances)
            assigned_colors = [
                RAINBOW_COLORS[i % len(RAINBOW_COLORS)] for i in range(num_instances)
            ]

            scores = instances.scores
            labels = [f"{score:.0%}" for score in scores]

            out = v.overlay_instances(
                masks=instances.pred_masks,
                boxes=None,
                labels=labels,
                assigned_colors=assigned_colors,
                alpha=0.2,
            )
        else:
            # Fallback if no masks are detected or model isn't outputting masks
            out = v.output

        # Save to disk
        filename = os.path.basename(d["file_name"])
        save_path = os.path.join(OUTPUT_VIS_DIR, f"pred_{filename}")
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        print(f"Saved visualization: {save_path}")

    plot_precision_recall(cfg.OUTPUT_DIR, "boxes_test")
    compute_f1_score(cfg.OUTPUT_DIR, "boxes_test", iou_threshold=0.5)
    compute_f1_score(cfg.OUTPUT_DIR, "boxes_test", iou_threshold=0.75)


if __name__ == "__main__":
    main()
