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
from tafel_subbox_id_extraction_model.lib import (  # type: ignore[import-untyped, import-not-found]
    get_id_dicts,
)

TEST_DATA_PATH = "experiments/id-extraction/second-catalogue/data/test"
WEIGHTS_PATH = "tafel_subbox_id_extraction_model/output/model_final.pth"
OUTPUT_VIS_DIR = "experiments/id-extraction/second-catalogue/output/test_result_visuals"
OUTPUT_DIR = "experiments/id-extraction/second-catalogue/output"

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

    plt.title("Precision-Recall Kurve")
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


def _compute_iou(bbox1, bbox2):
    """Compute IoU between two COCO-format bboxes [x, y, w, h]."""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_f1_score(output_dir, dataset_name, iou_threshold=0.5):
    """Compute Precision, Recall, and F1-Score at a given IoU threshold."""
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

        matched_gt = set()

        for pred in pred_anns:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_anns):
                if gt_idx in matched_gt:
                    continue
                iou = _compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
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
    print(f"F1-Score @ IoU={iou_threshold:.2f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("-" * 40)

    return {"precision": precision, "recall": recall, "f1": f1}


def find_failed_match(output_dir):
    from pycocotools.coco import COCO

    gt_file = os.path.join(output_dir, "test_ground_truth.json")
    pred_file = os.path.join(output_dir, "coco_instances_results.json")

    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(pred_file)

    all_img_ids = coco_gt.getImgIds()

    processed_files = set()
    results = []

    for img_id in all_img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        fname = img_info["file_name"]

        if fname in processed_files:
            continue
        processed_files.add(fname)

        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        if not ann_ids:
            continue

        gt_ann = coco_gt.loadAnns(ann_ids)[0]
        gt_bbox = gt_ann["bbox"]

        pred_ids = coco_dt.getAnnIds(imgIds=img_id)
        pred_anns = coco_dt.loadAnns(pred_ids)

        pred_anns = sorted(pred_anns, key=lambda x: x["score"], reverse=True)

        max_iou = 0.0

        if len(pred_anns) > 0:
            p = pred_anns[0]
            p_bbox = p["bbox"]

            x_left = max(gt_bbox[0], p_bbox[0])
            y_top = max(gt_bbox[1], p_bbox[1])
            x_right = min(gt_bbox[0] + gt_bbox[2], p_bbox[0] + p_bbox[2])
            y_bottom = min(gt_bbox[1] + gt_bbox[3], p_bbox[1] + p_bbox[3])

            if x_right < x_left or y_bottom < y_top:
                inter_area = 0.0
            else:
                inter_area = (x_right - x_left) * (y_bottom - y_top)

            gt_area = gt_bbox[2] * gt_bbox[3]
            pred_area = p_bbox[2] * p_bbox[3]
            union_area = gt_area + pred_area - inter_area

            max_iou = inter_area / union_area

        results.append((fname, max_iou, pred_anns))

    print(f"\n--- Analyzing IoU for {len(results)} Images ---")

    for fname, max_iou, pred_anns in results:
        if max_iou < 0.50:
            print(f"❌ MISS: {fname}")
            print(f"   IoU: {max_iou:.4f} (Threshold is 0.50)")
            print(f"   Confidence: {pred_anns[0]['score'] if pred_anns else 0}")
        else:
            print(f"✅ HIT : {fname} (IoU: {max_iou:.2f})")


def main() -> None:
    DatasetCatalog.register("ids_test", lambda: get_id_dicts(TEST_DATA_PATH))
    MetadataCatalog.get("ids_test").set(thing_classes=["id"])

    # Load model architecture
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}. Did you train first?")
        return

    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.MODEL.WEIGHTS = WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # We only have 'box'
    # Only show predictions with >70% confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)

    # quantitative evaluation
    print("\n--- Starting COCO Metric Evaluation ---")

    cache_file = os.path.join(cfg.OUTPUT_DIR, "ids_test_coco_format.json")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Cleared cached COCO format file to ensure fresh evaluation")

    evaluator = COCOEvaluator("ids_test", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "ids_test")

    # This prints the AP (Average Precision) table
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print("\nEvaluation Results:")
    print(metrics)

    # visual evaluation
    print(f"\n--- Saving Visual Predictions to '{OUTPUT_VIS_DIR}' ---")
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    dataset_dicts = get_id_dicts(TEST_DATA_PATH)

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # add padding to the image
        padding = 20
        padded_img = cv2.copyMakeBorder(
            img,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        # shift the bounding box coordinates
        instances.pred_boxes.tensor[:, 0::2] += padding  # Add padding to x1, x2
        instances.pred_boxes.tensor[:, 1::2] += padding  # Add padding to y1, y2

        # visualize using the PADDED image and SHIFTED instances
        v = Visualizer(
            padded_img[:, :, ::-1],
            metadata=MetadataCatalog.get("ids_test"),
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,
        )

        out = v.draw_instance_predictions(instances)

        # save result
        filename = os.path.basename(d["file_name"])
        save_path = os.path.join(OUTPUT_VIS_DIR, f"pred_{filename}")
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        print(f"Saved visualization: {save_path}")

    plot_precision_recall(cfg.OUTPUT_DIR, "ids_test")
    compute_f1_score(cfg.OUTPUT_DIR, "ids_test", iou_threshold=0.5)
    compute_f1_score(cfg.OUTPUT_DIR, "ids_test", iou_threshold=0.75)
    find_failed_match(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
