"""
Train a Detectron2 instance segmentation model on a annotated tafel dataset.

- Registers the training dataset.
- Loads the base Mask R-CNN architecture.
- Sets training hyperparameters and output directory.
- Begins model training and saves the final weights to disk.

Run after you have prepared your dataset in 'data/train/'.
"""

import os

from detectron2 import model_zoo  # type: ignore[import-untyped, import-not-found]
from detectron2.config import get_cfg  # type: ignore[import-untyped, import-not-found]
from detectron2.data import (  # type: ignore[import-untyped, import-not-found]
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import (  # type: ignore[import-untyped, import-not-found]
    DefaultTrainer,
)
from tafel_segmentation_model.lib import (  # type: ignore[import-untyped, import-not-found]
    get_box_dicts,
)

DatasetCatalog.register(
    "boxes_train",
    lambda: get_box_dicts(
        [
            "experiments/segmentation/first-catalogue/data/train",
            # "experiments/segmentation/second-catalogue/data/train",
        ]
    ),
)
MetadataCatalog.get("boxes_train").set(thing_classes=["box"])

if __name__ == "__main__":
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.OUTPUT_DIR = "tafel_segmentation_model/output_train_only_first_catalogue"
    cfg.DATASETS.TRAIN = ("boxes_train",)
    cfg.DATASETS.TEST = ()  # No metrics during training to save time
    cfg.DATALOADER.NUM_WORKERS = 2

    # Initialize with COCO weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # Lower learning rate because fine-tuning
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []  # Do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: 'box'
    cfg.MODEL.DEVICE = "cpu"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save the config for inference later
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())
