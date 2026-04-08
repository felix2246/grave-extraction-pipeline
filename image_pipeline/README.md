# Image Pipeline

Extends the extraction table produced by the [text pipeline](../text_pipeline/README.md) with visual information. The goal is a multimodal data representation: images and plates embedded in the PDF catalogue are extracted and linked to the corresponding grave entities. The output is a modified extraction table with a new column `matched_filenames` that stores the file paths of the assigned image files.

---

## Terminology

| Term | Meaning |
|---|---|
| **Abbildung (figure)** | Simple, isolated image embedded directly in the running text of the catalogue. Can be linked to a grave via regex-based matching. |
| **Tafel (plate)** | Complex graphic grouped at the end of the catalogue, composed of multiple sub-images. Requires segmentation before matching. |

This distinction is crucial for the routing logic of the pipeline.

---

## Prerequisites

The image pipeline depends on the output of the text pipeline. `grave_extract.csv` must exist before you start:

```
grave_image_matching/data/grave_extract.csv
```

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). [Detectron2](https://github.com/facebookresearch/detectron2) is pulled automatically from its source repository.

```bash
uv sync
```

Copy `.env.example` to `.env`:

```
GWDG_API_KEY=...      # or OPENAI_API_KEY=...
```

Tesseract has to be installed separately (used by strategy A2-OOL):

```bash
brew install tesseract          # macOS
apt install tesseract-ocr       # Ubuntu/Debian
```

---

## Pipeline

```
PDF catalogue  +  grave_extract.csv (from text pipeline)
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  1 · Image extraction & caption assignment                       │
│  All images embedded in the PDF are extracted. A heuristic       │
│  algorithm assigns the most suitable caption to each image       │
│  (captions.json).                                                │
└──────────────────────┬───────────────────────────────────────────┘
                       │  images + captions.json
                       ▼
              ┌────────────────┐
              │  Routing       │  Does the caption contain "Taf."?
              └───┬────────────┘
         No ──────┘         └──── Yes
         (figure)                 (plate)
              │                       │
              │                       ▼
              │         ┌─────────────────────────────────────────┐
              │         │  2 · Plate segmentation                 │
              │         │  Mask R-CNN (Detectron2), fine-tuned     │
              │         │  on annotated plate images.              │
              │         │  Splits each plate into sub-images.      │
              │         └──────────────────┬──────────────────────┘
              │                            │  sub-images
              │                            ▼
              │         ┌─────────────────────────────────────────┐
              │         │  3 · Matching  (two strategies)         │
              │         │                                         │
              │         │  A1-MLLM Vision-language model sends    │
              │         │          the sub-image directly to a    │
              │         │          MLLM                            │
              │         │                                         │
              │         │  A2-OOL  Faster R-CNN detects the ID    │
              │         │          label → OCR reads the text →   │
              │         │          LLM selects the grave          │
              │         └──────────────────┬──────────────────────┘
              │                            │
              └──────────────┬─────────────┘
              Regex lookup   │  plate matching
              (figure refs.) │
                             ▼
                  grave_matches_<timestamp>.csv
          (grave_extract.csv + column matched_filenames)
```

---

## Stage Details

### 1 · Image Extraction & Caption Assignment

Extracts all images from the PDF via PyMuPDF. For each image, surrounding text lines are scored as caption candidates – the closest, properly aligned line below the image wins. The result is a `captions.json` file. Images whose captions indicate a plate are then immediately split into sub-images by the segmentation model.

```bash
uv run python scripts/run_extract_images.py \
  --pdf-path grave_image_matching/data/katalog2_tafeln.pdf \
  --output-dir grave_image_matching/output/images \
  --seg-model-dir tafel_segmentation_model/output
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pdf-path` | `grave_image_matching/data/katalog2_tafeln.pdf` | Input PDF with plates and figures |
| `--output-dir` | `grave_image_matching/output/images` | Where to write images and `captions.json` |
| `--seg-model-dir` | `tafel_segmentation_model/output` | Dir with `config.yaml` and `model_final.pth` (Mask R-CNN) |

Outputs:
- `<output-dir>/` – extracted page images
- `<output-dir>/captions.json` – `{filename: caption}` map
- `<output-dir>/tafel-splits/<plate>/` – segmented sub-images

---

### 2 · Plate Segmentation (`tafel_segmentation_model/`)

Mask R-CNN model (Detectron2, `mask_rcnn_R_50_FPN_3x`) fine-tuned on manually annotated plate images to detect and crop the individual image boxes of each plate. The trained model is called automatically by Stage 1 during inference.

#### Annotation with LabelMe

Install [LabelMe](https://github.com/labelmeai/labelme) via the optional dependency group:

```bash
uv sync --extra annotation
```

Launch the annotation tool:

```bash
uv run labelme
```

Open the image directory in LabelMe and draw **polygons** or **rectangles** around each sub-image box on the plate. Use the label name **`box`** for every annotation. LabelMe saves one `.json` file per image in the same directory.

#### Data layout

Place annotated images and their `.json` files in a `train/` (and optionally `test/`) directory:

```
experiments/segmentation/<catalogue>/data/
├── train/
│   ├── page193_img0.png
│   ├── page193_img0.json   ← LabelMe annotation (label: "box")
│   ├── page194_img0.png
│   ├── page194_img0.json
│   └── ...
└── test/
    ├── page200_img0.png
    ├── page200_img0.json
    └── ...
```

The training script reads from `experiments/segmentation/first-catalogue/data/train` by default. To include additional catalogues, uncomment the corresponding line in `train_net.py`.

#### Train

```bash
uv run -m tafel_segmentation_model.train_net
# → tafel_segmentation_model/output_train_only_first_catalogue/model_final.pth
# → tafel_segmentation_model/output_train_only_first_catalogue/config.yaml
```

| Hyperparameter | Value |
|---|---|
| Base model | Mask R-CNN R-50 FPN 3× (COCO pre-trained) |
| Learning rate | 0.00025 |
| Max iterations | 1000 |
| Batch size | 2 images |
| Classes | 1 (`box`) |
| Device | CPU (change `MODEL.DEVICE` in script for GPU) |

#### Evaluate

```bash
uv run -m tafel_segmentation_model.test_net
```

Produces COCO metrics (AP, AR) and visual predictions in `experiments/segmentation/first-catalogue/output/test_result_visuals/`.

---

### 3 · Matching (`grave_image_matching/`)

Links the extracted image files to the grave entities in `grave_extract.csv`.

**Figures** are matched via regex against the `referenzierte_abbildungen` (referenced figures) column in the grave dataset -- no model required.

**Plate sub-images** require semantic matching. Two competing strategies were implemented and evaluated:

#### A1-MLLM (Multimodal Large Language Model)

The sub-image (optionally together with the full plate image) is sent directly to a MLLM. The model uses visual annotations in the image (letters, numbers) to identify the correct grave from a list of candidates.

Implemented in `tafel_matching_strategies/mllm/`:
- `SubImageOnlyStrategy` -- sends only the sub-image
- `FullTafelContextStrategy` -- sends sub-image plus full plate as context

#### A2-OOL (Object Detection → OCR → LLM)

Sequential pipeline:
1. **Faster R-CNN** (`tafel_subbox_id_extraction_model/`) detects the ID label within the sub-image
2. **EasyOCR** (fallback: Tesseract) reads the label text
3. **LLM** matches the extracted text against the plate references of the candidate graves

Implemented in `tafel_matching_strategies/extract_ids_first_strategy.py`.

#### Training the sub-box ID extraction model

**Annotation with LabelMe:**

Use the same LabelMe setup as above (`uv sync --extra annotation && uv run labelme`). Open the sub-image crops and draw **rectangles** around the ID label (the text annotation visible in each sub-image, e.g. "1", "A", "14"). Use the label name **`id`**.

**Data layout:**

```
tafel_subbox_id_extraction_model/data/
└── train/
    ├── 1.png
    ├── 1.json    ← LabelMe annotation (label: "id")
    ├── 2.png
    ├── 2.json
    └── ...
```

Test data lives in `experiments/id-extraction/<catalogue>/data/test/`.

**Train:**

```bash
uv run -m tafel_subbox_id_extraction_model.train_net
# → tafel_subbox_id_extraction_model/output/model_final.pth
# → tafel_subbox_id_extraction_model/output/config.yaml
```

| Hyperparameter | Value |
|---|---|
| Base model | Faster R-CNN R-50 FPN 3× (COCO pre-trained) |
| Learning rate | 0.00025 |
| Max iterations | 1000 |
| Batch size | 2 images |
| Classes | 1 (`id`) |
| Device | CPU (change `MODEL.DEVICE` in script for GPU) |

**Evaluate:**

```bash
uv run -m tafel_subbox_id_extraction_model.test_net
```

Produces COCO metrics, a PR curve, and per-image IoU analysis.

**Run matching:**

```bash
uv run python scripts/run_matching.py \
  --graves-csv grave_image_matching/data/grave_extract.csv \
  --captions-json grave_image_matching/output/images/captions.json \
  --output-dir grave_image_matching/output \
  --model mistral-large-instruct \
  --id-model-dir tafel_subbox_id_extraction_model/output
```

| Option | Default | Description |
|--------|---------|-------------|
| `--graves-csv` | `grave_image_matching/data/grave_extract.csv` | CSV from the text pipeline |
| `--captions-json` | `grave_image_matching/output/images/captions.json` | From stage 1 |
| `--output-dir` | `grave_image_matching/output` | Where to write `grave_matches_<timestamp>.csv` |
| `--model` | `mistral-large-instruct` | MLLM/LLM model name |
| `--id-model-dir` | `tafel_subbox_id_extraction_model/output` | Faster R-CNN config + weights (A2-OOL) |

**Run full pipeline (stage 1 + stage 2):**

```bash
uv run python scripts/run_pipeline.py \
  --pdf-path grave_image_matching/data/katalog2_tafeln.pdf \
  --graves-csv grave_image_matching/data/grave_extract.csv \
  --output-dir grave_image_matching/output \
  --model mistral-large-instruct
```

Output: `grave_image_matching/output/grave_matches_<timestamp>.csv`

---

## Project Structure

```
data/                               Input PDFs + grave_extract.csv (from text pipeline)
experiments/
  segmentation/                     Annotated training data for the segmentation model
  id-extraction/                    Annotated training data for the ID extraction model
  image-extraction/                 Image extraction experiments
tafel_segmentation_model/           Mask R-CNN for plate segmentation
  train_net.py                      Training script
  test_net.py                       Evaluation script
  lib.py                            Dataset loading utilities
tafel_subbox_id_extraction_model/   Faster R-CNN for sub-box ID detection (A2-OOL)
  train_net.py
  test_net.py
  lib.py
scripts/                            CLI entry points (Typer)
  run_extract_images.py             Stage 1: PDF → images + captions.json + tafel segmentation
  run_matching.py                   Stage 2: grave image matching
  run_pipeline.py                   Full pipeline (stage 1 + 2)
grave_image_matching/
  main.py                           Matching logic and run_matching()
  extract_images_with_captions.py   Image + caption extraction from PDF
  extract_boxes.py                  Detectron2-based sub-box segmentation (inference)
  segment_image_boxes.py            CV-based fallback for box segmentation
  tafel_matching_strategies/
    base.py                         Abstract matcher interface
    extract_ids_first_strategy.py   A2-OOL: Faster R-CNN + OCR + LLM
    mllm/                           A1-MLLM: multimodal large language model strategies
  constants.py                      Shared regex patterns and paths
  utils.py                          Helper functions
plots.ipynb                         Result analysis and visualisation
plot_learning_curve.py              Learning-curve visualisation for the models
```
