# Automatic Classification of Fresh and Rotten Fruits Using Deep Learning

A research benchmark for visual food quality assessment. The notebook classifies an image by both produce type and visual condition (`Healthy` or `Rotten`), yielding 24 classes across 12 produce categories.

This repository provides the experiment notebook and the verified dataset manifest. It does not contain the source images, trained weights, or generated result tables.

## Repository contents

| File | Description |
| --- | --- |
| [`automatic-classification-of-fresh-and-rotten.ipynb`](./automatic-classification-of-fresh-and-rotten.ipynb) | Dataset cleaning and partitioning, model training, evaluation, explainability, external validation, and sensitivity analysis. |
| [`final_verified_dataset_manifest.csv`](./final_verified_dataset_manifest.csv) | One row per retained image, including class, partition, relative paths, similarity-group ID, SHA-256 values, and verification fields. |

## Dataset and split

The notebook builds the cleaned corpus from the Kaggle input identified as `andreamencotovar/fruit-roller2025`. It inventories the source files, checks image readability, audits exact duplicates using SHA-256 and near-duplicates using perceptual hashes, and groups related images before assigning partitions. The grouped split is intended to reduce leakage from duplicates and augmentation variants.

| Split | Images |
| --- | ---: |
| Training | 22,672 |
| Validation | 2,835 |
| Test | 2,841 |
| **Total** | **28,348** |

In the published manifest, no recorded `split_guard_group_id` or `expected_sha256` appears in more than one split. This check applies to the identifiers recorded in the CSV; it does not prove the absence of every visually similar image.

## Benchmark design

The notebook defines **12 configurations**: four architectures (**ViT-B/16, Swin-Tiny, ConvNeXt-Tiny, and MobileNetV2**) crossed with three experimental settings: **no MixUp or CutMix**, **MixUp**, and **CutMix**. The first setting is not a claim that all image transforms are disabled.

Model selection uses validation macro-F1. The post-training sections cover internal test performance, confusion matrices and per-class metrics, calibration with temperature scaling, confidence intervals, paired model comparisons, inference efficiency, and Grad-CAM visualizations. Separate sections evaluate four selected configurations on an external set of 717 images and explore the sensitivity of the comparisons to image grouping and class balancing.

## Reproducing the experiments

1. Clone this repository and open the notebook in **Kaggle Notebooks**, the environment its paths target. A CUDA-capable GPU is recommended for training. The main dependencies include PyTorch, torchvision, timm, pandas, NumPy, scikit-learn, SciPy, Pillow, OpenCV, and Matplotlib.
2. Obtain the image inputs from their respective providers under the applicable terms. Attach the original corpus and review the `DATASET_ROOT`, input, and output paths before executing cells. The repository CSV records which images belong to each split; it cannot substitute for the image files.
3. Run the dataset-audit and split sections, or supply the equivalent cleaned image folders, manifests, and split artifacts at the locations expected by the training section.
4. Run training for the configurations you want to reproduce. The example training cell selects `vit_b16__mixup`; it does not train all 12 configurations in one invocation. The complete internal comparison expects the `best.pt` checkpoints for all 12 configurations.
5. For external validation, attach the separate external image set and supply the selected checkpoints. Run the overlap checks against the development corpus before external inference.

Cloning this repository alone is not sufficient for an end-to-end run. The notebook expects external data and, for evaluation-only runs, checkpoints. The repository does not pin dependency versions.

## Intended use

This is a research benchmark for classification from images, not a food-safety test. A visual `Healthy` prediction does not establish that a product is safe to consume.
