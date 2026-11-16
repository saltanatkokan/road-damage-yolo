# Road Damage Detection with YOLO26n

This project detects road surface damage (cracks and potholes) in images automatically. It uses the YOLO26n model, fine-tuned on the Japan subset of the RDD2020 dataset.

## Results

Test set (856 images, conf = 0.239):

| Metric | Value |
|---|---|
| Precision | 0.608 |
| Recall | 0.534 |
| F1 | 0.569 |
| mAP@0.5 | 0.572 |
| mAP@0.5:0.95 | 0.278 |

Per-class mAP@0.5: D00 (longitudinal crack) 0.523, D10 (transverse crack) 0.512, D20 (alligator crack) 0.723, D40 (pothole) 0.529.

Speed (GTX 1650 Ti, FP32, batch = 1): median 15.6 ms per image, about 64 frames per second.

## Dataset

Dataset: RDD2020, Japan subset.
Source: Mendeley Data, DOI 10.17632/5ty2wb6gvg.1.

The dataset is not included in this repository. Download it yourself and place it in `data/raw/rdd2020/`.

Preprocessing:
- Only the four official classes were kept (D00, D10, D20, D40) from the XML annotations.
- Images were split 80/10/10 with seed = 42.
- Near-duplicate images were removed using dHash (64-bit, threshold ≤ 5).

Final split: train 6,956, val 864, test 856 images.

## Project structure

```
configs/        data.yaml, threshold.json (conf selected on validation)
notebooks/      notebooks 01–08, run in order
results/
  figures/      plots and examples
  predictions/  predictions on validation images
  tables/       metrics, speed benchmark, error analysis tables
```

## Notebooks

1. `01_environment_check.ipynb`: check the environment and GPU
2. `02_dataset_exploration.ipynb`: explore the dataset
3. `03_yolo_conversion.ipynb`: convert XML to YOLO format, split, deduplicate
4. `04_yolo_training.ipynb`: training (30 epochs, imgsz = 640, batch = 16)
5. `05_visualization.ipynb`: training curves and examples
6. `06_evaluation.ipynb`: validation and test metrics, speed
7. `07_experiments.ipynb`: additional experiments (optional)
8. `08_error_analysis.ipynb`: error analysis on the test set

## Installation

Requires Python 3.12 and a PyTorch build with CUDA support.

```bash
pip install ultralytics==8.4.191
```

Install PyTorch with the CUDA version that matches your GPU: https://pytorch.org

## Notes

- The model was trained only on Japanese images. It has not been tested on roads in Kazakhstan.
- The experiment used one model and one random seed.
- The confidence threshold was selected on the validation set only; the test set was not used for this choice.
