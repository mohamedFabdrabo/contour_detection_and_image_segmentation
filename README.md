# Contour Detection and Image Segmentation

This repository has been reorganized into a cleaner structure with `code/`, `data/`, and `tests/` directories.

## Project structure
- `code/`: main Python modules and scripts
- `data/`: dataset and example assets
- `test/`: unit and integration checks
- `archive/`: backup and legacy files

## Setup
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running segmentation
Use the script in `code/`:

```bash
python code/test.py data/8068.jpg
```

To run contour detection with a downloaded weight file, pass the optional argument:

```bash
python code/test.py data/8068.jpg --contour_weights Contour-Detection-Pytorch/models/Detector_model.pth
```

This script will:
- run MeanShift segmentation
- run FCN ResNet101 segmentation
- optionally run external detectors if available in `PbLite-Contour-Detection/Code` and `Contour-Detection-Pytorch`
- save a comparison image in the root `results/` folder (created automatically)

> Note: The encoder-decoder contour detector requires a trained weight file at `Contour-Detection-Pytorch/models/Detector_model.pth`. If you do not have this file, the contour detector will skip automatically and still show segmentation results.

## Evaluating predictions
Use the evaluation script in `code/`:

```bash
python code/evaluate.py --ground_truth_dir data/project_dataset/images/test --prediction_dir data/Results/encoder_decoder_contours1 --output_csv evaluation_report.csv
```

Optional flags:
- `--prefix` to add a filename prefix for prediction files
- `--ext` to restrict evaluation to a specific extension, e.g. `.png`

## Tests
Run the repository tests with pytest:
```bash
pytest test
```
## Results:
To compare the 4 different models, we ran the code for this test image:

<img width="240" height="160" alt="8068" src="https://github.com/user-attachments/assets/995d0b2e-f20b-4371-b6b5-317fcaaf13c3" />

The results for this image:

<img width="640" height="480" alt="Results" src="https://github.com/user-attachments/assets/bb8043c4-9291-4371-9b3c-2eb736ec393d" />

## Notes
- The encoder-decoder detector and PbLite modules are optional and may require external setup.
- If those optional modules are not available, `code/test.py` will still run the built-in MeanShift and FCN segmentation.
