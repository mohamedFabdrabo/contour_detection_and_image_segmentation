import argparse
import importlib
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
for candidate in [ROOT / "PbLite-Contour-Detection" / "Code", ROOT / "Contour-Detection-Pytorch"]:
    if candidate.exists():
        sys.path.append(str(candidate))

try:
    ContourDetector = importlib.import_module("ContourDetector")
except ImportError:
    ContourDetector = None

try:
    PbLiteModule = importlib.import_module("PbLite")
except ImportError:
    PbLiteModule = None

from MySegmentation import MeanShiftSegmentation
from FCN import FCN_RESNET101


def run(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_name = image_path.stem

    results = []

    if PbLiteModule is not None and hasattr(PbLiteModule, "PbLite"):
        try:
            print("Running PbLite algorithm...")
            pb_result = PbLiteModule.PbLite(image_name, str(image_path))
            results.append(("PbLite", pb_result))
        except Exception as exc:
            print(f"PbLite execution failed: {exc}")
    else:
        print("PbLite module not available. Skipping PbLite.")

    if ContourDetector is not None and hasattr(ContourDetector, "detectContours"):
        try:
            print("Running encoder-decoder contour detector...")
            contour_result = ContourDetector.detectContours(
                image_name,
                str(image_path),
                model_name=args.contour_weights if args.contour_weights else None,
            )
            results.append(("EncoderDecoder", contour_result))
        except FileNotFoundError as exc:
            print(f"ContourDetector model not found: {exc}")
        except Exception as exc:
            print(f"ContourDetector failed: {exc}")
    else:
        print("ContourDetector module not available. Skipping encoder-decoder.")

    print("Running MeanShift segmentation...")
    ms_result = MeanShiftSegmentation(str(image_path), quantile=0.05, downsample=2)
    results.append(("MeanShift", ms_result))

    print("Running FCN ResNet101 segmentation...")
    fcn_result = FCN_RESNET101(str(image_path))
    results.append(("FCN", fcn_result))

    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    if n == 0:
        raise RuntimeError("No segmentation results were generated.")
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, (title, result) in zip(axes, results):
        ax.imshow(result, cmap="gray" if hasattr(result, 'ndim') and result.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(results):]:
        ax.axis("off")

    output_file = output_dir / f"{image_name}_comparison.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    print(f"Saved comparison image to {output_file}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Run contour detection and segmentation algorithms.")
    parser.add_argument("image_path", help="Path to the input image to segment.")
    parser.add_argument(
        "--contour_weights",
        default=None,
        help="Optional path to the trained contour detection weights file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start = time.perf_counter()
    run(args.image_path)
    print("Elapsed time: {:.2f}s".format(time.perf_counter() - start))
    