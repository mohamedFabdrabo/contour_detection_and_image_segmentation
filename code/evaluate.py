import argparse
from pathlib import Path
from math import log10, sqrt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def psnr(original, compressed):
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * log10(255.0 / sqrt(mse))


def load_image(path, gray=True):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def compare_directories(ground_truth_dir, prediction_dir, prefix="", ext=None):
    ground_truth_dir = Path(ground_truth_dir)
    prediction_dir = Path(prediction_dir)
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    image_files = sorted(ground_truth_dir.glob("*"))
    results = []
    for gt_path in image_files:
        if not gt_path.is_file():
            continue
        if ext and gt_path.suffix.lower() != ext.lower():
            continue
        pred_path = prediction_dir / f"{prefix}{gt_path.name}"
        if not pred_path.exists():
            continue
        gt_img = load_image(gt_path)
        pred_img = load_image(pred_path)
        if gt_img.shape != pred_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        results.append((gt_path.name, psnr(gt_img, pred_img), ssim(gt_img, pred_img)))
    return results


def summarize(results):
    if not results:
        return {}
    psnrs = [r[1] for r in results]
    ssims = [r[2] for r in results]
    return {
        "count": len(results),
        "average_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "max_psnr": float(np.max(psnrs)),
        "average_ssim": float(np.mean(ssims)),
        "min_ssim": float(np.min(ssims)),
        "max_ssim": float(np.max(ssims)),
    }


def save_report(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["image,psnr,ssim"]
    lines += [f"{name},{psnr_value:.4f},{ssim_value:.4f}" for name, psnr_value, ssim_value in results]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions against ground truth.")
    parser.add_argument("--ground_truth_dir", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--output_csv", default="evaluation_report.csv")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--ext", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results = compare_directories(args.ground_truth_dir, args.prediction_dir, args.prefix, args.ext)
    summary = summarize(results)
    if results:
        save_report(results, args.output_csv)
    print("Processed:", summary.get("count", 0))
    for key, value in summary.items():
        print(f"{key}: {value}")
    if not results:
        print("No matching image pairs found. Check directory paths and filename conventions.")


if __name__ == "__main__":
    main()
