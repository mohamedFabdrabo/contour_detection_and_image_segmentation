import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

def _build_feature_matrix(image):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]
    coords = np.indices((h, w)).transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)
    coords[:, 0] /= max(h - 1, 1)
    coords[:, 1] /= max(w - 1, 1)
    colors = image.reshape(-1, 3).astype(np.float32) / 255.0
    return np.concatenate([coords, colors], axis=1)


def MeanShiftSegmentation(image_path, quantile=0.05, downsample=2):
    """Segment an image with MeanShift clustering.

    Args:
        image_path: Path to the input image.
        quantile: Bandwidth quantile for MeanShift.
        downsample: Optional downsampling factor to speed up processing.

    Returns:
        Segmented RGB image as a uint8 NumPy array.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if downsample > 1:
        image = cv2.resize(
            image,
            (max(1, image.shape[1] // downsample), max(1, image.shape[0] // downsample)),
            interpolation=cv2.INTER_AREA,
        )

    features = _build_feature_matrix(image)
    bandwidth = estimate_bandwidth(features, quantile=quantile, n_jobs=-1)
    if bandwidth <= 0:
        raise RuntimeError("Could not estimate a valid bandwidth. Try increasing the quantile.")

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True, n_jobs=-1)
    labels = ms.fit_predict(features)
    segmented = ms.cluster_centers_[labels][:, 2:].reshape(image.shape)
    segmented = np.clip(segmented * 255, 0, 255).astype(np.uint8)
    return segmented

def main():
    return None

if __name__ == "__main__":
    main()