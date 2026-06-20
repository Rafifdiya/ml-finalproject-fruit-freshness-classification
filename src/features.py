import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path

IMG_SIZE = (128, 128)  # was (64,64) — higher res for real-world texture
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
}
COLOR_BINS = 32


def _load_and_resize(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return cv2.resize(img, IMG_SIZE)


def _foreground_mask(hsv_img):
    s, v = hsv_img[:, :, 1], hsv_img[:, :, 2]
    fg = ~((s < 30) & (v > 180))
    if fg.sum() < fg.size * 0.2:
        return np.ones(hsv_img.shape[:2], dtype=np.uint8) * 255
    return fg.astype(np.uint8) * 255


def _compute_features(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hog_feat = hog(gray, **HOG_PARAMS)
    mask = _foreground_mask(hsv)
    hist_feats = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], mask, [COLOR_BINS], [0, 256])
        hist_feats.append(cv2.normalize(h, h).flatten())
    return np.concatenate([hog_feat] + hist_feats)


def _augment(bgr):
    """2 deterministic augmentations — brightness variation untuk real-world lighting."""
    variants = []

    # brightness +40% (terang)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)
    variants.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))

    # brightness -35% (gelap)
    hsv2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv2[:, :, 2] = np.clip(hsv2[:, :, 2] * 0.65, 0, 255)
    variants.append(cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR))

    return variants


def extract_features(img_path):
    """Single image → feature vector. Used by app.py inference."""
    return _compute_features(_load_and_resize(img_path))


def extract_features_augmented(img_path):
    """Original + 2 augmentations → list of 3 feature vectors. Training only."""
    bgr = _load_and_resize(img_path)
    return [_compute_features(v) for v in [bgr] + _augment(bgr)]
