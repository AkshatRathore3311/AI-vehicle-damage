"""Feature extraction and visual similarity metric for damage patches across camera angles."""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

class DamagePatchFeatureEmbedder:
    """Extracts visual appearance signatures (color distribution, HSV histogram, Hu moments, texture) from damage masks."""

    def __init__(self, target_patch_size: Tuple[int, int] = (64, 64)):
        self.target_patch_size = target_patch_size

    def extract_embedding(self, image: np.ndarray, binary_mask: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extracts compact 128-dimensional normalized feature vector from a damage patch."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img_h, img_w = image.shape[:2]
        
        # Clip bbox bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(128, dtype=np.float32)

        patch_rgb = image[y1:y2, x1:x2]
        patch_mask = binary_mask[y1:y2, x1:x2]

        if patch_rgb.size == 0:
            return np.zeros(128, dtype=np.float32)

        # Resize patch to uniform dimensions
        resized_patch = cv2.resize(patch_rgb, self.target_patch_size)
        resized_mask = cv2.resize(patch_mask, self.target_patch_size, interpolation=cv2.INTER_NEAREST)

        # 1. HSV Color Histogram (32 bins for H, 16 for S, 16 for V = 64 dims)
        hsv = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], (resized_mask > 0).astype(np.uint8), [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], (resized_mask > 0).astype(np.uint8), [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], (resized_mask > 0).astype(np.uint8), [16], [0, 256])

        color_feat = np.concatenate([
            hist_h.flatten() / (np.sum(hist_h) + 1e-6),
            hist_s.flatten() / (np.sum(hist_s) + 1e-6),
            hist_v.flatten() / (np.sum(hist_v) + 1e-6)
        ])

        # 2. Grayscale Gradient / Texture Features (Sobel edges = 32 dims)
        gray = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        hist_grad = cv2.calcHist([ang], [0], (resized_mask > 0).astype(np.uint8), [32], [0, 2 * np.pi])
        grad_feat = hist_grad.flatten() / (np.sum(hist_grad) + 1e-6)

        # 3. Geometric Shape Invariants (Hu Moments + Spatial Proportions = 32 dims)
        moments = cv2.moments((resized_mask > 0).astype(np.uint8))
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log scale hu moments
        hu_log = -1.0 * np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        shape_pad = np.pad(hu_log, (0, 32 - len(hu_log)), mode="constant")

        # Concatenate 64 + 32 + 32 = 128 dimensions
        feature_vector = np.concatenate([color_feat, grad_feat, shape_pad]).astype(np.float32)
        norm = np.linalg.norm(feature_vector) + 1e-6
        return feature_vector / norm

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates cosine similarity in [-1, 1] normalized to [0, 1]."""
        if v1.size == 0 or v2.size == 0:
            return 0.0
        dot = float(np.dot(v1, v2))
        return float(np.clip((dot + 1.0) / 2.0, 0.0, 1.0))
