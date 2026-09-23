"""Geometric visual measurement extraction from YOLO11-seg instance segmentation masks."""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

@dataclass
class GeometricDamageMetrics:
    """Container for quantitative geometric measurements of a segmented damage instance."""
    pixel_area: int
    relative_area_ratio: float
    perimeter: float
    aspect_ratio: float
    solidity: float
    contour_complexity: float
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]           # (cx, cy)
    polygon_points: List[List[float]]       # [[x1, y1], [x2, y2], ...]
    width_px: int
    height_px: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel_area": self.pixel_area,
            "relative_area_ratio": round(self.relative_area_ratio, 6),
            "perimeter": round(self.perimeter, 2),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "solidity": round(self.solidity, 3),
            "contour_complexity": round(self.contour_complexity, 3),
            "bounding_box": list(self.bounding_box),
            "centroid": [round(self.centroid[0], 2), round(self.centroid[1], 2)],
            "width_px": self.width_px,
            "height_px": self.height_px
        }

class VisualDamageMeasurementEngine:
    """Extracts rigorous geometric and spatial metrics from binary masks and polygon coordinates."""

    def __init__(self, vehicle_area_estimate: Optional[float] = None):
        self.vehicle_area_estimate = vehicle_area_estimate

    def compute_metrics_from_mask(
        self,
        binary_mask: np.ndarray,
        image_shape: Tuple[int, int],
        vehicle_mask: Optional[np.ndarray] = None
    ) -> GeometricDamageMetrics:
        """Extracts geometric measurements from a 2D binary segmentation mask (0 or 255/1)."""
        img_h, img_w = image_shape[:2]
        total_img_pixels = img_h * img_w

        mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255
        pixel_area = int(np.sum(mask_uint8 > 0))

        if pixel_area == 0:
            return GeometricDamageMetrics(
                pixel_area=0,
                relative_area_ratio=0.0,
                perimeter=0.0,
                aspect_ratio=1.0,
                solidity=1.0,
                contour_complexity=1.0,
                bounding_box=(0, 0, 0, 0),
                centroid=(0.0, 0.0),
                polygon_points=[],
                width_px=0,
                height_px=0
            )

        # Contours extraction
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GeometricDamageMetrics(
                pixel_area=pixel_area,
                relative_area_ratio=pixel_area / total_img_pixels,
                perimeter=0.0,
                aspect_ratio=1.0,
                solidity=1.0,
                contour_complexity=1.0,
                bounding_box=(0, 0, 0, 0),
                centroid=(0.0, 0.0),
                polygon_points=[],
                width_px=0,
                height_px=0
            )

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest_contour, True))
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / max(float(h), 1.0)

        # Moments & centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float(x + w / 2)
            cy = float(y + h / 2)

        # Convex Hull & Solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(pixel_area / max(hull_area, 1.0))
        solidity = min(1.0, max(0.0, solidity))

        # Contour Complexity: Isoperimetric quotient (P^2 / (4 * pi * A))
        # Circle = 1.0, higher means jagged / complex crack or shatter
        contour_complexity = float((perimeter ** 2) / max(4.0 * np.pi * pixel_area, 1e-5))

        # Relative Area Ratio
        if vehicle_mask is not None and np.sum(vehicle_mask > 0) > 0:
            reference_area = np.sum(vehicle_mask > 0)
        elif self.vehicle_area_estimate is not None and self.vehicle_area_estimate > 0:
            reference_area = self.vehicle_area_estimate
        else:
            reference_area = total_img_pixels * 0.65  # Approximate vehicle coverage

        relative_area_ratio = float(pixel_area / max(reference_area, 1.0))
        relative_area_ratio = min(1.0, max(0.0, relative_area_ratio))

        # Polygon coordinates
        polygon_points = largest_contour.squeeze().tolist()
        if isinstance(polygon_points[0], int):
            polygon_points = [polygon_points]

        return GeometricDamageMetrics(
            pixel_area=pixel_area,
            relative_area_ratio=relative_area_ratio,
            perimeter=perimeter,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            contour_complexity=contour_complexity,
            bounding_box=(x, y, x + w, y + h),
            centroid=(cx, cy),
            polygon_points=polygon_points,
            width_px=w,
            height_px=h
        )

    def compute_metrics_from_polygon(
        self,
        polygon_coords: List[float],
        image_shape: Tuple[int, int]
    ) -> GeometricDamageMetrics:
        """Extracts geometric measurements from a list of normalized coordinates [x1, y1, x2, y2, ...]."""
        img_h, img_w = image_shape[:2]
        pts = []
        for i in range(0, len(polygon_coords), 2):
            px = int(polygon_coords[i] * img_w)
            py = int(polygon_coords[i+1] * img_h)
            pts.append([px, py])

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if len(pts) >= 3:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts_arr], 255)

        return self.compute_metrics_from_mask(mask, image_shape)
