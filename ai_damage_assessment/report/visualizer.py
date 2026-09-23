"""Visual Segmentation Mask and Annotation Overlay Renderer."""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import base64
import io

CLASS_COLORS = {
    "dent": (0, 165, 255),          # Orange
    "scratch": (0, 255, 255),       # Yellow
    "crack": (0, 0, 255),           # Red
    "glass_shatter": (255, 191, 0), # Deep Sky Blue
    "lamp_broken": (255, 0, 255),   # Magenta
    "tire_flat": (128, 0, 128)      # Purple
}

class DamageVisualizer:
    """Draws smooth semi-transparent instance segmentation masks, contours, and badges on vehicle images."""

    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha

    def draw_overlays(
        self,
        image: np.ndarray,
        damage_instances: List[Any],
        show_metrics: bool = True
    ) -> np.ndarray:
        """Renders colored segmentation polygons, bounding boxes, labels, and severity tags."""
        overlay = image.copy()
        output = image.copy()

        for dmg in damage_instances:
            # Handle both DamageDetectionResult and FusedDamageInstance
            if hasattr(dmg, "best_metrics"):
                metrics = dmg.best_metrics
                cls_name = dmg.class_name
                conf = dmg.best_confidence
                sev_level = dmg.severity.level.value
                action = dmg.decision.action.value
                cost = dmg.cost_estimate.cost_range.expected_cost
            elif hasattr(dmg, "binary_mask"):
                cls_name = dmg.class_name
                conf = dmg.confidence
                sev_level = "assessed"
                action = "inspect"
                cost = 0.0
                # Extract quick bounding box
                x1, y1, x2, y2 = dmg.bbox_xyxy
            else:
                continue

            color = CLASS_COLORS.get(cls_name.lower().replace(" ", "_"), (0, 255, 0))

            # Draw segmentation mask if available
            if hasattr(dmg, "binary_mask") and dmg.binary_mask is not None:
                mask_bool = dmg.binary_mask > 0
                overlay[mask_bool] = color
                contours, _ = cv2.findContours((mask_bool).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, color, 2)
            elif hasattr(dmg, "best_metrics") and dmg.best_metrics.polygon_points:
                pts = np.array(dmg.best_metrics.polygon_points, dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(output, [pts], True, color, 2)

            # Bounding box
            if hasattr(dmg, "best_metrics"):
                x1, y1, x2, y2 = dmg.best_metrics.bounding_box
            else:
                x1, y1, x2, y2 = dmg.bbox_xyxy

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Badge Text
            label_text = f"{cls_name.upper()} ({conf*100:.0f}%) | {sev_level.upper()} -> {action.upper()}"
            if cost > 0:
                label_text += f" | ${cost:,.0f}"

            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y = max(th + 5, y1 - 6)
            cv2.rectangle(output, (x1, badge_y - th - 4), (x1 + tw + 8, badge_y + baseline + 2), (20, 20, 20), -1)
            cv2.rectangle(output, (x1, badge_y - th - 4), (x1 + tw + 8, badge_y + baseline + 2), color, 1)
            cv2.putText(output, label_text, (x1 + 4, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Alpha blend overlay
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    @staticmethod
    def to_base64_jpeg(image: np.ndarray, quality: int = 85) -> str:
        """Encodes BGR numpy image to base64 JPEG string for API transport."""
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer).decode("utf-8")
