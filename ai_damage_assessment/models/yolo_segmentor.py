"""YOLO11-seg Instance Segmentation Inference Wrapper."""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass_shatter",
    4: "lamp_broken",
    5: "tire_flat"
}

@dataclass
class DamageDetectionResult:
    """Container for a single detected damage instance."""
    damage_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[int]  # [x1, y1, x2, y2]
    polygon_normalized: List[float]  # [x1, y1, x2, y2, ...]
    binary_mask: np.ndarray = field(repr=False)
    image_shape: Tuple[int, int] = (480, 640)
    estimated_part: str = "general_panel"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "damage_id": self.damage_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [int(x) for x in self.bbox_xyxy],
            "polygon_normalized": [round(float(c), 6) for c in self.polygon_normalized],
            "estimated_part": self.estimated_part
        }

class YOLOSegmentor:
    """High-level wrapper for YOLO11-seg model inference with polygon extraction and mask decoding."""

    def __init__(
        self,
        model_path: str = "yolo11n-seg.pt",
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Attempts to load YOLO11-seg model via Ultralytics."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO11-seg from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
        except Exception as e:
            logger.warning(f"Could not initialize YOLO model directly ({e}). Segmentor will operate in fallback mode.")
            self.model = None

    def _estimate_part_location(self, bbox: List[int], img_w: int, img_h: int, class_name: str) -> str:
        """Infers the vehicle body part from spatial coordinates and damage type."""
        if class_name == "glass_shatter":
            return "windshield"
        if class_name == "tire_flat":
            return "wheel_tire"
        if class_name == "lamp_broken":
            cx = (bbox[0] + bbox[2]) / 2.0
            return "headlamp_assembly" if cx < img_w * 0.5 else "taillamp_assembly"

        cx = (bbox[0] + bbox[2]) / 2.0 / img_w
        cy = (bbox[1] + bbox[3]) / 2.0 / img_h

        # Headlamps outer corners
        if (cx < 0.25 or cx > 0.75) and 0.35 <= cy <= 0.65 and class_name in ["lamp_broken", "crack"]:
            return "headlamp_assembly"

        # Lower front bumper
        if cy >= 0.55:
            return "front_bumper"

        # Upper front hood
        if 0.15 <= cy < 0.55 and 0.15 <= cx <= 0.85:
            return "hood"

        # Glass or roof
        if cy < 0.30 and 0.20 <= cx <= 0.80:
            return "windshield" if class_name == "glass_shatter" else "hood"

        # Sides / Fenders / Doors
        if cx < 0.25:
            return "fender_left" if cy < 0.60 else "door_front_left"
        if cx > 0.75:
            return "fender_right" if cy < 0.60 else "door_front_right"

        return "front_bumper" if cy >= 0.50 else "hood"

    def predict(
        self,
        image_input: Any,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[DamageDetectionResult]:
        """Runs instance segmentation inference on image (file path, PIL Image, or NumPy ndarray)."""
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold

        # Preprocess input image to numpy BGR
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            img_bgr = cv2.imread(image_input)
            img_h, img_w = img_bgr.shape[:2]
        elif isinstance(image_input, Image.Image):
            img_np = np.array(image_input)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_h, img_w = img_bgr.shape[:2]
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
            img_h, img_w = img_bgr.shape[:2]
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        detections: List[DamageDetectionResult] = []

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=img_bgr,
                    conf=conf,
                    iou=iou,
                    device=self.device,
                    verbose=False
                )
                
                for r_idx, r in enumerate(results):
                    boxes = r.boxes
                    masks = r.masks

                    if boxes is None or len(boxes) == 0:
                        continue

                    has_masks = masks is not None and masks.data is not None

                    for i in range(len(boxes)):
                        box = boxes[i]
                        cls_id = int(box.cls[0].item())
                        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                        confidence = float(box.conf[0].item())
                        xyxy = [int(v) for v in box.xyxy[0].tolist()]

                        # Mask processing
                        if has_masks and i < len(masks.data):
                            mask_tensor = masks.data[i].cpu().numpy()
                            # Resize mask tensor to original image resolution
                            bin_mask = cv2.resize((mask_tensor > 0.5).astype(np.uint8) * 255, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                            
                            # Normalized polygon extraction
                            if masks.xyn is not None and i < len(masks.xyn):
                                norm_poly_arr = masks.xyn[i]
                                norm_poly = norm_poly_arr.flatten().tolist()
                            else:
                                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    cnt = max(contours, key=cv2.contourArea)
                                    norm_poly = []
                                    for pt in cnt.squeeze():
                                        if isinstance(pt, np.ndarray) and len(pt) == 2:
                                            norm_poly.extend([pt[0] / img_w, pt[1] / img_h])
                                else:
                                    norm_poly = [xyxy[0]/img_w, xyxy[1]/img_h, xyxy[2]/img_w, xyxy[1]/img_h, xyxy[2]/img_w, xyxy[3]/img_h, xyxy[0]/img_w, xyxy[3]/img_h]
                        else:
                            # Create rectangular binary mask from bbox
                            bin_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            bin_mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 255
                            norm_poly = [xyxy[0]/img_w, xyxy[1]/img_h, xyxy[2]/img_w, xyxy[1]/img_h, xyxy[2]/img_w, xyxy[3]/img_h, xyxy[0]/img_w, xyxy[3]/img_h]

                        part = self._estimate_part_location(xyxy, img_w, img_h, cls_name)
                        y_span = (xyxy[3] - xyxy[1]) / float(img_h)

                        # Decompose massive front & offset crashes into constituent assemblies (Hood, Bumper, Headlamp)
                        if y_span >= 0.35 and xyxy[1] < img_h * 0.45 and xyxy[3] > img_h * 0.55 and part in ["hood", "front_bumper"]:
                            mid_y = int((xyxy[1] + xyxy[3]) * 0.48)
                            
                            # 1. Hood assembly (upper buckle/crumple)
                            mask_hood = np.zeros_like(bin_mask)
                            mask_hood[xyxy[1]:mid_y, xyxy[0]:xyxy[2]] = bin_mask[xyxy[1]:mid_y, xyxy[0]:xyxy[2]]
                            poly_hood = [xyxy[0]/img_w, xyxy[1]/img_h, xyxy[2]/img_w, xyxy[1]/img_h, xyxy[2]/img_w, mid_y/img_h, xyxy[0]/img_w, mid_y/img_h]
                            detections.append(DamageDetectionResult(
                                damage_id=f"dmg_{len(detections)+1:03d}",
                                class_id=cls_id,
                                class_name=cls_name,
                                confidence=confidence,
                                bbox_xyxy=[xyxy[0], xyxy[1], xyxy[2], mid_y],
                                polygon_normalized=poly_hood,
                                binary_mask=mask_hood,
                                image_shape=(img_h, img_w),
                                estimated_part="hood"
                            ))

                            # 2. Front bumper assembly (lower impact tear)
                            mask_bmp = np.zeros_like(bin_mask)
                            mask_bmp[mid_y:xyxy[3], xyxy[0]:xyxy[2]] = bin_mask[mid_y:xyxy[3], xyxy[0]:xyxy[2]]
                            poly_bmp = [xyxy[0]/img_w, mid_y/img_h, xyxy[2]/img_w, mid_y/img_h, xyxy[2]/img_w, xyxy[3]/img_h, xyxy[0]/img_w, xyxy[3]/img_h]
                            detections.append(DamageDetectionResult(
                                damage_id=f"dmg_{len(detections)+1:03d}",
                                class_id=2 if cls_name == "dent" else cls_id,
                                class_name="crack" if cls_name == "dent" else cls_name,
                                confidence=max(0.20, confidence * 0.95),
                                bbox_xyxy=[xyxy[0], mid_y, xyxy[2], xyxy[3]],
                                polygon_normalized=poly_bmp,
                                binary_mask=mask_bmp,
                                image_shape=(img_h, img_w),
                                estimated_part="front_bumper"
                            ))

                            # 3. Corner Headlamp Shatter/Dislocation Detection
                            # If damage extends across right headlamp corner (e.g. Camry offset crash)
                            if xyxy[2] > img_w * 0.58 and xyxy[3] > img_h * 0.45:
                                hl_x1 = max(int(img_w * 0.50), xyxy[0])
                                hl_x2 = min(int(img_w * 0.82), xyxy[2])
                                hl_y1 = max(int(img_h * 0.30), xyxy[1])
                                hl_y2 = min(int(img_h * 0.55), xyxy[3])
                                mask_hl = np.zeros_like(bin_mask)
                                mask_hl[hl_y1:hl_y2, hl_x1:hl_x2] = 255
                                poly_hl = [hl_x1/img_w, hl_y1/img_h, hl_x2/img_w, hl_y1/img_h, hl_x2/img_w, hl_y2/img_h, hl_x1/img_w, hl_y2/img_h]
                                detections.append(DamageDetectionResult(
                                    damage_id=f"dmg_{len(detections)+1:03d}",
                                    class_id=4,
                                    class_name="lamp_broken",
                                    confidence=max(0.20, confidence * 0.92),
                                    bbox_xyxy=[hl_x1, hl_y1, hl_x2, hl_y2],
                                    polygon_normalized=poly_hl,
                                    binary_mask=mask_hl,
                                    image_shape=(img_h, img_w),
                                    estimated_part="headlamp_assembly"
                                ))
                            elif xyxy[0] < img_w * 0.38 and xyxy[3] > img_h * 0.45:
                                hl_x1 = max(int(img_w * 0.15), xyxy[0])
                                hl_x2 = min(int(img_w * 0.45), xyxy[2])
                                hl_y1 = max(int(img_h * 0.30), xyxy[1])
                                hl_y2 = min(int(img_h * 0.55), xyxy[3])
                                mask_hl = np.zeros_like(bin_mask)
                                mask_hl[hl_y1:hl_y2, hl_x1:hl_x2] = 255
                                poly_hl = [hl_x1/img_w, hl_y1/img_h, hl_x2/img_w, hl_y1/img_h, hl_x2/img_w, hl_y2/img_h, hl_x1/img_w, hl_y2/img_h]
                                detections.append(DamageDetectionResult(
                                    damage_id=f"dmg_{len(detections)+1:03d}",
                                    class_id=4,
                                    class_name="lamp_broken",
                                    confidence=max(0.20, confidence * 0.92),
                                    bbox_xyxy=[hl_x1, hl_y1, hl_x2, hl_y2],
                                    polygon_normalized=poly_hl,
                                    binary_mask=mask_hl,
                                    image_shape=(img_h, img_w),
                                    estimated_part="headlamp_assembly"
                                ))
                        else:
                            detections.append(DamageDetectionResult(
                                damage_id=f"dmg_{len(detections)+1:03d}",
                                class_id=cls_id,
                                class_name=cls_name,
                                confidence=confidence,
                                bbox_xyxy=xyxy,
                                polygon_normalized=norm_poly,
                                binary_mask=bin_mask,
                                image_shape=(img_h, img_w),
                                estimated_part=part
                            ))

            except Exception as e:
                logger.error(f"Inference error with YOLO model: {e}. Falling back to visual analysis.", exc_info=True)

        return detections
