"""CarDD (Car Damage Detection Dataset) COCO-to-YOLO11-seg format converter."""
import os
import json
import shutil
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    "dent": 0,
    "scratch": 1,
    "crack": 2,
    "glass_shatter": 3,
    "lamp_broken": 4,
    "tire_flat": 5,
    "glass shatter": 3,
    "lamp broken": 4,
    "tire flat": 5,
    "glass_breakage": 3,
    "headlamp_damage": 4,
    "scratches": 1,
    "dents": 0,
    "cracks": 2
}

class CarDDConverter:
    """Converts CarDD COCO JSON format instance segmentation annotations to YOLO11-seg text format."""

    def __init__(self, output_dir: str = "ai_damage_assessment/data/cardd_yolo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def convert_coco_json(self, json_path: str, images_dir: str, split: str = "train") -> Dict[str, Any]:
        """Parses COCO JSON and converts polygon segmentations to YOLO11-seg normalized format."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"COCO JSON file not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        images = {img["id"]: img for img in coco_data.get("images", [])}
        categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

        annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        converted_count = 0
        skipped_count = 0

        target_img_dir = self.output_dir / "images" / split
        target_lbl_dir = self.output_dir / "labels" / split

        for img_id, img_info in images.items():
            file_name = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]

            src_img_path = Path(images_dir) / file_name
            dst_img_path = target_img_dir / file_name

            if src_img_path.exists() and not dst_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)

            lbl_file_name = Path(file_name).stem + ".txt"
            dst_lbl_path = target_lbl_dir / lbl_file_name

            lines = []
            img_anns = annotations_by_image.get(img_id, [])
            for ann in img_anns:
                cat_id = ann["category_id"]
                cat_name = categories.get(cat_id, "").lower().strip()
                target_cls = CLASS_MAPPING.get(cat_name)

                if target_cls is None:
                    skipped_count += 1
                    continue

                segmentation = ann.get("segmentation", [])
                if not segmentation:
                    continue

                if isinstance(segmentation, list):
                    for poly in segmentation:
                        if len(poly) < 6:
                            continue
                        normalized_poly = []
                        for i in range(0, len(poly), 2):
                            nx = max(0.0, min(1.0, poly[i] / width))
                            ny = max(0.0, min(1.0, poly[i+1] / height))
                            normalized_poly.extend([f"{nx:.6f}", f"{ny:.6f}"])
                        line = f"{target_cls} " + " ".join(normalized_poly)
                        lines.append(line)

            with open(dst_lbl_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines))
            converted_count += 1

        self._generate_yaml()

        return {
            "split": split,
            "converted_images": converted_count,
            "skipped_annotations": skipped_count,
            "output_dir": str(self.output_dir)
        }

    def _generate_yaml(self):
        """Generates data.yaml for Ultralytics YOLO11 training."""
        yaml_content = f'''# Ultralytics YOLO11-seg Dataset Config
path: {self.output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: dent
  1: scratch
  2: crack
  3: glass_shatter
  4: lamp_broken
  5: tire_flat
'''
        with open(self.output_dir / "data.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)
