import json
"""Synthetic Vehicle Damage Dataset Generator for rapid testing, validation, and benchmarking."""
import os
import random
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any

DAMAGE_CLASSES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass_shatter",
    4: "lamp_broken",
    5: "tire_flat"
}

PART_LOCATIONS = [
    "front_bumper", "rear_bumper", "hood", "fender_left", 
    "fender_right", "door_front_left", "door_front_right",
    "windshield", "headlamp_assembly", "taillamp_assembly", "wheel_tire"
]

class SyntheticDatasetGenerator:
    """Generates synthetic vehicle silhouettes with realistic damage polygon overlays and YOLO11-seg annotations."""

    def __init__(self, output_dir: str = "ai_damage_assessment/data/synthetic_dataset", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._setup_directories()

    def _setup_directories(self):
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _draw_vehicle_base(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Renders a realistic vehicle silhouette with metallic background and panel lines."""
        img = np.ones((height, width, 3), dtype=np.uint8) * random.randint(210, 240)
        
        # Vehicle body gradient
        car_color = (
            random.randint(40, 180),
            random.randint(40, 180),
            random.randint(40, 180)
        )
        
        # Draw vehicle body (sedan/SUV profile or quarter panel view)
        body_pts = np.array([
            [int(width * 0.10), int(height * 0.65)],
            [int(width * 0.15), int(height * 0.45)],
            [int(width * 0.35), int(height * 0.30)],
            [int(width * 0.70), int(height * 0.30)],
            [int(width * 0.88), int(height * 0.48)],
            [int(width * 0.95), int(height * 0.65)],
            [int(width * 0.92), int(height * 0.78)],
            [int(width * 0.08), int(height * 0.78)]
        ], np.int32)
        
        cv2.fillPoly(img, [body_pts], car_color)
        cv2.polylines(img, [body_pts], True, (30, 30, 30), 2)
        
        # Windshield
        window_pts = np.array([
            [int(width * 0.37), int(height * 0.32)],
            [int(width * 0.68), int(height * 0.32)],
            [int(width * 0.65), int(height * 0.45)],
            [int(width * 0.32), int(height * 0.45)]
        ], np.int32)
        cv2.fillPoly(img, [window_pts], (70, 90, 110))
        
        # Headlight / Taillight
        cv2.rectangle(img, (int(width * 0.08), int(height * 0.55)), (int(width * 0.14), int(height * 0.62)), (240, 240, 150), -1)
        cv2.rectangle(img, (int(width * 0.88), int(height * 0.55)), (int(width * 0.94), int(height * 0.62)), (50, 50, 220), -1)
        
        # Wheels
        cv2.circle(img, (int(width * 0.25), int(height * 0.76)), int(width * 0.08), (25, 25, 25), -1)
        cv2.circle(img, (int(width * 0.25), int(height * 0.76)), int(width * 0.04), (180, 180, 180), -1)
        cv2.circle(img, (int(width * 0.75), int(height * 0.76)), int(width * 0.08), (25, 25, 25), -1)
        cv2.circle(img, (int(width * 0.75), int(height * 0.76)), int(width * 0.04), (180, 180, 180), -1)
        
        return img

    def _generate_damage_polygon(self, cls_id: int, width: int, height: int) -> Tuple[np.ndarray, List[float], str]:
        """Generates realistic damage geometry, polygon coordinates, and associated part tag."""
        part = random.choice(PART_LOCATIONS)
        
        if cls_id == 0:  # Dent (elliptical deformation)
            cx, cy = random.randint(int(width * 0.2), int(width * 0.8)), random.randint(int(height * 0.45), int(height * 0.70))
            rx, ry = random.randint(25, 60), random.randint(20, 50)
            angles = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
            r_noise = rx * (1 + 0.2 * (np.random.rand(12) - 0.5))
            pts = np.stack([cx + r_noise * np.cos(angles), cy + (ry / rx) * r_noise * np.sin(angles)], axis=1).astype(np.int32)
            
        elif cls_id == 1:  # Scratch (elongated jagged polygon)
            x0, y0 = random.randint(int(width * 0.2), int(width * 0.7)), random.randint(int(height * 0.4), int(height * 0.7))
            length = random.randint(40, 120)
            angle = random.uniform(-0.6, 0.6)
            x1 = int(x0 + length * np.cos(angle))
            y1 = int(y0 + length * np.sin(angle))
            thickness = random.randint(4, 10)
            pts = np.array([
                [x0, y0 - thickness],
                [x1, y1 - thickness],
                [x1 + 3, y1 + thickness],
                [x0 - 2, y0 + thickness]
            ], np.int32)
            
        elif cls_id == 2:  # Crack (branching jagged polygon)
            x0, y0 = random.randint(int(width * 0.15), int(width * 0.85)), random.randint(int(height * 0.5), int(height * 0.75))
            pts_list = [[x0, y0]]
            curr_x, curr_y = x0, y0
            for _ in range(6):
                curr_x += random.randint(8, 20)
                curr_y += random.randint(-12, 12)
                pts_list.append([curr_x, curr_y])
            for _ in range(6):
                curr_x -= random.randint(8, 20)
                curr_y += random.randint(-5, 5) + 4
                pts_list.append([curr_x, curr_y])
            pts = np.array(pts_list, np.int32)
            
        elif cls_id == 3:  # Glass shatter (web / star polygon on windshield)
            part = "windshield"
            cx, cy = random.randint(int(width * 0.4), int(width * 0.6)), random.randint(int(height * 0.35), int(height * 0.42))
            angles = np.linspace(0, 2 * np.pi, num=14, endpoint=False)
            radii = np.random.uniform(20, 50, size=14)
            pts = np.stack([cx + radii * np.cos(angles), cy + radii * 0.7 * np.sin(angles)], axis=1).astype(np.int32)
            
        elif cls_id == 4:  # Lamp broken (headlight / taillight polygon)
            part = random.choice(["headlamp_assembly", "taillamp_assembly"])
            if "head" in part:
                cx, cy = int(width * 0.11), int(height * 0.58)
            else:
                cx, cy = int(width * 0.91), int(height * 0.58)
            w_box, h_box = random.randint(20, 35), random.randint(20, 30)
            pts = np.array([
                [cx - w_box // 2, cy - h_box // 2],
                [cx + w_box // 2, cy - h_box // 2],
                [cx + w_box // 2, cy + h_box // 2],
                [cx - w_box // 2, cy + h_box // 2]
            ], np.int32)
            
        else:  # Tire flat (tire deformation)
            part = "wheel_tire"
            cx, cy = random.choice([(int(width * 0.25), int(height * 0.76)), (int(width * 0.75), int(height * 0.76))])
            w_tire, h_tire = random.randint(35, 50), random.randint(25, 40)
            pts = np.array([
                [cx - w_tire // 2, cy - h_tire // 2],
                [cx + w_tire // 2, cy - h_tire // 2],
                [cx + w_tire // 2 + 5, cy + h_tire // 2],
                [cx - w_tire // 2 - 5, cy + h_tire // 2]
            ], np.int32)

        # Clip points to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 2, width - 3)
        pts[:, 1] = np.clip(pts[:, 1], 2, height - 3)

        # Normalized polygon for YOLO format
        norm_coords = []
        for p in pts:
            norm_coords.extend([float(p[0]) / width, float(p[1]) / height])

        return pts, norm_coords, part

    def generate_dataset(self, num_train: int = 60, num_val: int = 15, num_test: int = 15, width: int = 640, height: int = 480) -> Dict[str, Any]:
        """Generates complete synthetic train, validation, and test datasets."""
        splits = {
            "train": num_train,
            "val": num_val,
            "test": num_test
        }
        
        metadata = {}

        for split, count in splits.items():
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split
            split_meta = []

            for i in range(count):
                img = self._draw_vehicle_base(width, height)
                num_damages = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
                label_lines = []
                damage_info_list = []

                for _ in range(num_damages):
                    cls_id = random.randint(0, 5)
                    pts, norm_poly, part = self._generate_damage_polygon(cls_id, width, height)
                    
                    # Draw damage on image with distinct realistic visual cue
                    if cls_id == 0:  # dent: darker shadow + highlight
                        cv2.fillPoly(img, [pts], (40, 40, 40))
                        cv2.polylines(img, [pts], True, (20, 20, 20), 2)
                    elif cls_id == 1:  # scratch: white/silver line
                        cv2.polylines(img, [pts], True, (230, 230, 240), 2)
                        cv2.fillPoly(img, [pts], (200, 200, 210))
                    elif cls_id == 2:  # crack: dark jagged line
                        cv2.fillPoly(img, [pts], (15, 15, 15))
                        cv2.polylines(img, [pts], True, (5, 5, 5), 2)
                    elif cls_id == 3:  # glass shatter: frosted fractured web
                        cv2.fillPoly(img, [pts], (220, 240, 255))
                        cv2.polylines(img, [pts], True, (255, 255, 255), 2)
                    elif cls_id == 4:  # lamp broken: orange/red shards
                        cv2.fillPoly(img, [pts], (40, 40, 180))
                        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
                    else:  # tire flat: squashed rim
                        cv2.fillPoly(img, [pts], (10, 10, 10))
                        cv2.polylines(img, [pts], True, (80, 80, 80), 3)

                    # Build YOLO label line
                    poly_str = " ".join([f"{c:.6f}" for c in norm_poly])
                    label_lines.append(f"{cls_id} {poly_str}")
                    damage_info_list.append({
                        "class_id": cls_id,
                        "class_name": DAMAGE_CLASSES[cls_id],
                        "part": part,
                        "polygon": norm_poly
                    })

                img_name = f"synth_car_{split}_{i:04d}.jpg"
                lbl_name = f"synth_car_{split}_{i:04d}.txt"

                cv2.imwrite(str(img_dir / img_name), img)
                with open(lbl_dir / lbl_name, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(label_lines))

                split_meta.append({
                    "image": img_name,
                    "damages": damage_info_list
                })

            metadata[split] = split_meta

        # Write data.yaml
        yaml_content = f'''# Ultralytics YOLO11-seg Synthetic Dataset Config
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

        # Write metadata JSON
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "output_dir": str(self.output_dir),
            "train_count": num_train,
            "val_count": num_val,
            "test_count": num_test,
            "classes": DAMAGE_CLASSES
        }
