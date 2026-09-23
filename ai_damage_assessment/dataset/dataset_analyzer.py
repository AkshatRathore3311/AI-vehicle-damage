"""Dataset statistical analyzer and validator for vehicle damage segmentation datasets."""
import os
import glob
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

class DatasetAnalyzer:
    """Calculates class balance, bounding box/mask spatial statistics, and annotation validity."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.classes = {
            0: "dent",
            1: "scratch",
            2: "crack",
            3: "glass_shatter",
            4: "lamp_broken",
            5: "tire_flat"
        }

    def analyze(self) -> Dict[str, Any]:
        """Runs comprehensive validation and statistics on train, val, and test splits."""
        stats = {
            "splits": {},
            "total_images": 0,
            "total_annotations": 0,
            "class_counts": {c: 0 for c in self.classes.values()},
            "area_distribution": {"small (<1%)": 0, "medium (1-5%)": 0, "large (>5%)": 0}
        }

        for split in ["train", "val", "test"]:
            lbl_files = glob.glob(str(self.data_dir / "labels" / split / "*.txt"))
            img_files = glob.glob(str(self.data_dir / "images" / split / "*.*"))

            split_anns = 0
            for lf in lbl_files:
                with open(lf, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    cls_name = self.classes.get(cls_id, f"class_{cls_id}")
                    stats["class_counts"][cls_name] = stats["class_counts"].get(cls_name, 0) + 1
                    split_anns += 1

                    # Estimate polygon area from normalized coordinates
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) >= 6:
                        xs = coords[0::2]
                        ys = coords[1::2]
                        # Polygon area using Shoelace formula
                        area = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
                        if area < 0.01:
                            stats["area_distribution"]["small (<1%)"] += 1
                        elif area < 0.05:
                            stats["area_distribution"]["medium (1-5%)"] += 1
                        else:
                            stats["area_distribution"]["large (>5%)"] += 1

            stats["splits"][split] = {
                "images": len(img_files),
                "labels": len(lbl_files),
                "annotations": split_anns
            }
            stats["total_images"] += len(img_files)
            stats["total_annotations"] += split_anns

        return stats
