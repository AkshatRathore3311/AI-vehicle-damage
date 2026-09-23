"""Multi-Factor Severity Assessment Engine."""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from .measurements import GeometricDamageMetrics

class SeverityLevel(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"

@dataclass
class SeverityResult:
    """Complete output of severity evaluation for a damage instance."""
    score: float
    level: SeverityLevel
    confidence: float
    factors: Dict[str, float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level.value,
            "confidence": round(self.confidence, 4),
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "explanation": self.explanation
        }

class SeverityAssessmentEngine:
    """Evaluates damage severity using geometric measurements, damage class physics, and part location."""

    CLASS_WEIGHTS = {
        "scratch": 0.8,
        "dent": 1.2,
        "crack": 1.5,
        "glass_shatter": 2.2,
        "lamp_broken": 2.0,
        "tire_flat": 1.8
    }

    SAFETY_CRITICAL_PARTS = [
        "windshield", "headlamp_assembly", "taillamp_assembly", "wheel_tire", "roof"
    ]

    def __init__(
        self,
        minor_thresh: float = 0.35,
        moderate_thresh: float = 0.70,
        weights: Optional[Dict[str, float]] = None
    ):
        self.minor_thresh = minor_thresh
        self.moderate_thresh = moderate_thresh
        self.weights = weights or {
            "relative_area": 0.35,
            "pixel_area_log": 0.20,
            "class_weight": 0.30,
            "contour_complexity": 0.15
        }

    def evaluate(
        self,
        damage_class: str,
        metrics: GeometricDamageMetrics,
        part_location: str = "general_panel",
        detection_conf: float = 0.90
    ) -> SeverityResult:
        """Computes continuous severity score and assigns discrete categorical level with explainability."""
        class_key = damage_class.lower().replace(" ", "_")
        cls_w = self.CLASS_WEIGHTS.get(class_key, 1.0)

        # 1. Normalized relative area component [0, 1]
        norm_rel_area = min(1.0, metrics.relative_area_ratio * 15.0)

        # 2. Log pixel area component [0, 1]
        # 100 px -> ~0.2, 5000 px -> ~0.7, 20000 px -> 1.0
        log_area = float(np.log1p(metrics.pixel_area) / np.log1p(30000))
        log_area = min(1.0, max(0.0, log_area))

        # 3. Class severity factor normalized [0, 1]
        cls_factor = cls_w / 2.5

        # 4. Contour complexity normalized [0, 1]
        # Circle = 1.0 -> 0.0, jagged/crack (5-20) -> scaled
        comp_factor = min(1.0, max(0.0, (metrics.contour_complexity - 1.0) / 10.0))

        # Raw linear combination
        raw_score = (
            self.weights["relative_area"] * norm_rel_area +
            self.weights["pixel_area_log"] * log_area +
            self.weights["class_weight"] * cls_factor +
            self.weights["contour_complexity"] * comp_factor
        )

        # Safety-critical part amplification
        if part_location in self.SAFETY_CRITICAL_PARTS:
            raw_score = min(1.0, raw_score * 1.25)

        # Intrinsic class overrides: broken glass or lamps are inherently severe
        if class_key in ["glass_shatter", "lamp_broken"] and metrics.pixel_area > 300:
            raw_score = max(raw_score, 0.75)
        elif class_key == "tire_flat":
            raw_score = max(raw_score, 0.72)

        final_score = float(np.clip(raw_score, 0.0, 1.0))

        # Categorization
        if final_score < self.minor_thresh:
            level = SeverityLevel.MINOR
        elif final_score < self.moderate_thresh:
            level = SeverityLevel.MODERATE
        else:
            level = SeverityLevel.SEVERE

        # Explanations
        factors = {
            "relative_area_factor": norm_rel_area,
            "log_area_factor": log_area,
            "class_factor": cls_factor,
            "complexity_factor": comp_factor
        }

        reasons = []
        if level == SeverityLevel.MINOR:
            reasons.append(f"Superficial damage with low coverage ({metrics.pixel_area} px, {metrics.relative_area_ratio*100:.2f}% panel).")
        elif level == SeverityLevel.MODERATE:
            reasons.append(f"Noticeable damage spanning {metrics.pixel_area} px with moderate contour dispersion.")
        else:
            reasons.append(f"Critical or extensive damage on {part_location} ({class_key}).")

        if part_location in self.SAFETY_CRITICAL_PARTS:
            reasons.append(f"Location on safety-critical {part_location} escalated severity.")

        return SeverityResult(
            score=final_score,
            level=level,
            confidence=detection_conf * 0.95,
            factors=factors,
            explanation=" ".join(reasons)
        )
