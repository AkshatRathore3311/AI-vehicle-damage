"""Uncertainty Quantification Engine."""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from ..severity.measurements import GeometricDamageMetrics

class ConfidenceTier(str, Enum):
    HIGH = "high_confidence"
    MODERATE = "moderate_confidence"
    FLAG_HUMAN_REVIEW = "flag_for_human_review"

@dataclass
class UncertaintyResult:
    """Container for multi-factor uncertainty and confidence metrics."""
    uncertainty_score: float  # [0, 1] - lower is better
    confidence_score: float   # [0, 1] - higher is better
    tier: ConfidenceTier
    mask_entropy: float
    factors: Dict[str, float]
    requires_human_review: bool
    review_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uncertainty_score": round(self.uncertainty_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "tier": self.tier.value,
            "mask_entropy": round(self.mask_entropy, 4),
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "requires_human_review": self.requires_human_review,
            "review_reason": self.review_reason
        }

class UncertaintyEstimator:
    """Calculates calibrated confidence and uncertainty using detection probability, mask boundary sharpness, and spatial scale."""

    def __init__(
        self,
        high_thresh: float = 0.80,
        moderate_thresh: float = 0.60,
        entropy_weight: float = 0.40,
        size_penalty_weight: float = 0.20
    ):
        self.high_thresh = high_thresh
        self.moderate_thresh = moderate_thresh
        self.entropy_weight = entropy_weight
        self.size_penalty_weight = size_penalty_weight

    def evaluate(
        self,
        detection_confidence: float,
        metrics: GeometricDamageMetrics,
        mask_probabilities: Optional[np.ndarray] = None
    ) -> UncertaintyResult:
        """Evaluates uncertainty from detection confidence, mask entropy, and geometric scale."""
        # 1. Mask boundary entropy
        if mask_probabilities is not None and mask_probabilities.size > 0:
            p = np.clip(mask_probabilities, 1e-6, 1.0 - 1e-6)
            entropy_map = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
            mean_entropy = float(np.mean(entropy_map))
        else:
            # Approximate entropy from contour complexity & solidity
            boundary_fuzziness = (1.0 - metrics.solidity) * 0.5 + min(0.5, (metrics.contour_complexity - 1.0) / 10.0)
            mean_entropy = float(np.clip(boundary_fuzziness, 0.05, 0.85))

        # 2. Size penalty (extreme small objects < 100px or huge ambiguous segments)
        if metrics.pixel_area < 80:
            size_penalty = 0.35  # High uncertainty on tiny speckles
        elif metrics.pixel_area < 250:
            size_penalty = 0.15
        elif metrics.relative_area_ratio > 0.70:
            size_penalty = 0.25  # Large coverage might be lighting artifact
        else:
            size_penalty = 0.0

        # 3. Overall confidence score formulation
        # Detection conf penalized by entropy and small-size factor
        calibrated_conf = detection_confidence * (1.0 - self.entropy_weight * mean_entropy) * (1.0 - self.size_penalty_weight * size_penalty)
        calibrated_conf = float(np.clip(calibrated_conf, 0.05, 0.99))
        uncertainty_score = float(np.clip(1.0 - calibrated_conf, 0.01, 0.95))

        # 4. Confidence Tier
        review_reasons = []
        if calibrated_conf >= self.high_thresh:
            tier = ConfidenceTier.HIGH
            requires_review = False
            review_reason = None
        elif calibrated_conf >= self.moderate_thresh:
            tier = ConfidenceTier.MODERATE
            requires_review = False
            review_reason = None
        else:
            tier = ConfidenceTier.FLAG_HUMAN_REVIEW
            requires_review = True
            if detection_confidence < 0.50:
                review_reasons.append(f"Low initial detection confidence ({detection_confidence:.2f}).")
            if mean_entropy > 0.50:
                review_reasons.append(f"High mask boundary ambiguity / entropy ({mean_entropy:.2f}).")
            if size_penalty > 0.10:
                review_reasons.append("Small or borderline visual damage scale.")
            review_reason = " ".join(review_reasons) or "Overall confidence below automated settlement threshold."

        factors = {
            "raw_detection_conf": detection_confidence,
            "mask_entropy": mean_entropy,
            "size_penalty": size_penalty,
            "calibrated_conf": calibrated_conf
        }

        return UncertaintyResult(
            uncertainty_score=uncertainty_score,
            confidence_score=calibrated_conf,
            tier=tier,
            mask_entropy=mean_entropy,
            factors=factors,
            requires_human_review=requires_review,
            review_reason=review_reason
        )
