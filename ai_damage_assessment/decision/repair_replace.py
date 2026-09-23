"""Repair vs. Replacement Decision Engine."""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..severity.measurements import GeometricDamageMetrics
from ..severity.severity_engine import SeverityLevel

class ActionType(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT_FURTHER = "inspect_further"

@dataclass
class DecisionResult:
    """Output of repair vs replacement engineering assessment."""
    action: ActionType
    confidence: float
    technique: str
    rationale: str
    safety_risk: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "technique": self.technique,
            "rationale": self.rationale,
            "safety_risk": self.safety_risk
        }

class RepairReplaceEngine:
    """Rules and physics engine determining whether damaged panels/components should be repaired or replaced."""

    MANDATED_REPLACE_CLASSES = ["glass_shatter", "lamp_broken", "tire_flat"]

    def __init__(self):
        pass

    def evaluate(
        self,
        damage_class: str,
        part_location: str,
        severity_level: SeverityLevel,
        metrics: GeometricDamageMetrics,
        part_info: Optional[Dict[str, Any]] = None
    ) -> DecisionResult:
        """Determines whether to repair or replace based on structural, optical, and safety thresholds."""
        class_key = damage_class.lower().replace(" ", "_")
        part_info = part_info or {}
        material = part_info.get("material_type", "steel")

        # 1. Mandated replacement rules for optical/structural safety
        if class_key == "glass_shatter":
            return DecisionResult(
                action=ActionType.REPLACE,
                confidence=0.98,
                technique="OEM Windshield / Glass Replacement & ADAS Sensor Recalibration",
                rationale="Laminated/tempered glass structural integrity and optical clarity cannot be restored safely once shattered.",
                safety_risk="High - compromized cabin structural rigidity and driver visibility."
            )

        if class_key == "lamp_broken":
            return DecisionResult(
                action=ActionType.REPLACE,
                confidence=0.96,
                technique="Complete Headlamp / Taillamp Housing Assembly Replacement",
                rationale="Cracked polycarbonate lens permits moisture ingress, leading to electrical failure and beam dispersion.",
                safety_risk="High - reduced night visibility and illumination compliance failure."
            )

        if class_key == "tire_flat":
            return DecisionResult(
                action=ActionType.REPLACE,
                confidence=0.94,
                technique="Tire Replacement & Wheel Rim Alignment / Balancing",
                rationale="Sidewall compromise or puncture beyond tread crown cannot be patched under DOT safety standards.",
                safety_risk="Critical - risk of high-speed tire blowout."
            )

        # 2. Scratches logic
        if class_key == "scratch":
            if severity_level == SeverityLevel.MINOR:
                return DecisionResult(
                    action=ActionType.REPAIR,
                    confidence=0.92,
                    technique="Clear-Coat Buffing & Spot Polishing",
                    rationale="Superficial scratch limited to clear coat layer; paint substrate intact.",
                    safety_risk="Low - purely cosmetic."
                )
            elif severity_level == SeverityLevel.MODERATE:
                return DecisionResult(
                    action=ActionType.REPAIR,
                    confidence=0.88,
                    technique="Spot Sanding, Primer Application & Single Panel Refinish",
                    rationale="Scratch penetrated basecoat; panel respray required to prevent corrosion.",
                    safety_risk="Low - cosmetic and minor rust prevention."
                )
            else:
                return DecisionResult(
                    action=ActionType.REPAIR,
                    confidence=0.82,
                    technique="Full Panel Strip, Feather-Prime & Multi-Stage Respray",
                    rationale="Extensive deep gouges across panel body line; requires complete panel repainting.",
                    safety_risk="Medium - potential panel corrosion if untreated."
                )

        # 3. Dents logic
        if class_key == "dent":
            if severity_level == SeverityLevel.MINOR and metrics.relative_area_ratio < 0.15:
                return DecisionResult(
                    action=ActionType.REPAIR,
                    confidence=0.90,
                    technique="Paintless Dent Repair (PDR)",
                    rationale="Minor depression without paint fracture or sharp body crease; suitable for PDR.",
                    safety_risk="Low - superficial contour irregularity."
                )
            elif severity_level == SeverityLevel.MODERATE or (severity_level == SeverityLevel.SEVERE and material != "aluminum" and metrics.relative_area_ratio < 0.40):
                return DecisionResult(
                    action=ActionType.REPAIR,
                    confidence=0.84,
                    technique="Conventional Stud-Pulling, Body Filler & Refinishing",
                    rationale="Deformation exceeds PDR threshold but retains structural panel mount integrity.",
                    safety_risk="Low to Medium - panel alignment required."
                )
            else:
                return DecisionResult(
                    action=ActionType.REPLACE,
                    confidence=0.87,
                    technique="Full OEM Panel Replacement & Welding / Bolting",
                    rationale="Severe crumple (>40% panel coverage or aluminum work hardening); repair labor exceeds replacement cost.",
                    safety_risk="Medium to High - structural deformation weakens impact resistance."
                )

        # 4. Cracks logic
        if class_key == "crack":
            if material in ["polypropylene", "composite"] and severity_level == SeverityLevel.MINOR and metrics.width_px < 100:
                return DecisionResult(
                    action=ActionType.REPAIR,
                    confidence=0.80,
                    technique="Plastic Hot-Staple Welding & Flexible Bumper Epoxy",
                    rationale="Minor plastic split without damaged mounting tabs; weldable with structural reinforcement.",
                    safety_risk="Low - bumper cover cosmetic integrity."
                )
            else:
                return DecisionResult(
                    action=ActionType.REPLACE,
                    confidence=0.89,
                    technique="Component Replacement",
                    rationale="Severe fracture or structural crack across reinforcement ribs or mounting points.",
                    safety_risk="High - loss of impact absorption capability."
                )

        # Default fallback
        return DecisionResult(
            action=ActionType.REPAIR if severity_level != SeverityLevel.SEVERE else ActionType.REPLACE,
            confidence=0.75,
            technique="Standard Collision Body Repair / Refinish",
            rationale="General damage remediation following standard collision guidelines.",
            safety_risk="Low to Medium."
        )
