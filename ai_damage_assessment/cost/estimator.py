"""Explainable Parametric Vehicle Damage Repair Cost Estimator."""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from ..severity.measurements import GeometricDamageMetrics
from ..severity.severity_engine import SeverityLevel
from ..decision.repair_replace import ActionType

@dataclass
class CostItem:
    """Individual line-item in repair estimate."""
    category: str  # "parts", "body_labor", "paint_labor", "paint_materials", "shop_supplies"
    description: str
    quantity: float
    unit: str  # "hours", "units", "sq_ft", "flat"
    unit_price: float
    total_price: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "quantity": round(self.quantity, 2),
            "unit": self.unit,
            "unit_price": round(self.unit_price, 2),
            "total_price": round(self.total_price, 2)
        }

@dataclass
class CostRange:
    """Cost estimation range accounting for regional & uncertainty variance."""
    min_cost: float
    expected_cost: float
    max_cost: float
    currency: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "currency": self.currency,
            "min_cost": round(self.min_cost, 2),
            "expected_cost": round(self.expected_cost, 2),
            "max_cost": round(self.max_cost, 2)
        }

@dataclass
class CostEstimateResult:
    """Full explainable cost estimate for a single damage instance or complete vehicle."""
    cost_range: CostRange
    line_items: List[CostItem]
    parts_subtotal: float
    body_labor_subtotal: float
    paint_labor_subtotal: float
    paint_materials_subtotal: float
    shop_supplies_subtotal: float
    tax_total: float
    vehicle_multiplier: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cost_range": self.cost_range.to_dict(),
            "summary": {
                "parts_subtotal": round(self.parts_subtotal, 2),
                "body_labor_subtotal": round(self.body_labor_subtotal, 2),
                "paint_labor_subtotal": round(self.paint_labor_subtotal, 2),
                "paint_materials_subtotal": round(self.paint_materials_subtotal, 2),
                "shop_supplies_subtotal": round(self.shop_supplies_subtotal, 2),
                "tax_total": round(self.tax_total, 2),
                "vehicle_multiplier": round(self.vehicle_multiplier, 2)
            },
            "line_items": [item.to_dict() for item in self.line_items],
            "explanation": self.explanation
        }

class RepairCostEstimator:
    """Parametric, physics-grounded repair cost model based on industry collision estimating guides."""

    VEHICLE_MULTIPLIERS = {
        "sedan": 1.00,
        "hatchback": 0.95,
        "suv": 1.15,
        "truck": 1.25,
        "luxury": 1.60,
        "bike": 0.70,
        "bus": 1.80
    }

    def __init__(
        self,
        parts_catalog_path: str = "ai_damage_assessment/config/parts_catalog.json",
        labor_rate: float = 65.00,
        paint_rate: float = 75.00,
        paint_material_rate_per_sq_ft: float = 45.00,
        shop_supplies_rate: float = 0.08,
        tax_rate: float = 0.07
    ):
        self.labor_rate = labor_rate
        self.paint_rate = paint_rate
        self.paint_material_rate_per_sq_ft = paint_material_rate_per_sq_ft
        self.shop_supplies_rate = shop_supplies_rate
        self.tax_rate = tax_rate
        self.parts_catalog = self._load_parts_catalog(parts_catalog_path)

    def _load_parts_catalog(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def estimate_damage_instance_cost(
        self,
        damage_class: str,
        part_location: str,
        severity_level: SeverityLevel,
        action: ActionType,
        metrics: GeometricDamageMetrics,
        vehicle_type: str = "sedan",
        uncertainty_score: float = 0.15
    ) -> CostEstimateResult:
        """Calculates itemized cost breakdown for a single detected damage instance."""
        # Normalize enum/string inputs
        if isinstance(severity_level, str):
            severity_level = SeverityLevel(severity_level.lower())
        if isinstance(action, str):
            action = ActionType(action.lower())

        v_mult = self.VEHICLE_MULTIPLIERS.get(vehicle_type.lower(), 1.0)
        part_data = self.parts_catalog.get(part_location, self.parts_catalog.get("general_panel", {
            "base_part_price": 320.0,
            "standard_labor_replace_hours": 2.5,
            "standard_labor_repair_hours": 1.5,
            "paint_panels_count": 1.0,
            "paint_labor_hours": 2.0,
            "material_type": "steel"
        }))

        line_items: List[CostItem] = []
        explanation_steps: List[str] = []

        parts_subtotal = 0.0
        body_labor_hours = 0.0
        paint_labor_hours = 0.0
        paint_sq_ft = 0.0

        class_key = damage_class.lower().replace(" ", "_")

        # 1. Parts Replacement vs Body Labor
        if action == ActionType.REPLACE:
            base_part_cost = part_data.get("base_part_price", 300.0)
            parts_subtotal += base_part_cost
            line_items.append(CostItem(
                category="parts",
                description=f"OEM / Certified Replacement: {part_location.replace('_', ' ').title()}",
                quantity=1.0,
                unit="units",
                unit_price=base_part_cost,
                total_price=base_part_cost
            ))
            
            replace_hours = part_data.get("standard_labor_replace_hours", 2.5)
            body_labor_hours += replace_hours
            line_items.append(CostItem(
                category="body_labor",
                description=f"R&I / Replacement Labor for {part_location.replace('_', ' ')}",
                quantity=replace_hours,
                unit="hours",
                unit_price=self.labor_rate,
                total_price=replace_hours * self.labor_rate
            ))
            explanation_steps.append(f"Component replacement required for {part_location} (${base_part_cost:.2f} parts + {replace_hours:.1f}h R&I).")

        else:  # REPAIR
            base_repair_hours = part_data.get("standard_labor_repair_hours", 1.5)
            # Adjust hours based on severity and area
            if severity_level == SeverityLevel.MINOR:
                adj_hours = base_repair_hours * 0.7
            elif severity_level == SeverityLevel.MODERATE:
                adj_hours = base_repair_hours * 1.0
            else:
                adj_hours = base_repair_hours * 1.5

            # Area scaling: larger dents/scratches require additional metal/prep time
            area_scale = float(np.sqrt(max(1.0, metrics.pixel_area)) / 100.0) * 0.4
            total_repair_hours = round(max(0.5, adj_hours + area_scale), 1)

            body_labor_hours += total_repair_hours
            line_items.append(CostItem(
                category="body_labor",
                description=f"Body Repair & Panel Straightening ({class_key.title()} on {part_location.replace('_', ' ')})",
                quantity=total_repair_hours,
                unit="hours",
                unit_price=self.labor_rate,
                total_price=total_repair_hours * self.labor_rate
            ))
            explanation_steps.append(f"Body repair on {part_location} estimated at {total_repair_hours}h labor based on {severity_level.value} severity and {metrics.pixel_area} px mask size.")

        # 2. Paint & Refinishing
        if class_key in ["scratch", "dent", "crack"] or action == ActionType.REPLACE:
            base_paint_hours = part_data.get("paint_labor_hours", 0.0)
            if base_paint_hours > 0:
                if severity_level == SeverityLevel.MINOR and action == ActionType.REPAIR and class_key == "scratch":
                    # Spot buffing/blend only
                    paint_hours = 0.8
                    paint_sq_ft = 1.0
                    desc = f"Spot Sanding, Buffing & Clear Coat Blend ({part_location.replace('_', ' ')})"
                else:
                    paint_hours = base_paint_hours
                    paint_sq_ft = float(part_data.get("paint_panels_count", 1.0) * 4.5)  # Avg 4.5 sq ft per panel
                    desc = f"Full Panel Surface Prep, Basecoat & Clearcoat Refinishing ({part_location.replace('_', ' ')})"

                paint_labor_hours += paint_hours
                line_items.append(CostItem(
                    category="paint_labor",
                    description=desc,
                    quantity=paint_hours,
                    unit="hours",
                    unit_price=self.paint_rate,
                    total_price=paint_hours * self.paint_rate
                ))

                materials_cost = paint_sq_ft * self.paint_material_rate_per_sq_ft
                line_items.append(CostItem(
                    category="paint_materials",
                    description=f"Paint & Material Consumables ({paint_sq_ft:.1f} sq ft)",
                    quantity=paint_sq_ft,
                    unit="sq_ft",
                    unit_price=self.paint_material_rate_per_sq_ft,
                    total_price=materials_cost
                ))
                explanation_steps.append(f"Refinishing required: {paint_hours:.1f}h paint labor and {paint_sq_ft:.1f} sq ft material.")

        # Subtotals
        body_labor_cost = body_labor_hours * self.labor_rate
        paint_labor_cost = paint_labor_hours * self.paint_rate
        paint_materials_cost = sum(item.total_price for item in line_items if item.category == "paint_materials")

        raw_subtotal = parts_subtotal + body_labor_cost + paint_labor_cost + paint_materials_cost

        # Shop supplies & hazardous waste disposal
        shop_supplies = raw_subtotal * self.shop_supplies_rate
        line_items.append(CostItem(
            category="shop_supplies",
            description="Shop Supplies, Masking Hardware & Environmental Disposal",
            quantity=1.0,
            unit="flat",
            unit_price=shop_supplies,
            total_price=shop_supplies
        ))

        # Tax
        taxable_amount = parts_subtotal + paint_materials_cost + shop_supplies
        sales_tax = taxable_amount * self.tax_rate

        # Total before vehicle multiplier
        base_total = (raw_subtotal + shop_supplies + sales_tax) * v_mult
        
        # Uncertainty Cost Range
        variance_factor = max(0.10, min(0.35, uncertainty_score * 1.5))
        min_cost = base_total * (1.0 - variance_factor)
        max_cost = base_total * (1.0 + variance_factor)

        if v_mult != 1.0:
            explanation_steps.append(f"Applied {vehicle_type.upper()} vehicle class multiplier of {v_mult:.2f}x.")

        return CostEstimateResult(
            cost_range=CostRange(min_cost=min_cost, expected_cost=base_total, max_cost=max_cost),
            line_items=line_items,
            parts_subtotal=parts_subtotal * v_mult,
            body_labor_subtotal=body_labor_cost * v_mult,
            paint_labor_subtotal=paint_labor_cost * v_mult,
            paint_materials_subtotal=paint_materials_cost * v_mult,
            shop_supplies_subtotal=shop_supplies * v_mult,
            tax_total=sales_tax * v_mult,
            vehicle_multiplier=v_mult,
            explanation=" | ".join(explanation_steps)
        )
