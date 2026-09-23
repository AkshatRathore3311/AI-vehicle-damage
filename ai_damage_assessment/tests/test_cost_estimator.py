"""Unit tests for Parametric Cost Estimator."""
import numpy as np
import pytest
from ai_damage_assessment.severity.measurements import VisualDamageMeasurementEngine
from ai_damage_assessment.severity.severity_engine import SeverityLevel
from ai_damage_assessment.decision.repair_replace import ActionType
from ai_damage_assessment.cost.estimator import RepairCostEstimator

def test_cost_calculation_bounds_and_breakdown():
    estimator = RepairCostEstimator()
    meas = VisualDamageMeasurementEngine()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:180, 100:220] = 255
    metrics = meas.compute_metrics_from_mask(mask, (480, 640))

    cost_res = estimator.estimate_damage_instance_cost(
        damage_class="dent",
        part_location="front_bumper",
        severity_level=SeverityLevel.MODERATE,
        action=ActionType.REPAIR,
        metrics=metrics,
        vehicle_type="sedan"
    )

    cr = cost_res.cost_range
    assert cr.min_cost < cr.expected_cost < cr.max_cost
    assert cost_res.body_labor_subtotal > 0
    assert len(cost_res.line_items) >= 3

def test_luxury_vehicle_multiplier():
    estimator = RepairCostEstimator()
    meas = VisualDamageMeasurementEngine()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:180, 100:220] = 255
    metrics = meas.compute_metrics_from_mask(mask, (480, 640))

    sedan_res = estimator.estimate_damage_instance_cost("dent", "front_bumper", SeverityLevel.MODERATE, ActionType.REPAIR, metrics, "sedan")
    luxury_res = estimator.estimate_damage_instance_cost("dent", "front_bumper", SeverityLevel.MODERATE, ActionType.REPAIR, metrics, "luxury")

    assert pytest.approx(luxury_res.cost_range.expected_cost, rel=1e-2) == sedan_res.cost_range.expected_cost * 1.60
