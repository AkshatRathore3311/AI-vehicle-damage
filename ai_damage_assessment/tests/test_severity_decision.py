"""Unit tests for Severity Assessment and Repair vs Replace decision logic."""
import numpy as np
import pytest
from ai_damage_assessment.severity.measurements import VisualDamageMeasurementEngine
from ai_damage_assessment.severity.severity_engine import SeverityAssessmentEngine, SeverityLevel
from ai_damage_assessment.decision.repair_replace import RepairReplaceEngine, ActionType

def test_minor_scratch_decision():
    meas = VisualDamageMeasurementEngine()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:105, 100:160] = 255  # Thin scratch
    metrics = meas.compute_metrics_from_mask(mask, (480, 640))

    sev_engine = SeverityAssessmentEngine()
    sev_res = sev_engine.evaluate("scratch", metrics, "door_front_left")
    assert sev_res.level == SeverityLevel.MINOR

    dec_engine = RepairReplaceEngine()
    dec_res = dec_engine.evaluate("scratch", "door_front_left", sev_res.level, metrics)
    assert dec_res.action == ActionType.REPAIR

def test_shattered_glass_mandated_replace():
    meas = VisualDamageMeasurementEngine()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:250, 200:350] = 255
    metrics = meas.compute_metrics_from_mask(mask, (480, 640))

    sev_engine = SeverityAssessmentEngine()
    sev_res = sev_engine.evaluate("glass_shatter", metrics, "windshield")
    assert sev_res.level == SeverityLevel.SEVERE

    dec_engine = RepairReplaceEngine()
    dec_res = dec_engine.evaluate("glass_shatter", "windshield", sev_res.level, metrics)
    assert dec_res.action == ActionType.REPLACE
