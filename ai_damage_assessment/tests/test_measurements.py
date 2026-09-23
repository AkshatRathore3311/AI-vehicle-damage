"""Unit tests for geometric visual damage measurements."""
import numpy as np
import pytest
from ai_damage_assessment.severity.measurements import VisualDamageMeasurementEngine

def test_rectangular_mask_measurements():
    engine = VisualDamageMeasurementEngine()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:300] = 255  # 100 x 200 rectangle = 20,000 px

    metrics = engine.compute_metrics_from_mask(mask, (480, 640))
    assert metrics.pixel_area == 20000
    assert metrics.width_px == 200
    assert metrics.height_px == 100
    assert pytest.approx(metrics.aspect_ratio, rel=1e-2) == 2.0
    assert pytest.approx(metrics.solidity, rel=1e-2) == 1.0
    assert metrics.bounding_box == (100, 100, 300, 200)

def test_empty_mask_measurements():
    engine = VisualDamageMeasurementEngine()
    mask = np.zeros((480, 640), dtype=np.uint8)
    metrics = engine.compute_metrics_from_mask(mask, (480, 640))
    assert metrics.pixel_area == 0
    assert metrics.relative_area_ratio == 0.0
    assert metrics.bounding_box == (0, 0, 0, 0)
