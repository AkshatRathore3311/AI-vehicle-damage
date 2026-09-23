"""Unit tests for Multi-View Damage Fusion and Deduplication."""
import numpy as np
import pytest
from ai_damage_assessment.models.yolo_segmentor import DamageDetectionResult
from ai_damage_assessment.fusion.multi_view_fusion import MultiViewDamageFusion

def test_duplicate_view_deduplication():
    img1 = np.ones((480, 640, 3), dtype=np.uint8) * 180
    img2 = np.ones((480, 640, 3), dtype=np.uint8) * 185

    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[100:180, 100:200] = 255
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    mask2[105:185, 95:195] = 255

    det1 = DamageDetectionResult("d1", 0, "dent", 0.90, [100, 100, 200, 180], [0.15, 0.2, 0.31, 0.37], mask1, (480, 640), "front_bumper")
    det2 = DamageDetectionResult("d2", 0, "dent", 0.88, [95, 105, 195, 185], [0.14, 0.21, 0.30, 0.38], mask2, (480, 640), "front_bumper")

    fusion = MultiViewDamageFusion()
    result = fusion.fuse_multi_image_detections([img1, img2], [[det1], [det2]], "sedan")

    assert result.raw_detections_count == 2
    assert result.fused_unique_damages_count == 1
    assert result.duplicates_eliminated_count == 1
    assert result.fused_damages[0].view_count == 2
