"""Integration tests for FastAPI REST endpoints using TestClient."""
import io
import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from ai_damage_assessment.api.app import app

client = TestClient(app)

def _create_test_image_bytes():
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (100, 100), (200, 200), (30, 30, 30), -1)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_inspect_single_image_endpoint():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/v1/inspect/single",
        files={"file": ("test_car.jpg", img_bytes, "image/jpeg")},
        data={"vehicle_type": "sedan", "make_model": "Honda Civic"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "overall_cost_range" in data
    assert "markdown_report" in data

def test_inspect_multi_images_endpoint():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/v1/inspect/multi",
        files=[
            ("files", ("angle1.jpg", img_bytes, "image/jpeg")),
            ("files", ("angle2.jpg", img_bytes, "image/jpeg"))
        ],
        data={"vehicle_type": "suv", "make_model": "Toyota RAV4"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "summary" in data
    assert "fused_damages" in data
