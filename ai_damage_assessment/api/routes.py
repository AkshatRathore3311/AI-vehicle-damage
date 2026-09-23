"""FastAPI Route handlers for single and multi-image inspection endpoints."""
import io
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from .schemas import (
    SingleImageInspectionResponse,
    MultiImageInspectionResponse,
    TrainRequestSchema,
    TrainResponseSchema,
    HealthResponseSchema,
    VehicleTypeEnum
)
from ..models.yolo_segmentor import YOLOSegmentor, DamageDetectionResult
from ..severity.measurements import VisualDamageMeasurementEngine
from ..severity.severity_engine import SeverityAssessmentEngine
from ..decision.repair_replace import RepairReplaceEngine
from ..cost.estimator import RepairCostEstimator
from ..uncertainty.estimator import UncertaintyEstimator
from ..fusion.multi_view_fusion import MultiViewDamageFusion
from ..report.visualizer import DamageVisualizer
from ..report.report_builder import InspectionReportBuilder

router = APIRouter(prefix="/api/v1", tags=["Inspection & Estimation"])

# Global pipeline singletons
_segmentor = None
_meas_engine = VisualDamageMeasurementEngine()
_sev_engine = SeverityAssessmentEngine()
_dec_engine = RepairReplaceEngine()
_cost_engine = RepairCostEstimator()
_unc_engine = UncertaintyEstimator()
_fusion_engine = MultiViewDamageFusion()
_visualizer = DamageVisualizer()

def get_segmentor():
    global _segmentor
    if _segmentor is None:
        _segmentor = YOLOSegmentor(model_path="yolo11n-seg.pt", device="cpu")
    return _segmentor

@router.post("/inspect/single", response_model=SingleImageInspectionResponse)
async def inspect_single_image(
    file: UploadFile = File(...),
    vehicle_type: VehicleTypeEnum = Form(VehicleTypeEnum.SEDAN),
    make_model: str = Form("Vehicle"),
    include_image_overlay: bool = Form(True)
):
    """Inspects a single vehicle image, delineating damage polygons, severity, repair/replace decisions, and itemized cost breakdown."""
    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    segmentor = get_segmentor()
    detections = segmentor.predict(img_bgr)

    # Process all detections
    assessed_damages = []
    fused_instances_for_report = []

    for det in detections:
        metrics = _meas_engine.compute_metrics_from_mask(det.binary_mask, det.image_shape)
        sev = _sev_engine.evaluate(det.class_name, metrics, det.estimated_part, det.confidence)
        dec = _dec_engine.evaluate(det.class_name, det.estimated_part, sev.level, metrics)
        unc = _unc_engine.evaluate(det.confidence, metrics)
        cost = _cost_engine.estimate_damage_instance_cost(
            det.class_name,
            det.estimated_part,
            sev.level,
            dec.action,
            metrics,
            vehicle_type=vehicle_type.value,
            uncertainty_score=unc.uncertainty_score
        )

        assessed_damages.append({
            "damage_id": det.damage_id,
            "class_id": det.class_id,
            "class_name": det.class_name,
            "part_location": det.estimated_part,
            "confidence": det.confidence,
            "metrics": metrics.to_dict(),
            "severity": sev.to_dict(),
            "decision": dec.to_dict(),
            "cost_range": cost.cost_range.to_dict(),
            "cost_summary": cost.to_dict()["summary"],
            "line_items": [i.to_dict() for i in cost.line_items],
            "uncertainty": unc.to_dict(),
            "explanation": cost.explanation
        })

    # Wrap in multi-view fusion for uniform report building
    fusion_result = _fusion_engine.fuse_multi_image_detections([img_bgr], [detections], vehicle_type=vehicle_type.value)
    markdown_rep = InspectionReportBuilder.build_markdown_report(fusion_result, {"make_model": make_model, "vehicle_type": vehicle_type.value})

    # Overlay rendering
    overlay_base64 = None
    if include_image_overlay:
        annotated_bgr = _visualizer.draw_overlays(img_bgr, fusion_result.fused_damages)
        overlay_base64 = _visualizer.to_base64_jpeg(annotated_bgr)

    return SingleImageInspectionResponse(
        status="success",
        vehicle_type=vehicle_type.value,
        make_model=make_model,
        total_damages_detected=len(assessed_damages),
        overall_cost_range=fusion_result.overall_cost_range.to_dict(),
        overall_confidence=fusion_result.overall_confidence,
        requires_human_review=fusion_result.requires_human_review,
        damages=assessed_damages,
        annotated_image_base64=overlay_base64,
        markdown_report=markdown_rep
    )

@router.post("/inspect/multi", response_model=MultiImageInspectionResponse)
async def inspect_multi_images(
    files: List[UploadFile] = File(...),
    vehicle_type: VehicleTypeEnum = Form(VehicleTypeEnum.SEDAN),
    make_model: str = Form("Vehicle"),
    include_image_overlays: bool = Form(True)
):
    """Inspects multiple vehicle images across different viewpoints with Hungarian graph deduplication."""
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    images_bgr = []
    for f in files:
        content = await f.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(pil_img)
        images_bgr.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    segmentor = get_segmentor()
    all_detections = [segmentor.predict(img) for img in images_bgr]

    # Run multi-view fusion
    result = _fusion_engine.fuse_multi_image_detections(images_bgr, all_detections, vehicle_type=vehicle_type.value)
    markdown_rep = InspectionReportBuilder.build_markdown_report(result, {"make_model": make_model, "vehicle_type": vehicle_type.value})

    # Render overlays for each image
    overlays_base64 = []
    if include_image_overlays:
        for idx, img in enumerate(images_bgr):
            relevant_damages = [d for d in result.fused_damages if idx in d.observed_image_indices]
            annotated = _visualizer.draw_overlays(img, relevant_damages)
            overlays_base64.append(_visualizer.to_base64_jpeg(annotated))

    fused_damage_dicts = []
    for d in result.fused_damages:
        fused_damage_dicts.append({
            "fused_id": d.fused_id,
            "primary_image_idx": d.primary_image_idx,
            "observed_image_indices": d.observed_image_indices,
            "class_id": d.class_id,
            "class_name": d.class_name,
            "part_location": d.part_location,
            "view_count": d.view_count,
            "confidence": d.best_confidence,
            "metrics": d.best_metrics.to_dict(),
            "severity": d.severity.to_dict(),
            "decision": d.decision.to_dict(),
            "cost_range": d.cost_estimate.cost_range.to_dict(),
            "cost_summary": d.cost_estimate.to_dict()["summary"],
            "line_items": [item.to_dict() for item in d.cost_estimate.line_items],
            "uncertainty": d.uncertainty.to_dict()
        })

    return MultiImageInspectionResponse(
        status="success",
        vehicle_type=vehicle_type.value,
        make_model=make_model,
        summary=result.to_dict()["summary"],
        fused_damages=fused_damage_dicts,
        annotated_images_base64=overlays_base64,
        markdown_report=markdown_rep
    )
