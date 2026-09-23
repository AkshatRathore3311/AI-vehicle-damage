"""Pydantic v2 Data Schemas for REST API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class VehicleTypeEnum(str, Enum):
    SEDAN = "sedan"
    HATCHBACK = "hatchback"
    SUV = "suv"
    TRUCK = "truck"
    LUXURY = "luxury"
    BIKE = "bike"
    BUS = "bus"

class SeverityLevelEnum(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"

class ActionTypeEnum(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT_FURTHER = "inspect_further"

class CostItemSchema(BaseModel):
    category: str
    description: str
    quantity: float
    unit: str
    unit_price: float
    total_price: float

class CostRangeSchema(BaseModel):
    currency: str = "USD"
    min_cost: float
    expected_cost: float
    max_cost: float

class CostSummarySchema(BaseModel):
    parts_subtotal: float
    body_labor_subtotal: float
    paint_labor_subtotal: float
    paint_materials_subtotal: float
    shop_supplies_subtotal: float
    tax_total: float
    vehicle_multiplier: float

class GeometricMetricsSchema(BaseModel):
    pixel_area: int
    relative_area_ratio: float
    perimeter: float
    aspect_ratio: float
    solidity: float
    contour_complexity: float
    bounding_box: List[int]
    centroid: List[float]
    width_px: int
    height_px: int

class SeverityResultSchema(BaseModel):
    score: float
    level: SeverityLevelEnum
    confidence: float
    factors: Dict[str, float]
    explanation: str

class DecisionResultSchema(BaseModel):
    action: ActionTypeEnum
    confidence: float
    technique: str
    rationale: str
    safety_risk: str

class UncertaintyResultSchema(BaseModel):
    uncertainty_score: float
    confidence_score: float
    tier: str
    mask_entropy: float
    requires_human_review: bool
    review_reason: Optional[str] = None

class SingleDamageAssessmentSchema(BaseModel):
    damage_id: str
    class_id: int
    class_name: str
    part_location: str
    confidence: float
    metrics: GeometricMetricsSchema
    severity: SeverityResultSchema
    decision: DecisionResultSchema
    cost_range: CostRangeSchema
    cost_summary: CostSummarySchema
    line_items: List[CostItemSchema]
    uncertainty: UncertaintyResultSchema
    explanation: str

class SingleImageInspectionResponse(BaseModel):
    status: str = "success"
    vehicle_type: str
    make_model: Optional[str] = None
    total_damages_detected: int
    overall_cost_range: CostRangeSchema
    overall_confidence: float
    requires_human_review: bool
    damages: List[SingleDamageAssessmentSchema]
    annotated_image_base64: Optional[str] = None
    markdown_report: str

class FusedDamageSchema(BaseModel):
    fused_id: str
    primary_image_idx: int
    observed_image_indices: List[int]
    class_id: int
    class_name: str
    part_location: str
    view_count: int
    confidence: float
    metrics: GeometricMetricsSchema
    severity: SeverityResultSchema
    decision: DecisionResultSchema
    cost_range: CostRangeSchema
    cost_summary: CostSummarySchema
    line_items: List[CostItemSchema]
    uncertainty: UncertaintyResultSchema

class MultiImageInspectionSummarySchema(BaseModel):
    total_images_processed: int
    raw_detections_count: int
    fused_unique_damages_count: int
    duplicates_eliminated_count: int
    overall_cost_range: CostRangeSchema
    overall_cost_expected: float
    overall_parts_cost: float
    overall_labor_cost: float
    overall_paint_cost: float
    overall_confidence: float
    requires_human_review: bool

class MultiImageInspectionResponse(BaseModel):
    status: str = "success"
    vehicle_type: str
    make_model: Optional[str] = None
    summary: MultiImageInspectionSummarySchema
    fused_damages: List[FusedDamageSchema]
    annotated_images_base64: List[str] = []
    markdown_report: str

class TrainRequestSchema(BaseModel):
    data_yaml_path: str = "ai_damage_assessment/data/synthetic_dataset/data.yaml"
    base_model: str = "yolo11n-seg.pt"
    epochs: int = 10
    batch_size: int = 4
    imgsz: int = 640
    device: str = "cpu"

class TrainResponseSchema(BaseModel):
    status: str
    base_model: str
    epochs: int
    project_dir: str
    best_weights: str

class HealthResponseSchema(BaseModel):
    status: str
    service: str
    version: str
    model_status: str
    device: str
