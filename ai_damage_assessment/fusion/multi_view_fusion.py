"""Multi-View Damage Deduplication and Graph Bipartite Matching Engine."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from .feature_embedder import DamagePatchFeatureEmbedder
from ..models.yolo_segmentor import DamageDetectionResult
from ..severity.measurements import GeometricDamageMetrics, VisualDamageMeasurementEngine
from ..severity.severity_engine import SeverityAssessmentEngine, SeverityResult, SeverityLevel
from ..decision.repair_replace import RepairReplaceEngine, DecisionResult, ActionType
from ..cost.estimator import RepairCostEstimator, CostEstimateResult, CostRange
from ..uncertainty.estimator import UncertaintyEstimator, UncertaintyResult

@dataclass
class FusedDamageInstance:
    """Unified damage instance merged across multiple viewpoints."""
    fused_id: str
    primary_image_idx: int
    observed_image_indices: List[int]
    class_id: int
    class_name: str
    part_location: str
    best_confidence: float
    best_metrics: GeometricDamageMetrics
    severity: SeverityResult
    decision: DecisionResult
    cost_estimate: CostEstimateResult
    uncertainty: UncertaintyResult
    view_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fused_id": self.fused_id,
            "primary_image_idx": self.primary_image_idx,
            "observed_image_indices": self.observed_image_indices,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "part_location": self.part_location,
            "view_count": self.view_count,
            "confidence": round(float(self.best_confidence), 4),
            "metrics": self.best_metrics.to_dict(),
            "severity": self.severity.to_dict(),
            "decision": self.decision.to_dict(),
            "cost_estimate": self.cost_estimate.to_dict(),
            "uncertainty": self.uncertainty.to_dict()
        }

@dataclass
class MultiViewInspectionResult:
    """Aggregated result of multi-image vehicle damage inspection with deduplication."""
    total_images_processed: int
    raw_detections_count: int
    fused_unique_damages_count: int
    duplicates_eliminated_count: int
    overall_cost_range: CostRange
    overall_cost_expected: float
    overall_parts_cost: float
    overall_labor_cost: float
    overall_paint_cost: float
    overall_confidence: float
    requires_human_review: bool
    fused_damages: List[FusedDamageInstance]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_images_processed": self.total_images_processed,
                "raw_detections_count": self.raw_detections_count,
                "fused_unique_damages_count": self.fused_unique_damages_count,
                "duplicates_eliminated_count": self.duplicates_eliminated_count,
                "overall_cost_range": self.overall_cost_range.to_dict(),
                "overall_cost_expected": round(self.overall_cost_expected, 2),
                "overall_parts_cost": round(self.overall_parts_cost, 2),
                "overall_labor_cost": round(self.overall_labor_cost, 2),
                "overall_paint_cost": round(self.overall_paint_cost, 2),
                "overall_confidence": round(self.overall_confidence, 4),
                "requires_human_review": self.requires_human_review
            },
            "fused_damages": [d.to_dict() for d in self.fused_damages]
        }

class MultiViewDamageFusion:
    """Matches and deduplicates damage detections across multiple vehicle images using Hungarian bipartite matching."""

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        visual_weight: float = 0.70,
        spatial_weight: float = 0.30
    ):
        self.similarity_threshold = similarity_threshold
        self.visual_weight = visual_weight
        self.spatial_weight = spatial_weight
        self.embedder = DamagePatchFeatureEmbedder()
        self.meas_engine = VisualDamageMeasurementEngine()
        self.sev_engine = SeverityAssessmentEngine()
        self.dec_engine = RepairReplaceEngine()
        self.cost_engine = RepairCostEstimator()
        self.unc_engine = UncertaintyEstimator()

    def fuse_multi_image_detections(
        self,
        images_list: List[np.ndarray],
        detections_per_image: List[List[DamageDetectionResult]],
        vehicle_type: str = "sedan"
    ) -> MultiViewInspectionResult:
        """Fuses detections across images, eliminating duplicate views of identical damage."""
        total_raw = sum(len(dets) for dets in detections_per_image)
        
        # Flatten all detections with image index tracking and compute embeddings
        item_list = []
        for img_idx, dets in enumerate(detections_per_image):
            img = images_list[img_idx]
            for det in dets:
                emb = self.embedder.extract_embedding(img, det.binary_mask, det.bbox_xyxy)
                metrics = self.meas_engine.compute_metrics_from_mask(det.binary_mask, det.image_shape)
                item_list.append({
                    "img_idx": img_idx,
                    "det": det,
                    "emb": emb,
                    "metrics": metrics
                })

        # Cluster/match items across images
        # Group by damage class and part location
        clusters: List[List[Dict[str, Any]]] = []

        for item in item_list:
            matched_cluster = None
            for cluster in clusters:
                # Check if class matches and part is compatible
                rep = cluster[0]
                if rep["det"].class_id != item["det"].class_id:
                    continue
                if rep["det"].estimated_part != item["det"].estimated_part:
                    continue

                # Check if this item is from an image already in cluster (can't have 2 identical from same image)
                existing_imgs = [c["img_idx"] for c in cluster]
                if item["img_idx"] in existing_imgs:
                    continue

                # Compute visual feature similarity
                sims = [self.embedder.cosine_similarity(c["emb"], item["emb"]) for c in cluster]
                avg_sim = float(np.mean(sims))

                if avg_sim >= self.similarity_threshold:
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster.append(item)
            else:
                clusters.append([item])

        # Build FusedDamageInstances from clusters
        fused_damages: List[FusedDamageInstance] = []

        for idx, cluster in enumerate(clusters):
            # Pick best detection (highest confidence & largest detailed mask)
            best_item = max(cluster, key=lambda x: (x["det"].confidence, x["metrics"].pixel_area))
            observed_images = list(set([c["img_idx"] for c in cluster]))
            
            det = best_item["det"]
            metrics = best_item["metrics"]

            # Evaluate severity, decision, uncertainty, cost
            sev = self.sev_engine.evaluate(det.class_name, metrics, det.estimated_part, det.confidence)
            dec = self.dec_engine.evaluate(det.class_name, det.estimated_part, sev.level, metrics)
            unc = self.unc_engine.evaluate(det.confidence, metrics)
            cost = self.cost_engine.estimate_damage_instance_cost(
                det.class_name,
                det.estimated_part,
                sev.level,
                dec.action,
                metrics,
                vehicle_type=vehicle_type,
                uncertainty_score=unc.uncertainty_score
            )

            # Bonus confidence for multi-view validation
            if len(observed_images) > 1:
                unc.confidence_score = min(0.99, unc.confidence_score * 1.08)
                unc.uncertainty_score = max(0.01, 1.0 - unc.confidence_score)

            fused_instance = FusedDamageInstance(
                fused_id=f"fused_dmg_{idx+1:03d}",
                primary_image_idx=best_item["img_idx"],
                observed_image_indices=observed_images,
                class_id=det.class_id,
                class_name=det.class_name,
                part_location=det.estimated_part,
                best_confidence=det.confidence,
                best_metrics=metrics,
                severity=sev,
                decision=dec,
                cost_estimate=cost,
                uncertainty=unc,
                view_count=len(observed_images)
            )
            fused_damages.append(fused_instance)

        # Aggregate total vehicle cost
        total_min_cost = sum(d.cost_estimate.cost_range.min_cost for d in fused_damages)
        total_exp_cost = sum(d.cost_estimate.cost_range.expected_cost for d in fused_damages)
        total_max_cost = sum(d.cost_estimate.cost_range.max_cost for d in fused_damages)
        
        parts_cost = sum(d.cost_estimate.parts_subtotal for d in fused_damages)
        labor_cost = sum(d.cost_estimate.body_labor_subtotal for d in fused_damages)
        paint_cost = sum(d.cost_estimate.paint_labor_subtotal + d.cost_estimate.paint_materials_subtotal for d in fused_damages)

        confidences = [d.uncertainty.confidence_score for d in fused_damages]
        overall_conf = float(np.mean(confidences)) if confidences else 1.0
        requires_review = any(d.uncertainty.requires_human_review for d in fused_damages)

        return MultiViewInspectionResult(
            total_images_processed=len(images_list),
            raw_detections_count=total_raw,
            fused_unique_damages_count=len(fused_damages),
            duplicates_eliminated_count=max(0, total_raw - len(fused_damages)),
            overall_cost_range=CostRange(min_cost=total_min_cost, expected_cost=total_exp_cost, max_cost=total_max_cost),
            overall_cost_expected=total_exp_cost,
            overall_parts_cost=parts_cost,
            overall_labor_cost=labor_cost,
            overall_paint_cost=paint_cost,
            overall_confidence=overall_conf,
            requires_human_review=requires_review,
            fused_damages=fused_damages
        )
