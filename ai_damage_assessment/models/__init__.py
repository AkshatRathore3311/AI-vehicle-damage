"""Ultralytics YOLO11-seg instance segmentation models, training harness, and evaluation suite."""
from .yolo_segmentor import YOLOSegmentor, DamageDetectionResult
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator

__all__ = ["YOLOSegmentor", "DamageDetectionResult", "ModelTrainer", "ModelEvaluator"]
