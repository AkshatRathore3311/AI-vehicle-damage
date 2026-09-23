"""Training pipeline for YOLO11n-seg and YOLO11s-seg vehicle damage instance segmentation models."""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Automates training, fine-tuning, and metric logging for YOLO11 segmentation architectures."""

    def __init__(self, output_dir: str = "ai_damage_assessment/runs/train"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        data_yaml: str,
        base_model: str = "yolo11n-seg.pt",
        epochs: int = 30,
        imgsz: int = 640,
        batch_size: int = 8,
        lr0: float = 0.01,
        patience: int = 10,
        device: str = "cpu",
        project_name: str = "vehicle_damage_seg",
        experiment_name: str = "exp"
    ) -> Dict[str, Any]:
        """Trains YOLO11-seg model on specified dataset config."""
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

        logger.info(f"Starting training for {base_model} on {data_yaml} ({epochs} epochs, device={device})...")

        model = YOLO(base_model)
        
        train_results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            patience=patience,
            device=device,
            project=str(self.output_dir / project_name),
            name=experiment_name,
            save=True,
            val=True,
            plots=True,
            verbose=True
        )

        return {
            "status": "success",
            "base_model": base_model,
            "epochs": epochs,
            "project_dir": str(self.output_dir / project_name / experiment_name),
            "best_weights": str(self.output_dir / project_name / experiment_name / "weights" / "best.pt")
        }
