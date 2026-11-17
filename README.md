# 🚗 AutoDamage AI — Vehicle Damage Assessment & Cost Estimator

> **AI-powered vehicle damage inspection system** that uses YOLO11 instance segmentation to detect damage types, assess severity, determine repair vs. replacement decisions, and generate itemized cost estimates — all from one or multiple vehicle photos.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [AI Pipeline](#ai-pipeline)
- [API Endpoints](#api-endpoints)
- [Data Schemas](#data-schemas)
- [Configuration](#configuration)
- [Installation & Setup](#installation--setup)
- [Running the Server](#running-the-server)
- [Testing](#testing)
- [Supported Vehicle Types](#supported-vehicle-types)
- [Damage Classes](#damage-classes)
- [Cost Estimation Model](#cost-estimation-model)
- [Multi-View Fusion](#multi-view-fusion)
- [Uncertainty Quantification](#uncertainty-quantification)

---

## Overview

**AutoDamage AI** is a production-ready REST API service built with **FastAPI** and **Ultralytics YOLO11-seg**. It accepts one or more vehicle images and returns:

- Pixel-precise **damage polygons** (instance segmentation masks)
- **Severity classification** (Minor / Moderate / Severe) using geometric metrics
- **Repair vs. Replace decisions** grounded in engineering and material physics
- **Itemized cost breakdowns** (parts, labor, paint, shop supplies, tax)
- **Uncertainty scores** and human-review flags for low-confidence cases
- **Multi-view deduplication** via Hungarian algorithm to eliminate duplicate detections across angles
- **Markdown inspection certificates** for claims adjusters

---

## Features

| Feature | Details |
|---|---|
| 🔍 Instance Segmentation | YOLO11-seg with polygon mask extraction |
| 📐 Geometric Analysis | Pixel area, relative area ratio, contour complexity, solidity, aspect ratio |
| ⚡ Severity Engine | Multi-factor scoring: damage class weight × area × contour × safety-critical escalation |
| 🔧 Repair/Replace Engine | Physics-based rules: material type (steel/aluminum/polypropylene), DOT safety standards |
| 💰 Cost Estimator | Parametric model: OEM parts catalog × vehicle class multiplier × labor rates |
| 🎯 Uncertainty Estimation | Mask entropy + detection confidence calibration + size penalty |
| 🔄 Multi-View Fusion | Hungarian bipartite matching across camera angles |
| 📊 Report Generation | Markdown inspection certificates with audit trails |
| 🌐 Web UI | Built-in HTML dashboard served at root `/` |
| 🏎️ Vehicle Intelligence | Make/model classification into Indian market pricing tiers |

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         FastAPI REST API           │
                    │  POST /api/v1/inspect/single       │
                    │  POST /api/v1/inspect/multi        │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │      YOLOSegmentor (YOLO11-seg)   │
                    │  Confidence=0.15 | IoU=0.45       │
                    │  → DamageDetectionResult[]        │
                    └────────────┬─────────────────────┘
                                 │
          ┌──────────────────────┼─────────────────────────┐
          │                      │                         │
┌─────────▼──────────┐  ┌───────▼─────────┐  ┌───────────▼──────────┐
│ VisualMeasurement  │  │  SeverityEngine  │  │   RepairReplaceEngine │
│  (Geometric Metrics│  │  (Score + Level) │  │   (Action + Tech)     │
│  pixel_area, perim │  │  minor/mod/severe│  │   repair/replace/insp │
└─────────┬──────────┘  └───────┬─────────┘  └───────────┬──────────┘
          │                      │                         │
          └──────────────────────┼─────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │      RepairCostEstimator          │
                    │  Parts + Labor + Paint + Tax      │
                    │  × Vehicle Class Multiplier       │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │      UncertaintyEstimator         │
                    │  Mask entropy + Size penalty      │
                    │  → Confidence tier + Review flag  │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │      MultiViewDamageFusion        │  ← Multi-image only
                    │  Hungarian matching + Dedup       │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │      InspectionReportBuilder      │
                    │  Markdown certificate + overlays  │
                    └──────────────────────────────────┘
```

---

## Project Structure

```
ai_vehicle_damage/
├── run_server.py                        # Uvicorn ASGI launcher
├── requirements.txt                     # Python dependencies
├── yolo11n-seg.pt                       # YOLO11-nano segmentation weights
│
└── ai_damage_assessment/
    ├── api/
    │   ├── app.py                       # FastAPI app + CORS + web UI + health check
    │   ├── routes.py                    # POST /inspect/single & /inspect/multi
    │   └── schemas.py                   # Pydantic v2 request/response models
    │
    ├── models/
    │   ├── yolo_segmentor.py            # YOLO11-seg wrapper + part location estimator
    │   ├── evaluator.py                 # Model evaluation utilities
    │   └── trainer.py                   # Fine-tuning on custom dataset
    │
    ├── severity/
    │   ├── measurements.py              # Geometric metric extraction from binary masks
    │   └── severity_engine.py           # Multi-factor severity scoring (minor/moderate/severe)
    │
    ├── decision/
    │   └── repair_replace.py            # Engineering rules: repair vs. replace vs. inspect
    │
    ├── cost/
    │   ├── estimator.py                 # Parametric itemized cost estimation
    │   ├── vehicle_intel.py             # Make/model → pricing tier resolver
    │   └── breakdown.py                 # Cost breakdown formatting
    │
    ├── fusion/
    │   ├── multi_view_fusion.py         # Hungarian-algorithm multi-angle deduplication
    │   └── feature_embedder.py          # Visual patch embeddings for similarity matching
    │
    ├── uncertainty/
    │   └── estimator.py                 # Confidence calibration + human review flagging
    │
    ├── report/
    │   ├── report_builder.py            # Markdown inspection certificate generator
    │   └── visualizer.py               # Overlay rendering + base64 image encoder
    │
    ├── config/
    │   ├── config.yaml                  # Hyperparameters and service configuration
    │   └── parts_catalog.json           # OEM parts pricing database
    │
    ├── data/                            # Synthetic training dataset
    ├── dataset/                         # Dataset utilities
    ├── experiments/                     # Training experiment logs
    ├── research/                        # LaTeX research paper draft
    ├── tests/                           # Pytest test suite
    └── web/
        └── index.html                   # Built-in web dashboard
```

---

## AI Pipeline

### 1. Detection — `YOLOSegmentor`
- Wraps **Ultralytics YOLO11-seg** (`yolo11n-seg.pt`)
- Confidence threshold: **0.15**, IoU threshold: **0.45** (tuned for damage sensitivity)
- Extracts: class label, confidence score, bounding box, normalized polygon, binary mask
- **Part location estimation** from spatial bbox coordinates and damage class
- **Large-crash decomposition**: splits massive front-end detections spanning `>35%` of image height into constituent assemblies (hood + bumper + headlamp)

### 2. Geometric Metrics — `VisualDamageMeasurementEngine`
Extracted from each binary segmentation mask:

| Metric | Description |
|---|---|
| `pixel_area` | Absolute mask pixel count |
| `relative_area_ratio` | Fraction of total image area |
| `perimeter` | Contour perimeter (pixels) |
| `aspect_ratio` | Width / height of bounding box |
| `solidity` | Mask area / convex hull area |
| `contour_complexity` | Perimeter² / (4π × area) — circularity ratio |
| `centroid` | (x, y) center of mass |
| `bounding_box` | [x1, y1, x2, y2] |
| `width_px`, `height_px` | Bounding box dimensions |

### 3. Severity Assessment — `SeverityAssessmentEngine`
Computes a continuous score [0–1] using weighted factors:

```
score = 0.35 × norm_relative_area
      + 0.20 × log_pixel_area
      + 0.30 × class_severity_weight
      + 0.15 × contour_complexity
```

**Class severity weights:**

| Damage Class | Weight |
|---|---|
| `scratch` | 0.8 |
| `dent` | 1.2 |
| `crack` | 1.5 |
| `tire_flat` | 1.8 |
| `lamp_broken` | 2.0 |
| `glass_shatter` | 2.2 |

**Thresholds:** Minor < 0.35 ≤ Moderate < 0.70 ≤ Severe

**Escalation rules:**
- Safety-critical parts (windshield, headlamp, taillamp, wheel, roof): score × 1.25
- `glass_shatter` or `lamp_broken` with area > 300px: score forced ≥ 0.75
- `tire_flat`: score forced ≥ 0.72

### 4. Repair vs. Replace Decision — `RepairReplaceEngine`

| Condition | Action | Technique |
|---|---|---|
| `glass_shatter` | **REPLACE** | OEM Glass + ADAS Recalibration |
| `lamp_broken` | **REPLACE** | Headlamp/Taillamp Housing Assembly |
| `tire_flat` | **REPLACE** | Tire + Wheel Alignment/Balancing |
| `dent` on aluminum + moderate/severe | **REPLACE** | Full OEM Aluminum Panel |
| `dent` severe or large critical panel | **REPLACE** | OEM Component + Structural Alignment |
| `dent` minor, low complexity | **REPAIR** | Paintless Dent Repair (PDR) |
| `dent` moderate | **REPAIR** | Stud-Pulling + Body Filler |
| `scratch` minor | **REPAIR** | Clear-Coat Buffing & Spot Polishing |
| `crack` polypropylene minor | **REPAIR** | Plastic Hot-Staple Welding + Epoxy |
| `crack` severe or steel | **REPLACE** | Component Replacement |

### 5. Cost Estimation — `RepairCostEstimator`
Parametric model based on Indian collision estimating rates:

| Rate Type | Default Value |
|---|---|
| Body labor rate | ₹650/hr |
| Paint labor rate | ₹750/hr |
| Paint materials | ₹450/sq ft |
| Shop supplies | 8% of subtotal |
| Tax rate | 18% GST |

**Cost = (Parts + Body Labor + Paint Labor + Paint Materials + Shop Supplies + Tax) × Vehicle Multiplier**

Cost range is derived from uncertainty score:
```
variance = max(0.10, min(0.35, uncertainty_score × 1.5))
min_cost = expected × (1 − variance)
max_cost = expected × (1 + variance)
```

### 6. Uncertainty Estimation — `UncertaintyEstimator`

```
calibrated_conf = detection_confidence
               × (1 − 0.40 × mask_entropy)
               × (1 − 0.20 × size_penalty)
```

| Tier | Threshold | Action |
|---|---|---|
| `high_confidence` | ≥ 0.80 | Auto-approved |
| `moderate_confidence` | 0.60 – 0.80 | Auto-approved |
| `flag_for_human_review` | < 0.60 | Requires adjuster review |

---

## API Endpoints

### `GET /`
Serves the built-in HTML web dashboard.

### `GET /health`
```json
{
  "status": "healthy",
  "service": "AI Vehicle Damage Assessment Service",
  "version": "1.0.0",
  "model_status": "loaded",
  "device": "cpu"
}
```

### `POST /api/v1/inspect/single`
Inspect a **single vehicle image**.

**Request** (`multipart/form-data`):

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | File | — | Vehicle image (JPG/PNG) |
| `vehicle_type` | Enum | `sedan` | Vehicle category |
| `make_model` | str | `"Vehicle"` | Make and model string |
| `include_image_overlay` | bool | `true` | Return annotated image (base64) |

**Response** (`SingleImageInspectionResponse`):
```json
{
  "status": "success",
  "vehicle_type": "sedan",
  "make_model": "Honda City",
  "total_damages_detected": 3,
  "overall_cost_range": { "currency": "USD", "min_cost": 1200.0, "expected_cost": 1800.0, "max_cost": 2400.0 },
  "overall_confidence": 0.87,
  "requires_human_review": false,
  "damages": [ /* array of SingleDamageAssessmentSchema */ ],
  "annotated_image_base64": "...",
  "markdown_report": "# Vehicle Damage Inspection..."
}
```

### `POST /api/v1/inspect/multi`
Inspect **multiple images** from different viewpoints with cross-angle deduplication.

**Request** (`multipart/form-data`):

| Field | Type | Default | Description |
|---|---|---|---|
| `files` | File[] | — | Multiple vehicle images |
| `vehicle_type` | Enum | `sedan` | Vehicle category |
| `make_model` | str | `"Vehicle"` | Make and model |
| `include_image_overlays` | bool | `true` | Return annotated images |

**Response** includes fusion summary with duplicate elimination stats:
```json
{
  "summary": {
    "total_images_processed": 4,
    "raw_detections_count": 11,
    "fused_unique_damages_count": 7,
    "duplicates_eliminated_count": 4,
    ...
  },
  "fused_damages": [ /* array of FusedDamageSchema */ ]
}
```

**Interactive Docs**: `http://127.0.0.1:8000/docs` (Swagger UI)

---

## Data Schemas

### `SingleDamageAssessmentSchema`
```python
damage_id: str              # e.g. "dmg_001"
class_id: int               # 0–5
class_name: str             # "dent", "scratch", etc.
part_location: str          # "hood", "front_bumper", etc.
confidence: float
metrics: GeometricMetricsSchema
severity: SeverityResultSchema      # score, level, factors, explanation
decision: DecisionResultSchema      # action, technique, rationale, safety_risk
cost_range: CostRangeSchema         # min/expected/max in USD
cost_summary: CostSummarySchema     # itemized subtotals
line_items: List[CostItemSchema]    # detailed line items
uncertainty: UncertaintyResultSchema
explanation: str
```

---

## Configuration

### `config/parts_catalog.json`
JSON database mapping part names to pricing and labor hours:
```json
{
  "hood": {
    "base_part_price": 12000.0,
    "standard_labor_replace_hours": 3.0,
    "standard_labor_repair_hours": 2.0,
    "paint_panels_count": 1.0,
    "paint_labor_hours": 3.5,
    "material_type": "steel"
  },
  ...
}
```

**Supported part locations:**
`hood`, `front_bumper`, `fender_left`, `fender_right`, `door_front_left`, `door_front_right`, `windshield`, `headlamp_assembly`, `taillamp_assembly`, `wheel_tire`, `roof`, `general_panel`

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | ≥0.110.0 | REST API framework |
| `uvicorn[standard]` | ≥0.28.0 | ASGI server |
| `pydantic` | ≥2.6.0 | Data validation |
| `ultralytics` | ≥8.1.0 | YOLO11 model |
| `torch` | ≥2.0.0 | Deep learning backend |
| `torchvision` | ≥0.15.0 | Vision utilities |
| `opencv-python-headless` | ≥4.8.0 | Image processing |
| `numpy` | ≥1.24.0, <2.0.0 | Numerical computing |
| `scipy` | ≥1.11.0 | Hungarian algorithm (`linear_sum_assignment`) |
| `scikit-learn` | ≥1.3.0 | ML utilities |
| `pillow` | ≥10.0.0 | Image I/O |
| `pandas` | ≥2.0.0 | Data handling |
| `httpx` | ≥0.27.0 | Async HTTP client |
| `pytest` | ≥8.0.0 | Testing |

---

## Running the Server

```bash
python run_server.py
```

Server starts at: **http://127.0.0.1:8000**

| URL | Description |
|---|---|
| `http://127.0.0.1:8000/` | Web Dashboard |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc UI |
| `http://127.0.0.1:8000/health` | Health Check |

---

## Testing

```bash
pytest ai_damage_assessment/tests/ -v
```

**Test modules:**

| Test File | Coverage |
|---|---|
| `test_api.py` | FastAPI endpoint integration tests |
| `test_cost_estimator.py` | Parametric cost estimation logic |
| `test_measurements.py` | Geometric metrics extraction |
| `test_multi_view_fusion.py` | Hungarian matching & deduplication |
| `test_severity_decision.py` | Severity engine & repair/replace rules |

---

## Supported Vehicle Types

| Type | Multiplier | Description |
|---|---|---|
| `hatchback` | 0.85× | Economy / mass-market (Swift, Baleno, i10) |
| `sedan` | 1.00× | Standard passenger (default) |
| `suv` | 1.35× | Mid-size / compact SUVs (Creta, Nexon) |
| `luxury` | 4.50× | Premium / executive (BMW, Mercedes, Camry) |
| `truck` | 1.40× | Commercial trucks |
| `bus` | 2.00× | Buses |
| `bike` | 0.40× | Two-wheelers (motorcycles, scooters) |

**Vehicle Intelligence** auto-classifies popular Indian makes/models into their correct pricing tier from the `make_model` string.

---

## Damage Classes

| ID | Class | Description |
|---|---|---|
| 0 | `dent` | Surface depression / panel deformation |
| 1 | `scratch` | Surface paint/clear-coat abrasion |
| 2 | `crack` | Structural fracture in body panel / bumper |
| 3 | `glass_shatter` | Windshield / window glass breakage |
| 4 | `lamp_broken` | Headlamp or taillamp housing damage |
| 5 | `tire_flat` | Flat or damaged tire/wheel |

---

## Cost Estimation Model

The parametric cost model follows industry collision repair estimating standards:

```
Total Cost = (
    Parts Cost (OEM catalog)
  + Body Labor (hours × ₹650/hr)
  + Paint Labor (hours × ₹750/hr)
  + Paint Materials (sq_ft × ₹450)
  + Shop Supplies (8% of above)
  + GST (18% on taxable items)
) × Vehicle Class Multiplier
```

**Severity adjustments on labor hours:**
- Minor: 0.7× base hours
- Moderate: 1.0× base hours
- Severe: 1.5× base hours
- Area scaling: `+sqrt(pixel_area)/100 × 0.4` hours

---

## Multi-View Fusion

The `MultiViewDamageFusion` engine eliminates duplicate detections when the same damage appears in multiple camera angles:

1. **Feature embedding** — patch embeddings extracted per detection
2. **Clustering** — group by same `class_id` + `part_location`
3. **Similarity matching** — cosine similarity of visual embeddings (threshold: 0.65)
4. **Cross-image constraint** — prevents merging two detections from the same image
5. **Best representative selection** — highest confidence + largest mask wins
6. **Multi-view confidence boost** — detections confirmed in 2+ images get +8% confidence

**Result**: Raw detections → deduplicated unique damage instances

---

## Report Output

Each inspection returns a **Markdown inspection certificate** including:
- Executive cost summary with min/expected/max range
- Multi-view deduplication stats
- Itemized damage table (ID, type, body part, severity, action, confidence, cost)
- Engineering justification and audit trail per damage instance
- Settlement status: ✅ Auto-approved or ⚠️ Flagged for human review

---

*Built with FastAPI · YOLO11 · PyTorch · OpenCV · Pydantic v2*
