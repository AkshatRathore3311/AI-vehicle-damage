"""Multi-image damage fusion and multi-view deduplication engine."""
from .feature_embedder import DamagePatchFeatureEmbedder
from .multi_view_fusion import MultiViewDamageFusion, FusedDamageInstance, MultiViewInspectionResult

__all__ = [
    "DamagePatchFeatureEmbedder",
    "MultiViewDamageFusion",
    "FusedDamageInstance",
    "MultiViewInspectionResult"
]
