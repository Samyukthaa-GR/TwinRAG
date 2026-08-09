"""
Anomaly detection (Module 3).

Consumes the long-format hydraulic datasets from the simulation layer and
emits localized anomaly incidents for the graph-retrieval layer to
explain:

    SensorModel          -- what instruments would actually have read
    ResidualDetector     -- observed vs. twin prediction (primary)
    StatisticalDetector  -- observed vs. the asset's own history (fallback)
    AnomalyEvent         -- one deviating reading
    Incident             -- a time window plus ranked candidate assets,
                            the hand-off point for Phase 4's subgraph BFS
"""

from .base import AnomalyDetector
from .events import AnomalyEvent, AnomalyReport, Incident
from .residual import ResidualDetector
from .sensors import DEFAULT_NOISE, SensorModel, SensorSpec
from .statistical import StatisticalDetector


#: Maps detector names (as used on the command line) to their classes.
DETECTOR_REGISTRY = {
    ResidualDetector.name: ResidualDetector,
    StatisticalDetector.name: StatisticalDetector,
}

__all__ = [
    "AnomalyDetector",
    "AnomalyEvent",
    "AnomalyReport",
    "Incident",
    "ResidualDetector",
    "StatisticalDetector",
    "SensorModel",
    "SensorSpec",
    "DEFAULT_NOISE",
    "DETECTOR_REGISTRY",
]
