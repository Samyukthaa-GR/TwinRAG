"""Statistical anomaly detection components for TwinRAG."""
from twinrag.detection.aligner import align_baseline_and_scenario
from twinrag.detection.residuals import calculate_residuals
from twinrag.detection.batch import profile_all_scenarios
from twinrag.detection.events import aggregate_anomaly_events
from .builder import NetworkGraphBuilder
from .snapshot import GraphSnapshot
from twinrag.detection.profiling import (
    rank_affected_assets,
    summarize_residuals_by_parameter,
)
from twinrag.detection.loader import (
    ALIGNMENT_COLUMNS,
    REQUIRED_COLUMNS,
    SUPPORTED_PARAMETERS,
    DetectionDataError,
    load_baseline_dataset,
    load_scenario_dataset,
    validate_detection_dataset,
)
from twinrag.detection.detector import (
    AnomalySeverity,
    detect_anomalies,
)
from twinrag.detection.events import (
    aggregate_anomaly_events,
    aggregate_anomaly_events_with_evidence,
)
from twinrag.detection.events import (
    aggregate_anomaly_events,
    aggregate_anomaly_events_with_evidence,
    classify_event_phase,
)
from .validation import (
    GraphValidationError,
    GraphValidator,
)
__all__ = [
    "ALIGNMENT_COLUMNS",
    "REQUIRED_COLUMNS",
    "SUPPORTED_PARAMETERS",
    "DetectionDataError",
    "load_baseline_dataset",
    "load_scenario_dataset",
    "validate_detection_dataset",
    "align_baseline_and_scenario",
    "calculate_residuals",
    "rank_affected_assets",
    "summarize_residuals_by_parameter",
    "profile_all_scenarios",
    "AnomalySeverity",
    "detect_anomalies",
    "aggregate_anomaly_events",
    "aggregate_anomaly_events_with_evidence",
    "aggregate_anomaly_events",
    "aggregate_anomaly_events_with_evidence",
    "classify_event_phase",
    "GraphSnapshot",
    "NetworkGraphBuilder",
    "GraphValidationError",
    "GraphValidator",
]
