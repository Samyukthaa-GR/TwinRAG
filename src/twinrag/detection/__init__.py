"""
Anomaly detection (Phase 3).

Consumes the long-format hydraulic datasets from the simulation layer and
emits localized anomalies for the graph-retrieval layer to explain.

**Two independent implementations live side by side.** They solve the
same problem with different designs, and both are kept deliberately —
neither is a wrapper around the other. Pick one per consumer and do not
interleave them; they do not share types.

Dataclass pipeline (in-memory, object-returning)
------------------------------------------------
    SensorModel          -- what instruments would actually have read
    ResidualDetector     -- observed vs. twin prediction, scored in sigma
    StatisticalDetector  -- observed vs. the asset's own history
    AnomalyEvent         -- one deviating reading            (incidents.py)
    Incident             -- window + ranked candidate assets (incidents.py)
    AnomalyReport        -- one detector's findings          (incidents.py)

Entry points: ``scripts/detect_anomalies.py``,
``scripts/evaluate_detection.py``. Detectors expose a single method,
``detect(dataset) -> AnomalyReport``. Thresholds are expressed in
standard deviations of sensor noise, so one setting spans metres and
m3/s.

DataFrame pipeline (CSV-oriented, table-returning)
--------------------------------------------------
    load_baseline_dataset / load_scenario_dataset / validate_detection_dataset
    align_baseline_and_scenario   -- pair baseline and scenario rows
    calculate_residuals           -- per-reading deviation
    detect_anomalies              -- severity tiers from fixed thresholds
    aggregate_anomaly_events      -- point anomalies -> event rows
    classify_event_phase          -- label an event against ground truth
    rank_affected_assets / summarize_residuals_by_parameter
    profile_all_scenarios         -- batch profiling across the whole set

Entry points: ``scripts/run_anomaly_detection.py``,
``scripts/run_anomaly_profiling.py``, ``scripts/run_event_aggregation.py``,
``scripts/run_detection_evaluation.py``. Thresholds are absolute
per-parameter values in ``config.py``.

Naming note
-----------
``events.py`` belongs to the DataFrame pipeline (event *aggregation*).
The dataclass pipeline's ``AnomalyEvent``/``Incident``/``AnomalyReport``
live in ``incidents.py``. Both are re-exported here, so importing from
``twinrag.detection`` is unaffected by which file a name sits in.
"""

# ----------------------------------------------------------------------
# Dataclass pipeline
# ----------------------------------------------------------------------

from .base import AnomalyDetector
from .incidents import AnomalyEvent, AnomalyReport, Incident
from .residual import ResidualDetector
from .sensors import DEFAULT_NOISE, SensorModel, SensorSpec
from .statistical import StatisticalDetector

# ----------------------------------------------------------------------
# DataFrame pipeline
# ----------------------------------------------------------------------

from .aligner import align_baseline_and_scenario
from .batch import profile_all_scenarios
from .config import (
    FLOWRATE_THRESHOLD,
    PRESSURE_THRESHOLD,
    THRESHOLDS,
    ParameterThreshold,
)
from .detector import AnomalySeverity, detect_anomalies
from .events import (
    aggregate_anomaly_events,
    aggregate_anomaly_events_with_evidence,
    classify_event_phase,
)
from .loader import (
    ALIGNMENT_COLUMNS,
    REQUIRED_COLUMNS,
    SUPPORTED_PARAMETERS,
    DetectionDataError,
    load_baseline_dataset,
    load_scenario_dataset,
    validate_detection_dataset,
)
from .profiling import rank_affected_assets, summarize_residuals_by_parameter
from .residuals import calculate_residuals

# ----------------------------------------------------------------------
# Graph re-exports, kept for compatibility
#
# These names were previously exported from this package as
# ``from .builder import ...`` / ``from .validation import ...``, which
# could not work: those modules live in ``twinrag.graph``, not here, so
# importing ``twinrag.detection`` raised ImportError. They are re-exported
# from their real home so existing imports keep working.
#
# ``twinrag.graph`` remains the canonical place to import them from.
# ----------------------------------------------------------------------

from twinrag.graph import GraphSnapshot, NetworkGraphBuilder
from twinrag.graph.validation import GraphValidationError, GraphValidator


#: Maps detector names (as used on the command line) to their classes.
#: Dataclass pipeline only — the DataFrame pipeline's ``detect_anomalies``
#: is a function, not an ``AnomalyDetector`` subclass.
DETECTOR_REGISTRY = {
    ResidualDetector.name: ResidualDetector,
    StatisticalDetector.name: StatisticalDetector,
}

__all__ = [
    # Dataclass pipeline
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
    # DataFrame pipeline
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
    "classify_event_phase",
    "ParameterThreshold",
    "PRESSURE_THRESHOLD",
    "FLOWRATE_THRESHOLD",
    "THRESHOLDS",
    # Graph re-exports (canonical home: twinrag.graph)
    "GraphSnapshot",
    "NetworkGraphBuilder",
    "GraphValidationError",
    "GraphValidator",
]
