"""Point-level residual threshold anomaly detection."""

from enum import Enum

import pandas as pd

from twinrag.detection.config import THRESHOLDS


class AnomalySeverity(str, Enum):
    """Severity assigned to an individual residual observation."""

    NORMAL = "normal"
    ANOMALY = "anomaly"
    WARNING = "warning"
    CRITICAL = "critical"


REQUIRED_COLUMNS = {
    "timestamp_s",
    "asset_id",
    "asset_type",
    "parameter",
    "expected_value",
    "observed_value",
    "residual",
    "abs_residual",
    "scenario",
    "state",
}


def _classify_severity(
    parameter: str,
    abs_residual: float,
) -> AnomalySeverity:
    """Classify one residual value using parameter-specific thresholds."""
    threshold = THRESHOLDS.get(parameter)

    if threshold is None:
        return AnomalySeverity.NORMAL

    if abs_residual >= threshold.critical:
        return AnomalySeverity.CRITICAL

    if abs_residual >= threshold.warning:
        return AnomalySeverity.WARNING

    if abs_residual >= threshold.anomaly:
        return AnomalySeverity.ANOMALY

    return AnomalySeverity.NORMAL


def detect_anomalies(
    residual_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply residual thresholds to aligned Digital Twin measurements.

    Only parameters listed in ``THRESHOLDS`` actively contribute to
    anomaly detection. Other parameters, such as demand, are preserved
    as contextual records and classified as normal.

    Args:
        residual_dataframe: DataFrame produced by
            ``calculate_residuals``.

    Returns:
        A copy of the input DataFrame with the following columns added:

        - ``is_monitored``
        - ``severity``
        - ``is_anomaly``
        - ``threshold_anomaly``
        - ``threshold_warning``
        - ``threshold_critical``

    Raises:
        ValueError: If required columns are missing or residual values
            are invalid.
    """
    missing_columns = REQUIRED_COLUMNS.difference(
        residual_dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Residual DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if residual_dataframe.empty:
        raise ValueError(
            "Residual DataFrame contains no records."
        )

    result = residual_dataframe.copy()

    result["abs_residual"] = pd.to_numeric(
        result["abs_residual"],
        errors="raise",
    )

    if result["abs_residual"].isna().any():
        raise ValueError(
            "abs_residual contains missing values."
        )

    if (result["abs_residual"] < 0).any():
        raise ValueError(
            "abs_residual cannot contain negative values."
        )

    result["is_monitored"] = result["parameter"].isin(
        THRESHOLDS
    )

    result["threshold_anomaly"] = result["parameter"].map(
        lambda parameter: (
            THRESHOLDS[parameter].anomaly
            if parameter in THRESHOLDS
            else pd.NA
        )
    )

    result["threshold_warning"] = result["parameter"].map(
        lambda parameter: (
            THRESHOLDS[parameter].warning
            if parameter in THRESHOLDS
            else pd.NA
        )
    )

    result["threshold_critical"] = result["parameter"].map(
        lambda parameter: (
            THRESHOLDS[parameter].critical
            if parameter in THRESHOLDS
            else pd.NA
        )
    )

    result["severity"] = [
        _classify_severity(
            parameter=str(parameter),
            abs_residual=float(abs_residual),
        ).value
        for parameter, abs_residual in zip(
            result["parameter"],
            result["abs_residual"],
        )
    ]

    result["is_anomaly"] = (
        result["is_monitored"]
        & result["severity"].ne(
            AnomalySeverity.NORMAL.value
        )
    )

    return result