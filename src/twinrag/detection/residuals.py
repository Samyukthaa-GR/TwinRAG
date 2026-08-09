"""Residual calculation utilities for TwinRAG anomaly detection."""

import numpy as np
import pandas as pd

from twinrag.detection.loader import DetectionDataError


REQUIRED_ALIGNED_COLUMNS = {
    "timestamp_s",
    "asset_id",
    "asset_type",
    "parameter",
    "expected_value",
    "observed_value",
    "scenario",
    "state",
}


def calculate_residuals(
    aligned_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate signed and absolute residuals.

    Residuals compare fault-scenario observations against the normal
    Digital Twin baseline:

        residual = observed_value - expected_value

    Args:
        aligned_data: Exactly aligned baseline and scenario records.

    Returns:
        Copy of the aligned dataset containing:
        - residual
        - abs_residual

    Raises:
        DetectionDataError: If required columns are missing or numerical
            values are invalid.
    """
    missing_columns = REQUIRED_ALIGNED_COLUMNS.difference(
        aligned_data.columns
    )

    if missing_columns:
        raise DetectionDataError(
            "Aligned dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if aligned_data.empty:
        raise DetectionDataError(
            "Cannot calculate residuals from an empty dataset."
        )

    result = aligned_data.copy()

    expected = pd.to_numeric(
        result["expected_value"],
        errors="coerce",
    )
    observed = pd.to_numeric(
        result["observed_value"],
        errors="coerce",
    )

    invalid_expected = int(expected.isna().sum())
    invalid_observed = int(observed.isna().sum())

    if invalid_expected or invalid_observed:
        raise DetectionDataError(
            "Expected and observed values must be numeric. "
            f"Invalid expected values: {invalid_expected}; "
            f"invalid observed values: {invalid_observed}."
        )

    result["expected_value"] = expected
    result["observed_value"] = observed

    result["residual"] = (
        result["observed_value"] - result["expected_value"]
    )

    result["abs_residual"] = result["residual"].abs()

    finite_mask = np.isfinite(
        result[
            [
                "expected_value",
                "observed_value",
                "residual",
                "abs_residual",
            ]
        ].to_numpy()
    )

    if not finite_mask.all():
        raise DetectionDataError(
            "Residual calculation produced non-finite numerical values."
        )

    return result