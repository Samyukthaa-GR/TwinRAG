"""Residual profiling utilities for anomaly-detector calibration."""

import pandas as pd

from twinrag.detection.loader import DetectionDataError


REQUIRED_RESIDUAL_COLUMNS = {
    "timestamp_s",
    "asset_id",
    "asset_type",
    "parameter",
    "scenario",
    "state",
    "residual",
    "abs_residual",
}


def summarize_residuals_by_parameter(
    residual_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize residual magnitude for each state and parameter.

    Args:
        residual_data: Dataset produced by ``calculate_residuals``.

    Returns:
        Summary containing residual counts, mean, quantiles and maximum.
    """
    missing_columns = REQUIRED_RESIDUAL_COLUMNS.difference(
        residual_data.columns
    )

    if missing_columns:
        raise DetectionDataError(
            "Residual dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if residual_data.empty:
        raise DetectionDataError(
            "Cannot profile an empty residual dataset."
        )

    summary = (
        residual_data.groupby(
            ["state", "parameter"],
            as_index=False,
        )
        .agg(
            total_records=("abs_residual", "size"),
            changed_records=(
                "abs_residual",
                lambda values: int((values > 0).sum()),
            ),
            mean_abs_residual=("abs_residual", "mean"),
            median_abs_residual=("abs_residual", "median"),
            p95_abs_residual=(
                "abs_residual",
                lambda values: values.quantile(0.95),
            ),
            p99_abs_residual=(
                "abs_residual",
                lambda values: values.quantile(0.99),
            ),
            max_abs_residual=("abs_residual", "max"),
        )
    )

    summary["changed_fraction"] = (
        summary["changed_records"] / summary["total_records"]
    )

    return summary


def rank_affected_assets(
    residual_data: pd.DataFrame,
    state: str = "fault_active",
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank assets by their strongest residual during a selected state.

    Args:
        residual_data: Dataset produced by ``calculate_residuals``.
        state: Ground-truth state to analyse.
        top_n: Maximum number of asset-parameter records returned.

    Returns:
        Ranked asset-parameter residual summary.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    selected = residual_data.loc[
        residual_data["state"] == state
    ].copy()

    if selected.empty:
        raise DetectionDataError(
            f"No residual records found for state: {state}"
        )

    ranking = (
        selected.groupby(
            ["asset_id", "asset_type", "parameter"],
            as_index=False,
        )
        .agg(
            mean_abs_residual=("abs_residual", "mean"),
            max_abs_residual=("abs_residual", "max"),
            peak_signed_residual=(
                "residual",
                lambda values: values.loc[
                    values.abs().idxmax()
                ],
            ),
            first_changed_timestamp=(
                "timestamp_s",
                lambda timestamps: timestamps.loc[
                    selected.loc[
                        timestamps.index,
                        "abs_residual",
                    ].gt(0)
                ].min(),
            ),
        )
        .sort_values(
            "max_abs_residual",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return ranking