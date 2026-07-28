"""Exact baseline-to-scenario alignment for Module 2."""

import pandas as pd

from twinrag.detection.loader import (
    ALIGNMENT_COLUMNS,
    DetectionDataError,
)


def align_baseline_and_scenario(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
) -> pd.DataFrame:
    """Align baseline and scenario rows using the hydraulic record key.

    The alignment key is:

    - timestamp_s
    - asset_id
    - asset_type
    - parameter

    Args:
        baseline: Validated normal baseline dataset.
        scenario: Validated fault-scenario dataset.

    Returns:
        DataFrame containing matched expected and observed values.

    Raises:
        DetectionDataError: If the datasets cannot be matched one-to-one.
    """
    baseline_view = baseline[
        ALIGNMENT_COLUMNS + ["value"]
    ].rename(
        columns={"value": "expected_value"}
    )

    scenario_view = scenario[
        ALIGNMENT_COLUMNS + ["value", "scenario", "state"]
    ].rename(
        columns={"value": "observed_value"}
    )

    aligned = baseline_view.merge(
        scenario_view,
        on=ALIGNMENT_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_baseline = int((aligned["_merge"] == "left_only").sum())
    unmatched_scenario = int((aligned["_merge"] == "right_only").sum())

    if unmatched_baseline or unmatched_scenario:
        raise DetectionDataError(
            "Baseline and scenario could not be aligned exactly. "
            f"Baseline-only rows: {unmatched_baseline}; "
            f"Scenario-only rows: {unmatched_scenario}."
        )

    aligned = aligned.drop(columns="_merge")

    if len(aligned) != len(baseline) or len(aligned) != len(scenario):
        raise DetectionDataError(
            "Aligned dataset row count does not match the input datasets. "
            f"Baseline rows: {len(baseline)}; "
            f"Scenario rows: {len(scenario)}; "
            f"Aligned rows: {len(aligned)}."
        )

    aligned = aligned.sort_values(
        by=ALIGNMENT_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)

    return aligned