"""Aggregation of point-level anomalies into coherent anomaly events."""

from enum import IntEnum

import pandas as pd


class SeverityRank(IntEnum):
    """Numeric ranking used to compare anomaly severities."""

    NORMAL = 0
    ANOMALY = 1
    WARNING = 2
    CRITICAL = 3


SEVERITY_TO_RANK = {
    "normal": SeverityRank.NORMAL,
    "anomaly": SeverityRank.ANOMALY,
    "warning": SeverityRank.WARNING,
    "critical": SeverityRank.CRITICAL,
}


RANK_TO_SEVERITY = {
    rank: severity
    for severity, rank in SEVERITY_TO_RANK.items()
}


REQUIRED_COLUMNS = {
    "timestamp_s",
    "asset_id",
    "asset_type",
    "parameter",
    "abs_residual",
    "scenario",
    "state",
    "severity",
    "is_anomaly",
}


EVENT_COLUMNS = [
    "event_id",
    "scenario",
    "event_number",
    "start_timestamp_s",
    "end_timestamp_s",
    "event_span_s",
    "timestamp_count",
    "anomaly_record_count",
    "peak_severity",
    "peak_abs_residual",
    "mean_abs_residual",
    "affected_asset_count",
    "affected_assets",
    "affected_asset_types",
    "affected_parameters",
    "pressure_anomaly_count",
    "flowrate_anomaly_count",
    "ground_truth_states",
]


def _validate_detected_dataframe(
    detected_dataframe: pd.DataFrame,
) -> None:
    """Validate the point-level anomaly detector output."""
    missing_columns = REQUIRED_COLUMNS.difference(
        detected_dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Detected DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if detected_dataframe.empty:
        raise ValueError(
            "Detected DataFrame contains no records."
        )

    invalid_severities = set(
        detected_dataframe["severity"]
        .dropna()
        .astype(str)
        .unique()
    ).difference(SEVERITY_TO_RANK)

    if invalid_severities:
        raise ValueError(
            "Detected DataFrame contains unsupported severity values: "
            f"{sorted(invalid_severities)}"
        )

    if detected_dataframe["timestamp_s"].isna().any():
        raise ValueError(
            "timestamp_s contains missing values."
        )

    if detected_dataframe["scenario"].isna().any():
        raise ValueError(
            "scenario contains missing values."
        )


def _join_unique_values(
    series: pd.Series,
) -> str:
    """Return unique non-null values as a sorted pipe-separated string."""
    values = {
        str(value)
        for value in series.dropna()
    }

    return "|".join(sorted(values))


def _get_peak_severity(
    series: pd.Series,
) -> str:
    """Return the highest severity represented in a series."""
    severity_ranks = series.map(SEVERITY_TO_RANK)

    if severity_ranks.isna().any():
        invalid_values = (
            series.loc[severity_ranks.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unable to rank unsupported severity values: "
            f"{sorted(invalid_values)}"
        )

    peak_rank = SeverityRank(
        int(severity_ranks.max())
    )

    return RANK_TO_SEVERITY[peak_rank]


def _assign_event_groups(
    anomaly_records: pd.DataFrame,
    max_gap_s: int,
) -> pd.DataFrame:
    """Assign contiguous anomalous timestamps to event groups.

    A new event starts when the gap between two consecutive anomalous
    timestamps in the same scenario is greater than ``max_gap_s``.
    """
    result = anomaly_records.copy()

    unique_timestamps = (
        result[
            [
                "scenario",
                "timestamp_s",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "scenario",
                "timestamp_s",
            ]
        )
        .reset_index(drop=True)
    )

    timestamp_differences = (
        unique_timestamps
        .groupby(
            "scenario",
            sort=False,
        )["timestamp_s"]
        .diff()
    )

    unique_timestamps["new_event"] = (
        timestamp_differences.isna()
        | timestamp_differences.gt(max_gap_s)
    )

    unique_timestamps["event_group"] = (
        unique_timestamps
        .groupby(
            "scenario",
            sort=False,
        )["new_event"]
        .cumsum()
        .astype(int)
    )

    result = result.merge(
        unique_timestamps[
            [
                "scenario",
                "timestamp_s",
                "event_group",
            ]
        ],
        on=[
            "scenario",
            "timestamp_s",
        ],
        how="left",
        validate="many_to_one",
    )

    return result


def classify_event_phase(
    ground_truth_states: str,
) -> str:
    """Assign an offline evaluation label to an anomaly event.

    This function uses simulation ground-truth states and must only be
    used during offline evaluation. It must not be used by the deployed
    anomaly detector or graph reasoning pipeline.

    Args:
        ground_truth_states: Pipe-separated state values contained in an
            event, such as ``fault_active`` or
            ``fault_active|recovery``.

    Returns:
        One of:

        - ``primary_fault_event``
        - ``recovery_transient``
        - ``pre_fault_false_positive``
        - ``mixed_or_unknown``
    """
    states = {
        state.strip()
        for state in str(ground_truth_states).split("|")
        if state.strip()
    }

    if "fault_active" in states:
        return "primary_fault_event"

    if states == {"recovery"}:
        return "recovery_transient"

    if states == {"normal"}:
        return "pre_fault_false_positive"

    return "mixed_or_unknown"


def aggregate_anomaly_events(
    detected_dataframe: pd.DataFrame,
    max_gap_s: int = 3600,
) -> pd.DataFrame:
    """Aggregate point-level anomalies into network-level events.

    An anomaly event consists of consecutive anomalous timestamps within
    the same scenario. A new event begins when the time between two
    anomalous timestamps is greater than ``max_gap_s``.

    Args:
        detected_dataframe: DataFrame produced by
            ``detect_anomalies``.
        max_gap_s: Maximum allowed gap, in seconds, between consecutive
            anomalous timestamps belonging to the same event.

    Returns:
        An event-level DataFrame containing timing, severity, residual,
        parameter and affected-asset information.

        If no anomalous records are present, an empty DataFrame with the
        expected event columns is returned.

    Raises:
        ValueError: If the input is invalid or ``max_gap_s`` is negative.
    """
    _validate_detected_dataframe(detected_dataframe)

    if max_gap_s < 0:
        raise ValueError(
            "max_gap_s cannot be negative."
        )

    anomaly_records = detected_dataframe.loc[
        detected_dataframe["is_anomaly"].astype(bool)
    ].copy()

    if anomaly_records.empty:
        return pd.DataFrame(
            columns=EVENT_COLUMNS
        )

    anomaly_records["timestamp_s"] = pd.to_numeric(
        anomaly_records["timestamp_s"],
        errors="raise",
    )

    anomaly_records["abs_residual"] = pd.to_numeric(
        anomaly_records["abs_residual"],
        errors="raise",
    )

    anomaly_records = _assign_event_groups(
        anomaly_records=anomaly_records,
        max_gap_s=max_gap_s,
    )

    grouped = anomaly_records.groupby(
        [
            "scenario",
            "event_group",
        ],
        sort=True,
        dropna=False,
    )

    event_rows: list[dict] = []

    for (
        scenario_name,
        event_group,
    ), event_data in grouped:
        event_data = event_data.sort_values(
            [
                "timestamp_s",
                "parameter",
                "asset_id",
            ]
        )

        start_timestamp = int(
            event_data["timestamp_s"].min()
        )

        end_timestamp = int(
            event_data["timestamp_s"].max()
        )

        affected_assets = sorted(
            event_data["asset_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        event_number = int(event_group)

        event_rows.append(
            {
                "event_id": (
                    f"{scenario_name}_event_"
                    f"{event_number:02d}"
                ),
                "scenario": str(scenario_name),
                "event_number": event_number,
                "start_timestamp_s": start_timestamp,
                "end_timestamp_s": end_timestamp,
                "event_span_s": (
                    end_timestamp - start_timestamp
                ),
                "timestamp_count": int(
                    event_data["timestamp_s"].nunique()
                ),
                "anomaly_record_count": int(
                    len(event_data)
                ),
                "peak_severity": _get_peak_severity(
                    event_data["severity"]
                ),
                "peak_abs_residual": float(
                    event_data["abs_residual"].max()
                ),
                "mean_abs_residual": float(
                    event_data["abs_residual"].mean()
                ),
                "affected_asset_count": int(
                    len(affected_assets)
                ),
                "affected_assets": "|".join(
                    affected_assets
                ),
                "affected_asset_types": (
                    _join_unique_values(
                        event_data["asset_type"]
                    )
                ),
                "affected_parameters": (
                    _join_unique_values(
                        event_data["parameter"]
                    )
                ),
                "pressure_anomaly_count": int(
                    event_data["parameter"]
                    .eq("pressure")
                    .sum()
                ),
                "flowrate_anomaly_count": int(
                    event_data["parameter"]
                    .eq("flowrate")
                    .sum()
                ),
                "ground_truth_states": (
                    _join_unique_values(
                        event_data["state"]
                    )
                ),
            }
        )

    events = pd.DataFrame(
        event_rows,
        columns=EVENT_COLUMNS,
    )

    return (
        events
        .sort_values(
            [
                "scenario",
                "start_timestamp_s",
                "event_number",
            ]
        )
        .reset_index(drop=True)
    )


def aggregate_anomaly_events_with_evidence(
    detected_dataframe: pd.DataFrame,
    max_gap_s: int = 3600,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate anomalies and retain event-linked evidence records.

    Args:
        detected_dataframe: DataFrame produced by
            ``detect_anomalies``.
        max_gap_s: Maximum gap between consecutive anomalous timestamps
            belonging to the same event.

    Returns:
        A tuple containing:

        - event-level summary DataFrame;
        - point-level anomaly evidence with event identifiers.

    Raises:
        ValueError: If the input is invalid or ``max_gap_s`` is negative.
    """
    _validate_detected_dataframe(detected_dataframe)

    if max_gap_s < 0:
        raise ValueError(
            "max_gap_s cannot be negative."
        )

    events = aggregate_anomaly_events(
        detected_dataframe=detected_dataframe,
        max_gap_s=max_gap_s,
    )

    anomaly_records = detected_dataframe.loc[
        detected_dataframe["is_anomaly"].astype(bool)
    ].copy()

    if anomaly_records.empty:
        evidence_columns = [
            "event_id",
            "event_number",
            *detected_dataframe.columns.tolist(),
        ]

        return (
            events,
            pd.DataFrame(
                columns=evidence_columns
            ),
        )

    anomaly_records["timestamp_s"] = pd.to_numeric(
        anomaly_records["timestamp_s"],
        errors="raise",
    )

    anomaly_records["abs_residual"] = pd.to_numeric(
        anomaly_records["abs_residual"],
        errors="raise",
    )

    evidence = _assign_event_groups(
        anomaly_records=anomaly_records,
        max_gap_s=max_gap_s,
    )

    evidence = evidence.rename(
        columns={
            "event_group": "event_number",
        }
    )

    evidence["event_number"] = (
        evidence["event_number"].astype(int)
    )

    evidence["event_id"] = (
        evidence["scenario"].astype(str)
        + "_event_"
        + evidence["event_number"]
        .astype(str)
        .str.zfill(2)
    )

    preferred_columns = [
        "event_id",
        "event_number",
        "scenario",
        "timestamp_s",
        "asset_id",
        "asset_type",
        "parameter",
        "expected_value",
        "observed_value",
        "residual",
        "abs_residual",
        "severity",
        "is_anomaly",
        "state",
    ]

    available_preferred_columns = [
        column
        for column in preferred_columns
        if column in evidence.columns
    ]

    remaining_columns = [
        column
        for column in evidence.columns
        if column not in available_preferred_columns
    ]

    evidence = evidence[
        available_preferred_columns
        + remaining_columns
    ]

    evidence = (
        evidence
        .sort_values(
            [
                "scenario",
                "event_number",
                "timestamp_s",
                "parameter",
                "asset_id",
            ]
        )
        .reset_index(drop=True)
    )

    return events, evidence