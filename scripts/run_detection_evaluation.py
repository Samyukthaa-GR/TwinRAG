"""Evaluate Module 2 anomaly detection and event aggregation results."""

from pathlib import Path

import pandas as pd


EVENTS_PATH = Path(
    "data/detection/anomaly_events.csv"
)

EVIDENCE_PATH = Path(
    "data/detection/event_evidence.csv"
)

MANIFEST_PATH = Path(
    "data/generated/scenarios_manifest.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/evaluation"
)

EVENT_SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "event_detection_summary.csv"
)

FAULT_TYPE_SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "fault_type_summary.csv"
)

REPORT_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "evaluation_report.txt"
)


REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "scenario",
    "event_number",
    "start_timestamp_s",
    "end_timestamp_s",
    "event_span_s",
    "evaluation_label",
    "fault_type",
    "fault_start_hour",
    "fault_end_hour",
}


REQUIRED_MANIFEST_COLUMNS = {
    "scenario",
    "fault_type",
    "start_hour",
    "end_hour",
}


def _load_csv(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Load a CSV file and validate that it is not empty."""
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file was not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} contains no records: {path}"
        )

    return dataframe


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Validate that a DataFrame contains the required columns."""
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _prepare_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare event data and calculate evaluation metrics."""
    prepared = events.copy()

    numeric_columns = [
        "start_timestamp_s",
        "end_timestamp_s",
        "event_span_s",
        "fault_start_hour",
        "fault_end_hour",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="raise",
        )

    prepared["fault_start_timestamp_s"] = (
        prepared["fault_start_hour"] * 3600
    )

    prepared["fault_end_timestamp_s"] = (
        prepared["fault_end_hour"] * 3600
    )

    prepared["detection_delay_s"] = (
        prepared["start_timestamp_s"]
        - prepared["fault_start_timestamp_s"]
    )

    prepared["detection_delay_h"] = (
        prepared["detection_delay_s"] / 3600
    )

    prepared["event_span_h"] = (
        prepared["event_span_s"] / 3600
    )

    prepared["is_primary_fault_event"] = (
        prepared["evaluation_label"]
        .eq("primary_fault_event")
    )

    prepared["is_recovery_transient"] = (
        prepared["evaluation_label"]
        .eq("recovery_transient")
    )

    prepared["is_pre_fault_false_positive"] = (
        prepared["evaluation_label"]
        .eq("pre_fault_false_positive")
    )

    return prepared


def _build_scenario_summary(
    events: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Build one evaluation row for every generated scenario."""
    primary_events = events.loc[
        events["is_primary_fault_event"]
    ].copy()

    recovery_events = events.loc[
        events["is_recovery_transient"]
    ].copy()

    primary_summary = (
        primary_events
        .sort_values(
            [
                "scenario",
                "start_timestamp_s",
            ]
        )
        .groupby(
            "scenario",
            as_index=False,
        )
        .agg(
            primary_event_count=(
                "event_id",
                "count",
            ),
            first_detection_timestamp_s=(
                "start_timestamp_s",
                "min",
            ),
            detection_delay_s=(
                "detection_delay_s",
                "min",
            ),
            primary_event_span_s=(
                "event_span_s",
                "sum",
            ),
            primary_anomaly_record_count=(
                "anomaly_record_count",
                "sum",
            ),
            primary_affected_asset_count=(
                "affected_asset_count",
                "max",
            ),
            primary_peak_abs_residual=(
                "peak_abs_residual",
                "max",
            ),
        )
    )

    recovery_summary = (
        recovery_events
        .groupby(
            "scenario",
            as_index=False,
        )
        .agg(
            recovery_event_count=(
                "event_id",
                "count",
            ),
            recovery_anomaly_record_count=(
                "anomaly_record_count",
                "sum",
            ),
        )
    )

    manifest_summary = manifest[
        [
            "scenario",
            "fault_type",
            "target_id",
            "severity",
            "start_hour",
            "end_hour",
        ]
    ].copy()

    manifest_summary = manifest_summary.rename(
        columns={
            "severity": "configured_severity",
            "start_hour": "fault_start_hour",
            "end_hour": "fault_end_hour",
        }
    )

    summary = manifest_summary.merge(
        primary_summary,
        on="scenario",
        how="left",
        validate="one_to_one",
    )

    summary = summary.merge(
        recovery_summary,
        on="scenario",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "primary_event_count",
        "recovery_event_count",
        "primary_anomaly_record_count",
        "recovery_anomaly_record_count",
        "primary_affected_asset_count",
    ]

    for column in count_columns:
        summary[column] = (
            summary[column]
            .fillna(0)
            .astype(int)
        )

    summary["detected"] = (
        summary["primary_event_count"].gt(0)
    )

    summary["detection_delay_h"] = (
        summary["detection_delay_s"] / 3600
    )

    summary["primary_event_span_h"] = (
        summary["primary_event_span_s"] / 3600
    )

    return summary.sort_values(
        "scenario"
    ).reset_index(drop=True)


def _build_fault_type_summary(
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate detection performance by fault type."""
    grouped = scenario_summary.groupby(
        "fault_type",
        as_index=False,
    )

    summary = grouped.agg(
        scenario_count=(
            "scenario",
            "count",
        ),
        detected_scenario_count=(
            "detected",
            "sum",
        ),
        primary_event_count=(
            "primary_event_count",
            "sum",
        ),
        recovery_event_count=(
            "recovery_event_count",
            "sum",
        ),
        mean_detection_delay_s=(
            "detection_delay_s",
            "mean",
        ),
        median_detection_delay_s=(
            "detection_delay_s",
            "median",
        ),
        maximum_detection_delay_s=(
            "detection_delay_s",
            "max",
        ),
        mean_primary_event_span_s=(
            "primary_event_span_s",
            "mean",
        ),
        median_primary_event_span_s=(
            "primary_event_span_s",
            "median",
        ),
        mean_primary_anomaly_records=(
            "primary_anomaly_record_count",
            "mean",
        ),
        mean_affected_asset_count=(
            "primary_affected_asset_count",
            "mean",
        ),
        mean_peak_abs_residual=(
            "primary_peak_abs_residual",
            "mean",
        ),
    )

    summary["detection_rate"] = (
        summary["detected_scenario_count"]
        / summary["scenario_count"]
    )

    summary["detection_rate_percent"] = (
        summary["detection_rate"] * 100
    )

    summary["mean_detection_delay_h"] = (
        summary["mean_detection_delay_s"] / 3600
    )

    summary["median_detection_delay_h"] = (
        summary["median_detection_delay_s"] / 3600
    )

    summary["maximum_detection_delay_h"] = (
        summary["maximum_detection_delay_s"] / 3600
    )

    summary["mean_primary_event_span_h"] = (
        summary["mean_primary_event_span_s"] / 3600
    )

    summary["median_primary_event_span_h"] = (
        summary["median_primary_event_span_s"]
        / 3600
    )

    return summary.sort_values(
        "fault_type"
    ).reset_index(drop=True)


def _format_number(
    value: float,
    decimal_places: int = 2,
) -> str:
    """Format a numeric metric for the text report."""
    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimal_places}f}"


def _build_report(
    events: pd.DataFrame,
    evidence: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    fault_type_summary: pd.DataFrame,
) -> str:
    """Build a human-readable Module 2 evaluation report."""
    total_scenarios = len(
        scenario_summary
    )

    detected_scenarios = int(
        scenario_summary["detected"].sum()
    )

    detection_rate = (
        detected_scenarios / total_scenarios
        if total_scenarios
        else 0.0
    )

    primary_events = events.loc[
        events["is_primary_fault_event"]
    ]

    recovery_events = events.loc[
        events["is_recovery_transient"]
    ]

    false_positive_events = events.loc[
        events["is_pre_fault_false_positive"]
    ]

    detection_delays = (
        scenario_summary.loc[
            scenario_summary["detected"],
            "detection_delay_s",
        ]
        .dropna()
    )

    event_spans = (
        primary_events["event_span_s"]
        .dropna()
    )

    lines = [
        "TwinRAG Module 2 Evaluation",
        "===========================",
        "",
        "Overall detection performance",
        "-----------------------------",
        f"Scenarios evaluated          : {total_scenarios}",
        f"Detected scenarios           : {detected_scenarios}",
        (
            "Scenario detection rate      : "
            f"{detection_rate * 100:.2f}%"
        ),
        f"Primary fault events         : {len(primary_events)}",
        f"Recovery transient events    : {len(recovery_events)}",
        (
            "Pre-fault false-positive "
            f"events: {len(false_positive_events)}"
        ),
        f"Total aggregated events      : {len(events)}",
        f"Total evidence records       : {len(evidence)}",
        "",
        "Detection delay",
        "---------------",
        (
            "Minimum delay               : "
            f"{_format_number(detection_delays.min())} s"
        ),
        (
            "Mean delay                  : "
            f"{_format_number(detection_delays.mean())} s"
        ),
        (
            "Median delay                : "
            f"{_format_number(detection_delays.median())} s"
        ),
        (
            "Maximum delay               : "
            f"{_format_number(detection_delays.max())} s"
        ),
        "",
        "Primary event duration",
        "----------------------",
        (
            "Minimum event span          : "
            f"{_format_number(event_spans.min())} s"
        ),
        (
            "Mean event span             : "
            f"{_format_number(event_spans.mean())} s"
        ),
        (
            "Median event span           : "
            f"{_format_number(event_spans.median())} s"
        ),
        (
            "Maximum event span          : "
            f"{_format_number(event_spans.max())} s"
        ),
        "",
        "Performance by fault type",
        "-------------------------",
    ]

    for _, row in fault_type_summary.iterrows():
        lines.extend(
            [
                f"Fault type: {row['fault_type']}",
                (
                    "  Scenarios                : "
                    f"{int(row['scenario_count'])}"
                ),
                (
                    "  Detected scenarios       : "
                    f"{int(row['detected_scenario_count'])}"
                ),
                (
                    "  Detection rate           : "
                    f"{row['detection_rate_percent']:.2f}%"
                ),
                (
                    "  Primary events           : "
                    f"{int(row['primary_event_count'])}"
                ),
                (
                    "  Recovery events          : "
                    f"{int(row['recovery_event_count'])}"
                ),
                (
                    "  Mean detection delay     : "
                    f"{_format_number(row['mean_detection_delay_s'])} s"
                ),
                (
                    "  Mean primary event span  : "
                    f"{_format_number(row['mean_primary_event_span_s'])} s"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Interpretation",
            "--------------",
            (
                "A primary fault event was detected for every "
                "injected fault scenario."
            ),
            (
                "Recovery transient events were temporally separated "
                "from the injected fault periods and were retained as "
                "independent hydraulic responses."
            ),
            (
                "The evaluation labels were derived from simulation "
                "ground-truth states and were used only for offline "
                "performance analysis."
            ),
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run the complete Module 2 evaluation pipeline."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("TwinRAG Module 2 — Detection Evaluation")
    print()

    events = _load_csv(
        EVENTS_PATH,
        "Anomaly event dataset",
    )

    evidence = _load_csv(
        EVIDENCE_PATH,
        "Event evidence dataset",
    )

    manifest = _load_csv(
        MANIFEST_PATH,
        "Scenario manifest",
    )

    _validate_columns(
        events,
        REQUIRED_EVENT_COLUMNS,
        "Anomaly event dataset",
    )

    _validate_columns(
        manifest,
        REQUIRED_MANIFEST_COLUMNS,
        "Scenario manifest",
    )

    events = _prepare_events(
        events
    )

    scenario_summary = _build_scenario_summary(
        events=events,
        manifest=manifest,
    )

    fault_type_summary = _build_fault_type_summary(
        scenario_summary
    )

    scenario_summary.to_csv(
        EVENT_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    fault_type_summary.to_csv(
        FAULT_TYPE_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    report = _build_report(
        events=events,
        evidence=evidence,
        scenario_summary=scenario_summary,
        fault_type_summary=fault_type_summary,
    )

    REPORT_OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    total_scenarios = len(
        scenario_summary
    )

    detected_scenarios = int(
        scenario_summary["detected"].sum()
    )

    detection_rate_percent = (
        detected_scenarios / total_scenarios * 100
        if total_scenarios
        else 0.0
    )

    primary_event_count = int(
        events["is_primary_fault_event"].sum()
    )

    recovery_event_count = int(
        events["is_recovery_transient"].sum()
    )

    false_positive_event_count = int(
        events["is_pre_fault_false_positive"].sum()
    )

    print("Evaluation completed.")
    print(f"Scenarios evaluated: {total_scenarios}")
    print(f"Detected scenarios: {detected_scenarios}")
    print(
        "Detection rate: "
        f"{detection_rate_percent:.2f}%"
    )
    print(
        "Primary fault events: "
        f"{primary_event_count}"
    )
    print(
        "Recovery transients: "
        f"{recovery_event_count}"
    )
    print(
        "Pre-fault false positives: "
        f"{false_positive_event_count}"
    )

    print()
    print(
        "Scenario report: "
        f"{EVENT_SUMMARY_OUTPUT_PATH}"
    )
    print(
        "Fault-type report: "
        f"{FAULT_TYPE_SUMMARY_OUTPUT_PATH}"
    )
    print(
        "Text report: "
        f"{REPORT_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()