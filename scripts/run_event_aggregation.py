"""Aggregate point anomalies into event summaries and evidence tables."""

import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project path setup
# The package is not installed, so src/ has to be on sys.path before
# any twinrag import.
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from twinrag.detection.aligner import align_baseline_and_scenario
from twinrag.detection.batch import (
    _load_manifest,
    _resolve_scenario_path,
)
from twinrag.detection.detector import detect_anomalies
from twinrag.detection.events import (
    aggregate_anomaly_events_with_evidence,
    classify_event_phase,
)
from twinrag.detection.loader import (
    load_baseline_dataset,
    load_scenario_dataset,
)
from twinrag.detection.residuals import calculate_residuals


BASELINE_PATH = Path("data/processed/baseline.csv")
MANIFEST_PATH = Path("data/generated/scenarios_manifest.csv")
PROCESSED_DIRECTORY = Path("data/generated/processed")
OUTPUT_DIRECTORY = Path("data/detection")

EVENTS_OUTPUT_PATH = OUTPUT_DIRECTORY / "anomaly_events.csv"
EVIDENCE_OUTPUT_PATH = OUTPUT_DIRECTORY / "event_evidence.csv"

MAX_EVENT_GAP_S = 3600


def _add_evaluation_labels(
    events: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add offline evaluation labels to events and evidence.

    These labels are derived from simulation ground-truth states and must
    only be used for offline evaluation. They must not be used by the
    deployed detector or graph-reasoning pipeline.
    """
    if events.empty:
        return events, evidence

    events = events.copy()
    evidence = evidence.copy()

    events["evaluation_label"] = (
        events["ground_truth_states"]
        .apply(classify_event_phase)
    )

    if not evidence.empty:
        event_label_mapping = events.set_index(
            "event_id"
        )["evaluation_label"]

        evidence["evaluation_label"] = (
            evidence["event_id"].map(
                event_label_mapping
            )
        )

        if evidence["evaluation_label"].isna().any():
            missing_event_ids = sorted(
                evidence.loc[
                    evidence["evaluation_label"].isna(),
                    "event_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Unable to assign evaluation labels to evidence rows "
                "for event IDs: "
                f"{missing_event_ids}"
            )

    return events, evidence


def _add_scenario_metadata(
    events: pd.DataFrame,
    evidence: pd.DataFrame,
    manifest_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach scenario configuration metadata to output DataFrames."""
    events = events.copy()
    evidence = evidence.copy()

    metadata = {
        "fault_type": str(
            manifest_row["fault_type"]
        ),
        "target_id": str(
            manifest_row["target_id"]
        ),
        "configured_severity": float(
            manifest_row["severity"]
        ),
        "fault_start_hour": int(
            manifest_row["start_hour"]
        ),
        "fault_end_hour": int(
            manifest_row["end_hour"]
        ),
    }

    for column, value in metadata.items():
        events[column] = value

        if not evidence.empty:
            evidence[column] = value

    return events, evidence


def _print_evaluation_summary(
    combined_events: pd.DataFrame,
) -> None:
    """Print counts for each offline event-evaluation label."""
    print()
    print("Event evaluation summary:")

    if (
        combined_events.empty
        or "evaluation_label" not in combined_events.columns
    ):
        print("No evaluation labels available.")
        return

    label_counts = (
        combined_events["evaluation_label"]
        .value_counts()
        .sort_index()
    )

    for label, count in label_counts.items():
        print(f"  {label}: {count}")


def main() -> None:
    """Aggregate anomaly events across all generated scenarios."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_baseline_dataset(
        BASELINE_PATH
    )

    manifest = _load_manifest(
        MANIFEST_PATH
    )

    all_events: list[pd.DataFrame] = []
    all_evidence: list[pd.DataFrame] = []

    total_scenarios = len(manifest)

    print("TwinRAG Module 2 — Event Aggregation")
    print()

    for position, (_, manifest_row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        scenario_name = str(
            manifest_row["scenario"]
        )

        scenario_path = _resolve_scenario_path(
            dataset_file=str(
                manifest_row["dataset_file"]
            ),
            processed_directory=PROCESSED_DIRECTORY,
        )

        scenario = load_scenario_dataset(
            scenario_path
        )

        aligned = align_baseline_and_scenario(
            baseline=baseline,
            scenario=scenario,
        )

        residuals = calculate_residuals(
            aligned
        )

        detected = detect_anomalies(
            residuals
        )

        events, evidence = (
            aggregate_anomaly_events_with_evidence(
                detected_dataframe=detected,
                max_gap_s=MAX_EVENT_GAP_S,
            )
        )

        if not events.empty:
            events, evidence = (
                _add_evaluation_labels(
                    events=events,
                    evidence=evidence,
                )
            )

            events, evidence = (
                _add_scenario_metadata(
                    events=events,
                    evidence=evidence,
                    manifest_row=manifest_row,
                )
            )

            all_events.append(events)

            if not evidence.empty:
                all_evidence.append(evidence)

        print(
            f"[{position:03d}/{total_scenarios:03d}] "
            f"{scenario_name}: "
            f"events={len(events)}, "
            f"evidence={len(evidence)}"
        )

    if all_events:
        combined_events = pd.concat(
            all_events,
            ignore_index=True,
        )

        combined_events = (
            combined_events
            .sort_values(
                [
                    "scenario",
                    "start_timestamp_s",
                    "event_number",
                ]
            )
            .reset_index(drop=True)
        )
    else:
        combined_events = pd.DataFrame()

    if all_evidence:
        combined_evidence = pd.concat(
            all_evidence,
            ignore_index=True,
        )

        combined_evidence = (
            combined_evidence
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
    else:
        combined_evidence = pd.DataFrame()

    combined_events.to_csv(
        EVENTS_OUTPUT_PATH,
        index=False,
    )

    combined_evidence.to_csv(
        EVIDENCE_OUTPUT_PATH,
        index=False,
    )

    scenarios_represented = (
        combined_events["scenario"].nunique()
        if not combined_events.empty
        else 0
    )

    print()
    print("Event aggregation completed.")
    print(f"Total events: {len(combined_events)}")
    print(
        "Scenarios represented: "
        f"{scenarios_represented}"
    )
    print(
        "Total evidence rows: "
        f"{len(combined_evidence)}"
    )

    _print_evaluation_summary(
        combined_events
    )

    print()
    print(
        f"Event report: "
        f"{EVENTS_OUTPUT_PATH}"
    )
    print(
        f"Evidence report: "
        f"{EVIDENCE_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()