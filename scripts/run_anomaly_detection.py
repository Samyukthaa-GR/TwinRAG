"""Run point-level anomaly detection across all generated scenarios."""

from pathlib import Path

import pandas as pd

from twinrag.detection.aligner import align_baseline_and_scenario
from twinrag.detection.batch import (
    _load_manifest,
    _resolve_scenario_path,
)
from twinrag.detection.detector import detect_anomalies
from twinrag.detection.loader import (
    load_baseline_dataset,
    load_scenario_dataset,
)
from twinrag.detection.residuals import calculate_residuals


BASELINE_PATH = Path("data/processed/baseline.csv")
MANIFEST_PATH = Path("data/generated/scenarios_manifest.csv")
PROCESSED_DIRECTORY = Path("data/generated/processed")
OUTPUT_DIRECTORY = Path("data/detection")

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "scenario_detection_summary.csv"
)


def main() -> None:
    """Evaluate point-level detection across all scenarios."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_baseline_dataset(BASELINE_PATH)
    manifest = _load_manifest(MANIFEST_PATH)

    scenario_summaries: list[dict] = []
    total_scenarios = len(manifest)

    print("TwinRAG Module 2 — Point Anomaly Detection")
    print()

    for index, manifest_row in manifest.iterrows():
        scenario_name = str(manifest_row["scenario"])

        scenario_path = _resolve_scenario_path(
            dataset_file=str(manifest_row["dataset_file"]),
            processed_directory=PROCESSED_DIRECTORY,
        )

        scenario = load_scenario_dataset(scenario_path)

        aligned = align_baseline_and_scenario(
            baseline=baseline,
            scenario=scenario,
        )

        residuals = calculate_residuals(aligned)
        detected = detect_anomalies(residuals)

        normal_mask = detected["state"].eq("normal")
        active_mask = detected["state"].eq("fault_active")
        recovery_mask = detected["state"].eq("recovery")

        normal_anomalies = int(
            detected.loc[
                normal_mask,
                "is_anomaly",
            ].sum()
        )

        active_anomalies = int(
            detected.loc[
                active_mask,
                "is_anomaly",
            ].sum()
        )

        recovery_anomalies = int(
            detected.loc[
                recovery_mask,
                "is_anomaly",
            ].sum()
        )

        active_detected = detected.loc[
            active_mask & detected["is_anomaly"]
        ]

        pressure_anomalies = int(
            active_detected["parameter"]
            .eq("pressure")
            .sum()
        )

        flowrate_anomalies = int(
            active_detected["parameter"]
            .eq("flowrate")
            .sum()
        )

        first_detection_s = (
            int(active_detected["timestamp_s"].min())
            if not active_detected.empty
            else pd.NA
        )

        fault_start_s = int(
            detected.loc[
                active_mask,
                "timestamp_s",
            ].min()
        )

        detection_delay_s = (
            first_detection_s - fault_start_s
            if not pd.isna(first_detection_s)
            else pd.NA
        )

        scenario_summaries.append(
            {
                "scenario": scenario_name,
                "fault_type": manifest_row["fault_type"],
                "target_id": manifest_row["target_id"],
                "severity": manifest_row["severity"],
                "start_hour": manifest_row["start_hour"],
                "end_hour": manifest_row["end_hour"],
                "normal_anomalies": normal_anomalies,
                "fault_active_anomalies": active_anomalies,
                "recovery_anomalies": recovery_anomalies,
                "pressure_anomalies": pressure_anomalies,
                "flowrate_anomalies": flowrate_anomalies,
                "fault_detected": active_anomalies > 0,
                "first_detection_s": first_detection_s,
                "detection_delay_s": detection_delay_s,
            }
        )

        status = (
            "DETECTED"
            if active_anomalies > 0
            else "MISSED"
        )

        print(
            f"[{index + 1:03d}/{total_scenarios:03d}] "
            f"{scenario_name}: {status} "
            f"(active={active_anomalies}, "
            f"pre-fault={normal_anomalies})"
        )

    summary = pd.DataFrame(scenario_summaries)

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print()
    print("Point anomaly detection completed.")
    print(
        "Detected scenarios: "
        f"{int(summary['fault_detected'].sum())}/"
        f"{len(summary)}"
    )
    print(
        "Scenarios with pre-fault anomalies: "
        f"{int((summary['normal_anomalies'] > 0).sum())}"
    )
    print(
        "Maximum detection delay: "
        f"{summary['detection_delay_s'].dropna().max()} seconds"
    )
    print(f"Summary: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()