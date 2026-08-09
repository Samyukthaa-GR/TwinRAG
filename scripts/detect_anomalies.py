"""
Run anomaly detection across the generated scenario batch.

Reads data/generated/scenarios_manifest.csv, runs the chosen detector on
every scenario dataset, and writes the results for
scripts/evaluate_detection.py to score.

The unfaulted baseline is always processed too, as a negative control.
Its readings differ from the twin's prediction by sensor noise alone, so
whatever fires there is a false alarm by construction — that run, not the
fault runs, is what measures the false-alarm rate.

    python scripts/detect_anomalies.py
    python scripts/detect_anomalies.py --detector statistical
    python scripts/detect_anomalies.py --threshold 6 --min-assets 5
    python scripts/detect_anomalies.py --no-noise --name residual_noisefree
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# --------------------------------------------------
# TwinRAG imports
# These must come AFTER src is added to sys.path
# --------------------------------------------------

import pandas as pd

from twinrag.detection import (
    DEFAULT_NOISE,
    DETECTOR_REGISTRY,
    ResidualDetector,
    SensorModel,
    SensorSpec,
)


GENERATED_ROOT = PROJECT_ROOT / "data" / "generated"
MANIFEST_PATH = GENERATED_ROOT / "scenarios_manifest.csv"
BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "baseline.csv"

DETECTION_ROOT = GENERATED_ROOT / "detection"

BASELINE_SCENARIO = "normal"


def _resolve(path_str: str) -> Path:
    """
    Resolve a manifest path relative to the project root.

    The manifest is written by csv on Windows, so it carries backslash
    separators; normalise them so the same file works on POSIX too.
    """

    return PROJECT_ROOT / Path(str(path_str).replace("\\", "/"))


def _sensor_model(args) -> SensorModel:
    """
    Build the sensor model from the command-line noise settings.
    """

    specs = {
        parameter: SensorSpec(
            absolute=absolute * args.noise_scale,
            relative=relative * args.noise_scale,
        )
        for parameter, (absolute, relative) in DEFAULT_NOISE.items()
    }

    return SensorModel(
        specs=specs,
        seed=args.seed,
        noise_enabled=not args.no_noise,
    )


def _build_detector(args, baseline):
    """
    Instantiate the requested detector.
    """

    detector_cls = DETECTOR_REGISTRY[args.detector]

    shared = {
        "sensor_model": _sensor_model(args),
        "score_threshold": args.threshold,
        "min_assets": args.min_assets,
    }

    if detector_cls is ResidualDetector:
        return detector_cls(baseline=baseline, **shared)

    return detector_cls(**shared)


def _summary_row(report) -> dict:
    """
    Flatten one report into a summary line.
    """

    incident = report.incidents[0] if report.incidents else None

    top = incident.top_candidates(5) if incident else []

    return {
        "scenario": report.scenario,
        "detector": report.detector,
        "detected": int(report.detected),
        "first_detection_s": (
            report.first_detection_s
            if report.first_detection_s is not None
            else ""
        ),
        "incident_count": len(report.incidents),
        "event_count": len(report.events),
        "epicenter": incident.epicenter if incident else "",
        "epicenter_score": (
            round(top[0]["score"], 3) if top else ""
        ),
        "top_candidates": "|".join(entry["asset_id"] for entry in top),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run anomaly detection over the generated batch."
    )

    parser.add_argument(
        "--detector",
        choices=sorted(DETECTOR_REGISTRY),
        default="residual",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Score threshold in standard deviations (default: 4).",
    )

    parser.add_argument(
        "--min-assets",
        type=int,
        default=3,
        help="Assets that must deviate at one timestamp (default: 3).",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier on the default sensor noise. Sweep this to "
        "trace detection rate against instrument quality.",
    )

    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Feed the detector noise-free truth. Shows how much "
        "apparent skill comes from the simulator being deterministic.",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Output subdirectory name (default: the detector name).",
    )

    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"Manifest not found: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}\n"
            "Run scripts/run_generated_scenarios.py first."
        )

    if not BASELINE_PATH.exists():
        raise SystemExit(
            f"Baseline not found: {BASELINE_PATH.relative_to(PROJECT_ROOT)}\n"
            "Run scripts/run_fault_simulation.py first."
        )

    baseline = pd.read_csv(BASELINE_PATH)

    detector = _build_detector(args, baseline)

    output_dir = DETECTION_ROOT / (args.name or args.detector)
    events_dir = output_dir / "events"

    events_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)

    print(f"Detector : {detector.name}")
    print(f"Threshold: {args.threshold} sigma, min_assets={args.min_assets}")
    print(f"Sensors  : {detector.sensor_model.describe()}")
    print(f"Scenarios: {len(manifest)} + baseline control\n")

    # The baseline is prepended so the control is visible at the top of
    # the output rather than buried after 35 fault runs.
    jobs = [(BASELINE_SCENARIO, BASELINE_PATH)] + [
        (row["scenario"], _resolve(row["dataset_file"]))
        for _, row in manifest.iterrows()
    ]

    rows = []
    incidents = {}

    for index, (scenario, dataset_path) in enumerate(jobs, start=1):
        if not dataset_path.exists():
            print(f"[{index:03d}] SKIP {scenario} (missing dataset)")
            continue

        dataset = pd.read_csv(dataset_path)

        report = detector.detect(dataset)

        # The baseline's own scenario column already reads "normal"; for
        # the control we want that name regardless of what produced it.
        report.scenario = scenario

        for incident in report.incidents:
            incident.scenario = scenario

        row = _summary_row(report)
        rows.append(row)

        incidents[scenario] = report.to_dict()

        report.events_dataframe().to_csv(
            events_dir / f"{scenario}.csv",
            index=False,
        )

        flag = "DETECTED" if report.detected else "quiet   "

        marker = " <- control" if scenario == BASELINE_SCENARIO else ""

        print(
            f"[{index:03d}/{len(jobs):03d}] {flag} {scenario:32s} "
            f"events={len(report.events):5d} "
            f"epicenter={row['epicenter'] or '-':>6s}{marker}"
        )

    # --------------------------------------------------
    # Write artifacts
    # --------------------------------------------------

    summary_path = output_dir / "summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "incidents.json").open("w", encoding="utf-8") as handle:
        json.dump(incidents, handle, indent=2)

    with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "detector": detector.name,
                "config": detector.describe(),
                "sensors": detector.sensor_model.describe(),
                "noise_scale": args.noise_scale,
                "scenarios": len(rows),
            },
            handle,
            indent=2,
        )

    detected = sum(row["detected"] for row in rows if row["scenario"] != BASELINE_SCENARIO)
    control = next(
        (row for row in rows if row["scenario"] == BASELINE_SCENARIO),
        None,
    )

    print(f"\nFault scenarios detected: {detected}/{len(rows) - 1}")

    if control is not None:
        print(
            "Baseline control        : "
            f"{'FALSE ALARM' if control['detected'] else 'clean'} "
            f"({control['event_count']} point detections)"
        )

    print(f"\nResults -> {output_dir.relative_to(PROJECT_ROOT)}")
    print("Score them with: python scripts/evaluate_detection.py "
          f"--name {output_dir.name}")


if __name__ == "__main__":
    main()
