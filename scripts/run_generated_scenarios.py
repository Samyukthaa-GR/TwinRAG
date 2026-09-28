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
# --------------------------------------------------

from twinrag.simulation.config import load_config
from twinrag.simulation.faults import create_fault
from twinrag.simulation.scenario import FaultScenarioRunner
from twinrag.simulation.scenario_generator import (
    generate_fault_scenarios,
)


# --------------------------------------------------
# Defaults (the Net3 evaluation batch)
# --------------------------------------------------

DEFAULT_CONFIG = "configs/simulation.yaml"
DEFAULT_OUTPUT_ROOT = "data/generated"


def _resolve(path_str: str) -> Path:
    """
    Resolve a config path relative to the project root.
    """

    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _prepare_directories(
    processed_dir: Path,
    metadata_dir: Path,
) -> None:
    """
    Create generated-data directories if needed.
    """

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def _save_scenario(
    dataset,
    metadata: dict,
    scenario_name: str,
    processed_dir: Path,
    metadata_dir: Path,
):
    """
    Save one generated scenario's CSV and metadata JSON.
    """

    csv_path = (
        processed_dir
        / f"{scenario_name}.csv"
    )

    metadata_path = (
        metadata_dir
        / f"{scenario_name}.json"
    )

    dataset.to_csv(
        csv_path,
        index=False,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    return csv_path, metadata_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and simulate the evaluation scenario batch "
            "defined by a config's scenario_generation block."
        )
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Experiment config (default: {DEFAULT_CONFIG}).",
    )

    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Directory receiving processed/, metadata/ and "
            f"scenarios_manifest.csv (default: {DEFAULT_OUTPUT_ROOT})."
        ),
    )

    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help=(
            "Also simulate the unfaulted network and write "
            "processed/baseline.csv under the output root."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Generate and simulate the complete evaluation scenario set.
    """

    args = _parse_args()

    config_path = _resolve(
        args.config
    )

    output_root = _resolve(
        args.output_root
    )

    processed_dir = output_root / "processed"
    metadata_dir = output_root / "metadata"
    manifest_path = output_root / "scenarios_manifest.csv"

    config = load_config(
        config_path
    )

    generated_configs = (
        generate_fault_scenarios(
            config.scenario_generation
        )
    )

    if not generated_configs:
        print(
            "No generated scenarios found. "
            "Check scenario_generation.enabled."
        )
        return

    network_path = _resolve(
        config.network_file
    )

    runner = FaultScenarioRunner(
        network_path,
        config.simulation,
    )

    _prepare_directories(
        processed_dir,
        metadata_dir,
    )

    print(
        f"Network: {network_path}"
    )

    if args.with_baseline:
        baseline, baseline_metadata = (
            runner.run_baseline()
        )

        csv_path, _ = _save_scenario(
            baseline,
            baseline_metadata,
            "baseline",
            processed_dir,
            metadata_dir,
        )

        print(
            f"Baseline -> "
            f"{csv_path.relative_to(PROJECT_ROOT)} "
            f"(rows={len(baseline)})"
        )

    print(
        f"Generated scenarios: "
        f"{len(generated_configs)}\n"
    )

    manifest_rows = []

    for index, fault_cfg in enumerate(
        generated_configs,
        start=1,
    ):

        fault = create_fault(
            fault_type=fault_cfg.type,
            target_id=fault_cfg.target_id,
            severity=fault_cfg.severity,
            start_hour=fault_cfg.start_hour,
            end_hour=fault_cfg.end_hour,
            **fault_cfg.params,
        )

        scenario_name = (
            fault.scenario_name()
        )

        print(
            f"[{index:03d}/"
            f"{len(generated_configs):03d}] "
            f"{scenario_name}"
        )

        dataset, metadata = (
            runner.run_fault(
                fault
            )
        )

        csv_path, metadata_path = (
            _save_scenario(
                dataset,
                metadata,
                scenario_name,
                processed_dir,
                metadata_dir,
            )
        )

        manifest_rows.append(
            {
                "scenario_id": index,
                "scenario": scenario_name,
                "fault_type": fault.fault_type,
                "target_id": fault.target_id,
                "severity": fault.severity,
                "start_hour": fault.start_hour,
                "end_hour": fault.end_hour,
                "dataset_file": str(
                    csv_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "metadata_file": str(
                    metadata_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "rows": len(dataset),
            }
        )

        print(
            f"    rows={len(dataset)}"
        )

    # --------------------------------------------------
    # Write manifest
    # --------------------------------------------------

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        fieldnames = [
            "scenario_id",
            "scenario",
            "fault_type",
            "target_id",
            "severity",
            "start_hour",
            "end_hour",
            "dataset_file",
            "metadata_file",
            "rows",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    print(
        "\nGenerated scenario batch complete."
    )

    print(
        f"Manifest -> "
        f"{manifest_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()