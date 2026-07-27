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
# Generated dataset directories
# --------------------------------------------------

GENERATED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "generated"
)

PROCESSED_DIR = (
    GENERATED_ROOT
    / "processed"
)

METADATA_DIR = (
    GENERATED_ROOT
    / "metadata"
)

MANIFEST_PATH = (
    GENERATED_ROOT
    / "scenarios_manifest.csv"
)


def _resolve(path_str: str) -> Path:
    """
    Resolve a config path relative to the project root.
    """

    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _prepare_directories() -> None:
    """
    Create generated-data directories if needed.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _save_scenario(
    dataset,
    metadata: dict,
    scenario_name: str,
):
    """
    Save one generated scenario's CSV and metadata JSON.
    """

    csv_path = (
        PROCESSED_DIR
        / f"{scenario_name}.csv"
    )

    metadata_path = (
        METADATA_DIR
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


def main() -> None:
    """
    Generate and simulate the complete evaluation scenario set.
    """

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "simulation.yaml"
    )

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

    _prepare_directories()

    print(
        f"Network: {network_path}"
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

    with MANIFEST_PATH.open(
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
        f"{MANIFEST_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()