import argparse
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

from twinrag.simulation.config import load_config
from twinrag.simulation.faults import create_fault
from twinrag.simulation.scenario import FaultScenarioRunner


# --------------------------------------------------
# Output directories
# --------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def _resolve(path_str: str) -> Path:
    """
    Resolve a path from the configuration file.

    Relative paths are interpreted from the project root.
    Absolute paths are returned unchanged.
    """

    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _write_outputs(
    dataset,
    metadata: dict,
    filename_stem: str,
) -> None:
    """
    Save a simulation dataset and its ground-truth metadata.

    Dataset:
        data/processed/<scenario>.csv

    Metadata:
        data/metadata/<scenario>.json
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = PROCESSED_DIR / f"{filename_stem}.csv"
    meta_path = METADATA_DIR / f"{filename_stem}.json"

    dataset.to_csv(
        csv_path,
        index=False,
    )

    with meta_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print(
        f"    dataset  -> "
        f"{csv_path.relative_to(PROJECT_ROOT)}  "
        f"shape={dataset.shape}"
    )

    print(
        f"    metadata -> "
        f"{meta_path.relative_to(PROJECT_ROOT)}"
    )


def main() -> None:
    """
    Run the configured baseline and fault-condition simulations.
    """

    # --------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Run baseline and fault-condition "
            "hydraulic simulations."
        )
    )

    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "simulation.yaml"
        ),
        help="Path to the simulation YAML config.",
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Do not regenerate the normal "
            "baseline dataset."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    config = load_config(args.config)

    network_path = _resolve(
        config.network_file
    )

    runner = FaultScenarioRunner(
        network_path,
        config.simulation,
    )

    print(f"Network: {network_path}")
    print(
        f"Scenarios: baseline + "
        f"{len(config.faults)} fault(s)\n"
    )

    # --------------------------------------------------
    # Baseline simulation
    # --------------------------------------------------

    if not args.skip_baseline:
        print("[baseline] normal")

        dataset, metadata = (
            runner.run_baseline()
        )

        # The file is called baseline.csv,
        # while its scenario column remains "normal".
        _write_outputs(
            dataset,
            metadata,
            "baseline",
        )

    # --------------------------------------------------
    # Fault simulations
    # --------------------------------------------------

    for fault_cfg in config.faults:

        fault = create_fault(
            fault_type=fault_cfg.type,
            target_id=fault_cfg.target_id,
            severity=fault_cfg.severity,
            start_hour=fault_cfg.start_hour,
            end_hour=fault_cfg.end_hour,
            **fault_cfg.params,
        )

        print(
            f"[fault] "
            f"{fault.scenario_name()}  "
            f"{fault!r}"
        )

        dataset, metadata = (
            runner.run_fault(fault)
        )

        _write_outputs(
            dataset,
            metadata,
            fault.scenario_name(),
        )

    print("\nAll scenarios complete.")


if __name__ == "__main__":
    main()