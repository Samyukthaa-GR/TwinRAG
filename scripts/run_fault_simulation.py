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
# TwinRAG imports (must come AFTER src is on sys.path)
# --------------------------------------------------

from twinrag.simulation.config import load_config
from twinrag.simulation.faults import create_fault
from twinrag.simulation.scenario import FaultScenarioRunner


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to the project root unless absolute."""

    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _write_outputs(dataset, metadata, filename_stem: str) -> None:
    """Persist a scenario's dataset (CSV) and ground-truth label (JSON)."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED_DIR / f"{filename_stem}.csv"
    meta_path = METADATA_DIR / f"{filename_stem}.json"

    dataset.to_csv(csv_path, index=False)

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"    dataset  -> {csv_path.relative_to(PROJECT_ROOT)}  shape={dataset.shape}")
    print(f"    metadata -> {meta_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline and fault-condition hydraulic simulations."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "simulation.yaml"),
        help="Path to the simulation YAML config.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not (re)generate the normal baseline dataset.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    network_path = _resolve(config.network_file)

    runner = FaultScenarioRunner(network_path, config.simulation)

    print(f"Network: {network_path}")
    print(f"Scenarios: baseline + {len(config.faults)} fault(s)\n")

    if not args.skip_baseline:
        print("[baseline] normal")
        dataset, metadata = runner.run_baseline()
        # Written as 'baseline' to match the existing baseline.csv convention;
        # the dataset's own 'scenario' column value is 'normal'.
        _write_outputs(dataset, metadata, "baseline")

    for fault_cfg in config.faults:
        fault = create_fault(
            fault_type=fault_cfg.type,
            target_id=fault_cfg.target_id,
            severity=fault_cfg.severity,
            **fault_cfg.params,
        )

        print(f"[fault] {fault.scenario_name()}  {fault!r}")
        dataset, metadata = runner.run_fault(fault)
        _write_outputs(dataset, metadata, fault.scenario_name())

    print("\nAll scenarios complete.")


if __name__ == "__main__":
    main()
