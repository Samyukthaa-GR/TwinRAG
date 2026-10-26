"""
Print the generated fault-scenario matrix without simulating anything.

    python scripts/inspect_generated_scenarios.py
    python scripts/inspect_generated_scenarios.py --config configs/building.yaml
    python scripts/inspect_generated_scenarios.py --config configs/building.yaml \
        configs/building_holdout.yaml configs/building_holdout2.yaml \
        configs/building_holdout3.yaml

Several configs print one block each, then a combined total.
"""

import argparse
import sys
from collections import Counter
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
from twinrag.simulation.scenario_generator import (
    generate_fault_scenarios,
)


FAULT_TYPES = ("leak", "pump_failure", "blockage")


def inspect(config_path: Path) -> Counter:
    config = load_config(
        config_path
    )

    generation = config.scenario_generation

    scenarios = generate_fault_scenarios(
        generation
    )

    print("=" * 96)
    print(f"Config : {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Network: {config.network_file}")
    print(
        f"Horizon: {config.simulation.duration_hours} h, "
        f"hydraulic step {config.simulation.hydraulic_timestep_seconds} s, "
        f"report step {config.simulation.report_timestep_seconds} s"
    )
    print("=" * 96)

    # How each fault block expands: targets x severities x start hours.
    print("\nGeneration matrix (targets x severities x start hours):\n")

    for fault_type in FAULT_TYPES:
        block = getattr(generation, fault_type)

        if block is None:
            continue

        n = (
            len(block.target_ids)
            * len(block.severities)
            * len(block.start_hours)
        )

        print(
            f"  {fault_type:<13} "
            f"{len(block.target_ids)} targets x "
            f"{len(block.severities)} severities x "
            f"{len(block.start_hours)} start hours "
            f"= {n:>3} scenarios  "
            f"(duration {block.duration_hours} h)"
        )
        print(f"  {'':<13} targets    : {', '.join(block.target_ids)}")
        print(f"  {'':<13} severities : {block.severities}")
        print(f"  {'':<13} start hours: {block.start_hours}")

    counts = Counter(
        scenario.type
        for scenario in scenarios
    )

    print(f"\nGenerated scenarios: {len(scenarios)}")
    print("Scenario counts by fault type:")

    for fault_type in FAULT_TYPES:
        if counts[fault_type]:
            print(f"  {fault_type:<13} {counts[fault_type]:>3}")

    print("\nGenerated scenario definitions:\n")

    width = max((len(s.target_id) for s in scenarios), default=6)

    for index, scenario in enumerate(
        scenarios,
        start=1,
    ):
        name = create_fault(
            scenario.type,
            scenario.target_id,
            scenario.severity,
            scenario.start_hour,
            scenario.end_hour,
            **scenario.params,
        ).scenario_name()

        print(
            f"{index:03d} | "
            f"type={scenario.type:<13} "
            f"target={scenario.target_id:<{width}} "
            f"severity={scenario.severity:<4} "
            f"start={scenario.start_hour:>2}h "
            f"end={scenario.end_hour:>2}h | "
            f"{name}"
        )

    print()

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "configs" / "simulation.yaml"],
    )
    args = parser.parse_args()

    totals = {}

    for path in args.config:
        totals[path] = inspect(
            (path if path.is_absolute() else Path.cwd() / path).resolve()
        )

    if len(totals) > 1:
        print("=" * 96)
        print("Combined across configs:\n")
        print(f"  {'config':<36} {'leak':>6} {'pump':>6} {'block':>6} {'total':>6}")

        grand = Counter()

        for path, counts in totals.items():
            grand.update(counts)
            print(
                f"  {path.name:<36} "
                f"{counts['leak']:>6} {counts['pump_failure']:>6} "
                f"{counts['blockage']:>6} {sum(counts.values()):>6}"
            )

        print(
            f"  {'TOTAL':<36} "
            f"{grand['leak']:>6} {grand['pump_failure']:>6} "
            f"{grand['blockage']:>6} {sum(grand.values()):>6}"
        )
        print("=" * 96)


if __name__ == "__main__":
    main()
