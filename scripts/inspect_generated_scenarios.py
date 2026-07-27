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
from twinrag.simulation.scenario_generator import (
    generate_fault_scenarios,
)


def main():
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "simulation.yaml"
    )

    config = load_config(
        config_path
    )

    scenarios = generate_fault_scenarios(
        config.scenario_generation
    )

    print(
        f"\nGenerated scenarios: "
        f"{len(scenarios)}\n"
    )

    counts = Counter(
        scenario.type
        for scenario in scenarios
    )

    print("Scenario counts by fault type:")

    for fault_type, count in counts.items():
        print(
            f"  {fault_type}: {count}"
        )

    print("\nGenerated scenario definitions:\n")

    for index, scenario in enumerate(
        scenarios,
        start=1,
    ):
        print(
            f"{index:03d} | "
            f"type={scenario.type:<13} "
            f"target={scenario.target_id:<6} "
            f"severity={scenario.severity:<4} "
            f"start={scenario.start_hour:>2}h "
            f"end={scenario.end_hour:>2}h"
        )


if __name__ == "__main__":
    main()