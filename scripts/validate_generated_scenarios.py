import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from twinrag.simulation.config import load_config
from twinrag.simulation.network_loader import WaterNetworkLoader
from twinrag.simulation.scenario_generator import generate_fault_scenarios


def main():
    config_path = PROJECT_ROOT / "configs" / "simulation.yaml"

    config = load_config(config_path)

    network_path = PROJECT_ROOT / config.network_file

    network = WaterNetworkLoader(
        network_path
    ).load()

    scenarios = generate_fault_scenarios(
        config.scenario_generation
    )

    errors = []

    for index, scenario in enumerate(
        scenarios,
        start=1,
    ):
        target_id = scenario.target_id

        if scenario.type == "leak":
            valid = (
                target_id
                in network.junction_name_list
            )

            expected = "junction"

        elif scenario.type == "pump_failure":
            valid = (
                target_id
                in network.pump_name_list
            )

            expected = "pump"

        elif scenario.type == "blockage":
            valid = (
                target_id
                in network.link_name_list
            )

            expected = "link"

        else:
            valid = False
            expected = "known fault target"

        if not valid:
            errors.append(
                {
                    "scenario": index,
                    "type": scenario.type,
                    "target": target_id,
                    "expected": expected,
                }
            )

    print(
        f"\nGenerated scenarios checked: "
        f"{len(scenarios)}"
    )

    if errors:
        print(
            f"Invalid scenarios found: "
            f"{len(errors)}\n"
        )

        for error in errors:
            print(
                f"Scenario {error['scenario']:03d}: "
                f"type={error['type']} "
                f"target={error['target']} "
                f"(expected {error['expected']})"
            )

    else:
        print(
            "All generated scenario targets are valid."
        )


if __name__ == "__main__":
    main()