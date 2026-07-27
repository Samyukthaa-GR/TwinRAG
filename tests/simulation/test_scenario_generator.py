from pathlib import Path

from twinrag.simulation.config import load_config
from twinrag.simulation.scenario_generator import (
    generate_fault_scenarios,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "simulation.yaml"


def test_generated_scenario_count():
    config = load_config(
        CONFIG_PATH
    )

    scenarios = generate_fault_scenarios(
        config.scenario_generation
    )

    assert len(scenarios) == 35


def test_generated_fault_counts():
    config = load_config(
        CONFIG_PATH
    )

    scenarios = generate_fault_scenarios(
        config.scenario_generation
    )

    leak_count = sum(
        scenario.type == "leak"
        for scenario in scenarios
    )

    pump_count = sum(
        scenario.type == "pump_failure"
        for scenario in scenarios
    )

    blockage_count = sum(
        scenario.type == "blockage"
        for scenario in scenarios
    )

    assert leak_count == 27
    assert pump_count == 2
    assert blockage_count == 6


def test_generated_scenario_windows_are_valid():
    config = load_config(
        CONFIG_PATH
    )

    scenarios = generate_fault_scenarios(
        config.scenario_generation
    )

    duration = (
        config.simulation.duration_hours
    )

    for scenario in scenarios:
        assert scenario.start_hour >= 0
        assert scenario.end_hour is not None
        assert scenario.end_hour > scenario.start_hour
        assert scenario.end_hour <= duration


def test_generated_scenario_names_are_unique():
    from twinrag.simulation.faults import create_fault

    config = load_config(
        CONFIG_PATH
    )

    scenarios = generate_fault_scenarios(
        config.scenario_generation
    )

    names = []

    for scenario in scenarios:
        fault = create_fault(
            fault_type=scenario.type,
            target_id=scenario.target_id,
            severity=scenario.severity,
            start_hour=scenario.start_hour,
            end_hour=scenario.end_hour,
            **scenario.params,
        )

        names.append(
            fault.scenario_name()
        )

    assert len(names) == len(set(names))