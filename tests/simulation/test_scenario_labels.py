from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from twinrag.simulation.scenario import FaultScenarioRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Net3.inp"


def _make_runner():
    simulation_config = SimpleNamespace(
        duration_hours=24,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )

    return FaultScenarioRunner(
        NETWORK_PATH,
        simulation_config,
    )


def test_baseline_labels_are_normal():
    runner = _make_runner()

    dataset = pd.DataFrame(
        {
            "timestamp_s": [
                0,
                3600,
                7200,
            ],
            "value": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    result = (
        runner._add_baseline_state_labels(
            dataset
        )
    )

    assert set(result["state"]) == {
        "normal"
    }


def test_fault_state_labels():
    runner = _make_runner()

    timestamps = [
        hour * 3600
        for hour in range(25)
    ]

    dataset = pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "value": range(25),
        }
    )

    fault = SimpleNamespace(
        start_hour=8,
        end_hour=18,
    )

    result = (
        runner._add_fault_state_labels(
            dataset,
            fault,
        )
    )

    states = dict(
        zip(
            result["timestamp_s"],
            result["state"],
        )
    )

    assert states[7 * 3600] == "normal"

    assert (
        states[8 * 3600]
        == "fault_active"
    )

    assert (
        states[17 * 3600]
        == "fault_active"
    )

    assert (
        states[18 * 3600]
        == "recovery"
    )

    assert (
        states[24 * 3600]
        == "recovery"
    )