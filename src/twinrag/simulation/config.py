"""
Typed configuration for simulation experiments.

An experiment is described by a YAML file (see ``configs/simulation.yaml``):
a network file, shared simulation timing, and a list of fault scenarios. The
loader turns that YAML into the dataclasses below so the rest of the code works
with validated Python objects instead of raw dicts.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SimulationConfig:
    """Timing options shared by every scenario in an experiment."""

    duration_hours: int = 24
    hydraulic_timestep_seconds: int = 3600
    report_timestep_seconds: int = 3600


@dataclass
class FaultScenarioConfig:
    """A single fault to inject: its type, target asset, and severity."""

    type: str
    target_id: str
    severity: float = 0.5
    #: Optional injector-specific overrides (e.g. max_emitter_coefficient).
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """A full experiment: one network, shared timing, and N fault scenarios."""

    network_file: str
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    faults: list = field(default_factory=list)


def load_config(path) -> ExperimentConfig:
    """
    Load and validate an experiment config from a YAML file.

    Raises ``ValueError`` if required keys are missing.
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "network_file" not in raw:
        raise ValueError("Config must define 'network_file'.")

    sim_raw = raw.get("simulation") or {}
    simulation = SimulationConfig(
        duration_hours=sim_raw.get("duration_hours", 24),
        hydraulic_timestep_seconds=sim_raw.get("hydraulic_timestep_seconds", 3600),
        report_timestep_seconds=sim_raw.get("report_timestep_seconds", 3600),
    )

    faults = []
    for entry in raw.get("faults") or []:
        if "type" not in entry or "target_id" not in entry:
            raise ValueError(
                "Each fault entry must define 'type' and 'target_id'."
            )

        faults.append(
            FaultScenarioConfig(
                type=entry["type"],
                target_id=str(entry["target_id"]),
                severity=float(entry.get("severity", 0.5)),
                params=entry.get("params") or {},
            )
        )

    return ExperimentConfig(
        network_file=raw["network_file"],
        simulation=simulation,
        faults=faults,
    )
