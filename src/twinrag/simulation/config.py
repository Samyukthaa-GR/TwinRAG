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

    #: Optional lower bound on reported pressure, in metres of head.
    #: EPANET reports meaningless heads for any section cut off from
    #: every source -- a building floor below a closed riser comes back
    #: at -9 to -12 m. A real transducer on a drained pipe reads
    #: atmospheric, i.e. 0 m gauge, so the building sets this to 0.0.
    #: ``None`` (the default) leaves results untouched.
    pressure_floor_m: float | None = None


@dataclass
class FaultScenarioConfig:
    """A single fault to inject."""

    type: str
    target_id: str
    severity: float = 0.5
    start_hour: int = 0
    end_hour: int | None = None
    params: dict = field(default_factory=dict)

@dataclass
class FaultGenerationConfig:
    """Systematic scenario-generation settings for one fault type."""

    target_ids: list[str] = field(default_factory=list)
    severities: list[float] = field(default_factory=list)
    start_hours: list[int] = field(default_factory=list)
    duration_hours: int = 6
    params: dict = field(default_factory=dict)


@dataclass
class ScenarioGenerationConfig:
    """Settings for automatically generating fault scenarios."""

    enabled: bool = False
    leak: FaultGenerationConfig | None = None
    pump_failure: FaultGenerationConfig | None = None
    blockage: FaultGenerationConfig | None = None

@dataclass
class ExperimentConfig:
    """A full experiment: one network, shared timing, and N fault scenarios."""

    network_file: str
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    faults: list = field(default_factory=list)
    scenario_generation: ScenarioGenerationConfig = field(
    default_factory=ScenarioGenerationConfig
)


def load_config(path) -> ExperimentConfig:
    """
    Load and validate an experiment configuration from a YAML file.

    The configuration can contain:

    - network_file
    - simulation settings
    - explicitly defined fault scenarios
    - systematic scenario-generation settings

    Returns
    -------
    ExperimentConfig
        Parsed and typed experiment configuration.

    Raises
    ------
    ValueError
        If required configuration fields are missing or invalid.
    """

    path = Path(path)

    # --------------------------------------------------
    # Read YAML
    # --------------------------------------------------

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = yaml.safe_load(handle) or {}

    # --------------------------------------------------
    # Validate required top-level fields
    # --------------------------------------------------

    if "network_file" not in raw:
        raise ValueError(
            "Config must define 'network_file'."
        )

    # --------------------------------------------------
    # Parse shared simulation settings
    # --------------------------------------------------

    sim_raw = raw.get("simulation") or {}

    simulation = SimulationConfig(
        duration_hours=int(
            sim_raw.get(
                "duration_hours",
                24,
            )
        ),
        hydraulic_timestep_seconds=int(
            sim_raw.get(
                "hydraulic_timestep_seconds",
                3600,
            )
        ),
        report_timestep_seconds=int(
            sim_raw.get(
                "report_timestep_seconds",
                3600,
            )
        ),
        pressure_floor_m=(
            float(sim_raw["pressure_floor_m"])
            if sim_raw.get("pressure_floor_m") is not None
            else None
        ),
    )

    # --------------------------------------------------
    # Parse explicitly defined fault scenarios
    # --------------------------------------------------

    faults = []

    for entry in raw.get("faults") or []:

        if (
            "type" not in entry
            or "target_id" not in entry
        ):
            raise ValueError(
                "Each fault entry must define "
                "'type' and 'target_id'."
            )

        severity = float(
            entry.get(
                "severity",
                0.5,
            )
        )

        start_hour = int(
            entry.get(
                "start_hour",
                0,
            )
        )

        end_hour = (
            int(entry["end_hour"])
            if entry.get("end_hour") is not None
            else None
        )

        # Basic validation
        if not 0.0 < severity <= 1.0:
            raise ValueError(
                f"Fault severity must be in "
                f"(0.0, 1.0], got {severity}."
            )

        if start_hour < 0:
            raise ValueError(
                "Fault start_hour must be >= 0."
            )

        if (
            end_hour is not None
            and end_hour <= start_hour
        ):
            raise ValueError(
                "Fault end_hour must be "
                "greater than start_hour."
            )

        faults.append(
            FaultScenarioConfig(
                type=str(
                    entry["type"]
                ),
                target_id=str(
                    entry["target_id"]
                ),
                severity=severity,
                start_hour=start_hour,
                end_hour=end_hour,
                params=(
                    entry.get("params")
                    or {}
                ),
            )
        )

    # --------------------------------------------------
    # Parse automatic scenario-generation settings
    # --------------------------------------------------

    generation_raw = (
        raw.get("scenario_generation")
        or {}
    )

    def _parse_generation_block(
        fault_type: str,
    ):
        """
        Parse one fault-generation block.

        Example:
            scenario_generation:
              leak:
                target_ids: [...]
                severities: [...]
                start_hours: [...]
                duration_hours: 6
                params: {...}   # optional, forwarded to the injector
        """

        block = generation_raw.get(
            fault_type
        )

        if not block:
            return None

        target_ids = [
            str(value)
            for value in block.get(
                "target_ids",
                [],
            )
        ]

        severities = [
            float(value)
            for value in block.get(
                "severities",
                [],
            )
        ]

        start_hours = [
            int(value)
            for value in block.get(
                "start_hours",
                [],
            )
        ]

        duration_hours = int(
            block.get(
                "duration_hours",
                6,
            )
        )

        # ----------------------------------------------
        # Validate generation parameters
        # ----------------------------------------------

        if duration_hours <= 0:
            raise ValueError(
                f"{fault_type} generation "
                "duration_hours must be positive."
            )

        for severity in severities:
            if not 0.0 < severity <= 1.0:
                raise ValueError(
                    f"{fault_type} generation severity "
                    f"must be in (0.0, 1.0], "
                    f"got {severity}."
                )

        for start_hour in start_hours:
            if start_hour < 0:
                raise ValueError(
                    f"{fault_type} generation "
                    "start_hours must be >= 0."
                )

            if (
                start_hour
                + duration_hours
                > simulation.duration_hours
            ):
                raise ValueError(
                    f"{fault_type} scenario beginning "
                    f"at hour {start_hour} with duration "
                    f"{duration_hours} exceeds the "
                    f"{simulation.duration_hours}-hour "
                    "simulation duration."
                )

        return FaultGenerationConfig(
            target_ids=target_ids,
            severities=severities,
            start_hours=start_hours,
            duration_hours=duration_hours,
            params=dict(block.get("params") or {}),
        )

    scenario_generation = (
        ScenarioGenerationConfig(
            enabled=bool(
                generation_raw.get(
                    "enabled",
                    False,
                )
            ),
            leak=(
                _parse_generation_block(
                    "leak"
                )
            ),
            pump_failure=(
                _parse_generation_block(
                    "pump_failure"
                )
            ),
            blockage=(
                _parse_generation_block(
                    "blockage"
                )
            ),
        )
    )

    # --------------------------------------------------
    # Return fully parsed experiment configuration
    # --------------------------------------------------

    return ExperimentConfig(
        network_file=str(
            raw["network_file"]
        ),
        simulation=simulation,
        faults=faults,
        scenario_generation=(
            scenario_generation
        ),
    )