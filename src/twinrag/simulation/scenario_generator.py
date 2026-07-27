from itertools import product

from .config import FaultScenarioConfig


def generate_fault_scenarios(
    generation_config,
) -> list[FaultScenarioConfig]:
    """
    Generate deterministic fault scenarios from the
    scenario-generation configuration.

    Each scenario is created from the Cartesian product of:

    target_id × severity × start_hour

    Example:
        3 targets
        3 severities
        3 start hours

        -> 27 scenarios
    """

    scenarios = []

    if not generation_config.enabled:
        return scenarios

    fault_blocks = {
        "leak": generation_config.leak,
        "pump_failure": generation_config.pump_failure,
        "blockage": generation_config.blockage,
    }

    for fault_type, block in fault_blocks.items():

        if block is None:
            continue

        for (
            target_id,
            severity,
            start_hour,
        ) in product(
            block.target_ids,
            block.severities,
            block.start_hours,
        ):

            end_hour = (
                start_hour
                + block.duration_hours
            )

            scenarios.append(
                FaultScenarioConfig(
                    type=fault_type,
                    target_id=target_id,
                    severity=severity,
                    start_hour=start_hour,
                    end_hour=end_hour,
                )
            )

    return scenarios