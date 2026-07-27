from .base import FaultInjector


class LeakFault(FaultInjector):
    """
    Simulates a timed leak at a junction using an
    EPANET-compatible additional demand pattern.

    The junction's original demand remains unchanged.
    A second demand is added only during the configured
    leak interval.

    Severity scales the leak demand relative to a
    configurable maximum leak demand.
    """

    fault_type = "leak"

    def __init__(
        self,
        target_id: str,
        severity: float = 0.5,
        start_hour: int = 0,
        end_hour: int | None = None,
        max_leak_demand: float = 0.05,
    ):
        super().__init__(
            target_id=target_id,
            severity=severity,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        if max_leak_demand <= 0:
            raise ValueError(
                "max_leak_demand must be positive."
            )

        self.max_leak_demand = max_leak_demand

        self.leak_demand = (
            self.severity
            * self.max_leak_demand
        )

    def _validate_target(self, network) -> None:
        """
        Ensure that the leak target exists and is a junction.
        """

        if self.target_id not in network.junction_name_list:
            raise ValueError(
                f"Leak target '{self.target_id}' "
                "is not a junction in the network."
            )

    def _inject(self, network) -> None:
        """
        Add a timed leak-demand pattern to the target junction.
        """

        junction = network.get_node(
            self.target_id
        )

        pattern_timestep = (
            network.options.time.pattern_timestep
        )

        duration_seconds = (
            network.options.time.duration
        )

        if pattern_timestep <= 0:
            raise ValueError(
                "Pattern timestep must be positive."
            )

        number_of_steps = int(
            duration_seconds // pattern_timestep
        ) + 1

        start_time_s = (
            self.start_hour * 3600
        )

        end_time_s = (
            self.end_hour * 3600
            if self.end_hour is not None
            else duration_seconds + pattern_timestep
        )

        pattern_values = []

        for step in range(number_of_steps):
            current_time_s = (
                step * pattern_timestep
            )

            if (
                start_time_s
                <= current_time_s
                < end_time_s
            ):
                pattern_values.append(1.0)
            else:
                pattern_values.append(0.0)

        pattern_name = (
            f"leak_{self.target_id}_pattern"
        )

        network.add_pattern(
            pattern_name,
            pattern_values,
        )

        junction.add_demand(
            base=self.leak_demand,
            pattern_name=pattern_name,
            category="Leak",
        )

    def describe(self) -> dict:
        """
        Extend common fault metadata with
        leak-specific information.
        """

        info = super().describe()

        info.update(
            {
                "mechanism": "timed_additional_demand",
                "leak_demand": self.leak_demand,
                "max_leak_demand": self.max_leak_demand,
                "affected_nodes": [self.target_id],
                "affected_links": [],
            }
        )

        return info