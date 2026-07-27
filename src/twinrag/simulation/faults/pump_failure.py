from .base import FaultInjector


class PumpFailureFault(FaultInjector):
    """
    Simulates a partial or complete pump failure.

    Complete failure:
        The pump is shut down only during the configured
        start/end time interval.

    Partial failure:
        Currently falls back to a constant speed reduction.
        Timed partial-speed degradation can be added later.
    """

    fault_type = "pump_failure"

    def __init__(
        self,
        target_id: str,
        severity: float = 0.5,
        start_hour: int = 0,
        end_hour: int | None = None,
    ):
        super().__init__(
            target_id=target_id,
            severity=severity,
            start_hour=start_hour,
            end_hour=end_hour,
        )

    def _validate_target(self, network) -> None:
        """
        Ensure that the target exists and is a pump.
        """

        if self.target_id not in network.pump_name_list:
            raise ValueError(
                f"Pump failure target '{self.target_id}' "
                "is not a pump in the network."
            )

    def _inject(self, network) -> None:
        """
        Inject the pump failure.

        For complete failure, WNTR's timed outage mechanism is used.

        For partial failure, pump speed is currently reduced for
        the whole simulation. Timed partial degradation will be
        implemented separately if needed.
        """

        pump = network.get_link(self.target_id)

        # --------------------------------------------------
        # Complete pump failure
        # --------------------------------------------------

        if self.severity >= 1.0:
            start_time_s = self.start_hour * 3600

            end_time_s = (
                self.end_hour * 3600
                if self.end_hour is not None
                else None
            )

            pump.add_outage(
                network,
                start_time=start_time_s,
                end_time=end_time_s,
                priority=6,
                add_after_outage_rule=True,
            )

            return

        # --------------------------------------------------
        # Partial pump degradation
        # --------------------------------------------------

        remaining_speed = 1.0 - self.severity

        pump.base_speed *= remaining_speed

    def describe(self) -> dict:
        """
        Extend the shared fault metadata with
        pump-specific details.
        """

        info = super().describe()

        if self.severity >= 1.0:
            mechanism = "timed_complete_outage"
            remaining_speed = 0.0
        else:
            mechanism = "speed_reduction"
            remaining_speed = 1.0 - self.severity

        info.update(
            {
                "mechanism": mechanism,
                "remaining_speed_fraction": remaining_speed,
                "affected_nodes": [],
                "affected_links": [self.target_id],
            }
        )

        return info