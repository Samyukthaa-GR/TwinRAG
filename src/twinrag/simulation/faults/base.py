from abc import ABC, abstractmethod


class FaultInjector(ABC):
    """
    Base class for all fault injectors.
    """

    fault_type: str = "generic"

    def __init__(
        self,
        target_id: str,
        severity: float = 0.5,
        start_hour: int = 0,
        end_hour: int | None = None,
    ):
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError(
                "target_id must be a non-empty string."
            )

        if not 0.0 < severity <= 1.0:
            raise ValueError(
                f"severity must be in the range "
                f"(0.0, 1.0], got: {severity}"
            )

        if start_hour < 0:
            raise ValueError(
                "start_hour must be >= 0."
            )

        if (
            end_hour is not None
            and end_hour <= start_hour
        ):
            raise ValueError(
                "end_hour must be greater than start_hour."
            )

        self.target_id = target_id
        self.severity = severity
        self.start_hour = start_hour
        self.end_hour = end_hour

    def apply(self, network) -> None:
        """
        Validate the target, then mutate the network.
        """

        self._validate_target(network)
        self._inject(network)

    @abstractmethod
    def _validate_target(self, network) -> None:
        """
        Raise ValueError if target_id is not valid.
        """

    @abstractmethod
    def _inject(self, network) -> None:
        """
        Mutate the network to introduce the fault.
        """

    def scenario_name(self) -> str:
        severity_pct = int(
            round(self.severity * 100)
        )

        return (
            f"{self.fault_type}_"
            f"{self.target_id}_"
            f"sev{severity_pct}"
        )

    def describe(self) -> dict:
        return {
            "fault_type": self.fault_type,
            "target_id": self.target_id,
            "severity": self.severity,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "scenario": self.scenario_name(),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"target_id={self.target_id!r}, "
            f"severity={self.severity}, "
            f"start_hour={self.start_hour}, "
            f"end_hour={self.end_hour})"
        )