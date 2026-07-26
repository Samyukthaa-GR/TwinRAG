from abc import ABC, abstractmethod


class FaultInjector(ABC):
    """
    Base class for all fault injectors.

    A fault injector mutates a WNTR ``WaterNetworkModel`` *in place* so that a
    subsequent hydraulic simulation reflects the faulted condition. Every
    concrete fault targets a single asset and carries a ``severity`` in the
    range ``(0.0, 1.0]``, where ``1.0`` is the most severe form of that fault.

    Subclasses implement two hooks:

    * ``_validate_target`` -- raise if ``target_id`` is not a valid target.
    * ``_inject``          -- perform the network mutation.

    ``apply`` runs them in order so callers never mutate an invalid network.
    """

    #: Short machine-readable fault name; overridden by each subclass.
    fault_type: str = "generic"

    def __init__(self, target_id: str, severity: float = 0.5):
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id must be a non-empty string.")

        if not 0.0 < severity <= 1.0:
            raise ValueError(
                f"severity must be in the range (0.0, 1.0], got: {severity}"
            )

        self.target_id = target_id
        self.severity = severity

    def apply(self, network) -> None:
        """
        Validate the target, then mutate the network to introduce the fault.
        """

        self._validate_target(network)
        self._inject(network)

    @abstractmethod
    def _validate_target(self, network) -> None:
        """Raise ``ValueError`` if ``target_id`` is not valid for this fault."""

    @abstractmethod
    def _inject(self, network) -> None:
        """Mutate ``network`` in place to introduce the fault."""

    def scenario_name(self) -> str:
        """
        Stable, filesystem-safe label used to tag the output dataset, e.g.
        ``leak_101_sev60``. Reused as the ``scenario`` column value and as the
        dataset/metadata filename stem.
        """

        severity_pct = int(round(self.severity * 100))
        return f"{self.fault_type}_{self.target_id}_sev{severity_pct}"

    def describe(self) -> dict:
        """
        Ground-truth metadata describing the injected fault. Subclasses extend
        this with the concrete mechanism and the affected node/link IDs so the
        downstream anomaly-detection and diagnosis modules have a label to
        compare their predictions against.
        """

        return {
            "fault_type": self.fault_type,
            "target_id": self.target_id,
            "severity": self.severity,
            "scenario": self.scenario_name(),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"(target_id={self.target_id!r}, severity={self.severity})"
        )
