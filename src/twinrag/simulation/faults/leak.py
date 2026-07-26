from .base import FaultInjector


class LeakFault(FaultInjector):
    """
    Models a leak by attaching an EPANET *emitter* to a junction.

    EPANET emitters model pressure-dependent outflow ``q = C * p^gamma`` at a
    node, which is the EPANET-compatible way to represent a leak. (WNTR's
    ``Junction.add_leak`` API only takes effect under the ``WNTRSimulator``,
    whereas this project simulates with the ``EpanetSimulator`` -- so emitters
    are the correct mechanism here.)

    Severity scales the emitter coefficient ``C`` linearly between ``0`` and
    ``max_emitter_coefficient``. The coefficient's units follow the network's
    configured flow units, so ``max_emitter_coefficient`` should be tuned per
    network; the default is a reasonable starting point for Net3 (US units).
    """

    fault_type = "leak"

    def __init__(
        self,
        target_id: str,
        severity: float = 0.5,
        max_emitter_coefficient: float = 100.0,
    ):
        super().__init__(target_id, severity)

        if max_emitter_coefficient <= 0:
            raise ValueError("max_emitter_coefficient must be positive.")

        self.max_emitter_coefficient = max_emitter_coefficient
        self.emitter_coefficient = severity * max_emitter_coefficient

    def _validate_target(self, network) -> None:
        if self.target_id not in network.junction_name_list:
            raise ValueError(
                f"Leak target '{self.target_id}' is not a junction in the network."
            )

    def _inject(self, network) -> None:
        junction = network.get_node(self.target_id)
        junction.emitter_coefficient = self.emitter_coefficient

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "mechanism": "emitter_coefficient",
                "emitter_coefficient": self.emitter_coefficient,
                "affected_nodes": [self.target_id],
                "affected_links": [],
            }
        )
        return info
