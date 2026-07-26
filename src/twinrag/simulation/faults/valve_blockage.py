from .base import FaultInjector


class BlockageFault(FaultInjector):
    """
    Models a valve blockage -- or, when the network has no explicit valves, a
    pipe obstruction -- by adding minor head loss to a link.

    Net3 (and many EPANET example networks) contain no ``[VALVES]``, so this
    fault deliberately targets *any* link: a valve if present, otherwise a pipe.
    The physical effect of a blockage is the same either way -- extra head loss
    that restricts flow -- which makes ``minor_loss`` a valve-type-agnostic
    mechanism.

    * **Partial** severity raises the link's ``minor_loss`` coefficient by
      ``severity * max_minor_loss`` (restricting flow).
    * **Full** severity (``1.0``) closes the link entirely -- a total blockage.

    Both ``minor_loss`` and ``initial_status`` are honoured by the
    ``EpanetSimulator``.
    """

    fault_type = "blockage"

    def __init__(
        self,
        target_id: str,
        severity: float = 0.5,
        max_minor_loss: float = 1000.0,
    ):
        super().__init__(target_id, severity)

        if max_minor_loss <= 0:
            raise ValueError("max_minor_loss must be positive.")

        self.max_minor_loss = max_minor_loss

    def _validate_target(self, network) -> None:
        if self.target_id not in network.link_name_list:
            raise ValueError(
                f"Blockage target '{self.target_id}' is not a link (pipe/valve) "
                f"in the network."
            )

    def _inject(self, network) -> None:
        link = network.get_link(self.target_id)

        if self.severity >= 1.0:
            link.initial_status = "Closed"
            self._mode = "closed"
            self._added_minor_loss = None
        else:
            added = self.severity * self.max_minor_loss
            link.minor_loss = (link.minor_loss or 0.0) + added
            self._mode = "restricted"
            self._added_minor_loss = added

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "mechanism": getattr(self, "_mode", "unknown"),
                "added_minor_loss": getattr(self, "_added_minor_loss", None),
                "affected_nodes": [],
                "affected_links": [self.target_id],
            }
        )
        return info
