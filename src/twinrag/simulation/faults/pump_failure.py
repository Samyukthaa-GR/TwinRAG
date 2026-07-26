from .base import FaultInjector


class PumpFailureFault(FaultInjector):
    """
    Models a pump failure / power outage.

    * A **full** failure (``severity == 1.0``) closes the pump so it delivers no
      flow -- an EPANET-native power-outage representation. Existing ``[CONTROLS]``
      that operate the pump (Net3 cycles its pumps on/off by time and tank level)
      are removed first; otherwise they would re-open the pump mid-simulation and
      the outage would have no effect. This mirrors WNTR's guidance to strip
      conflicting pump controls when modelling outages.
    * A **partial** failure reduces the pump's ``base_speed`` to
      ``(1 - severity)`` of nominal, degrading the head/flow it supplies while
      still letting its normal controls cycle it (e.g. severity ``0.4`` leaves
      the pump running at 60% speed when on).

    Both mechanisms are honoured by the ``EpanetSimulator``.
    """

    fault_type = "pump_failure"

    def _validate_target(self, network) -> None:
        if self.target_id not in network.pump_name_list:
            raise ValueError(
                f"Pump-failure target '{self.target_id}' is not a pump in the network."
            )

    def _remove_controls_targeting_pump(self, network) -> int:
        """
        Remove every control whose action targets this pump, so a forced
        outage is not overridden by the network's own on/off logic.
        Returns the number of controls removed.
        """

        to_remove = []
        for control_name in list(network.control_name_list):
            control = network.get_control(control_name)
            for action in control.actions():
                target_obj = action.target()[0]
                if getattr(target_obj, "name", None) == self.target_id:
                    to_remove.append(control_name)
                    break

        for control_name in to_remove:
            network.remove_control(control_name)

        return len(to_remove)

    def _inject(self, network) -> None:
        pump = network.get_link(self.target_id)

        if self.severity >= 1.0:
            self._removed_controls = self._remove_controls_targeting_pump(network)
            pump.initial_status = "Closed"
            self._mode = "closed"
            self._resulting_speed = 0.0
        else:
            pump.base_speed = pump.base_speed * (1.0 - self.severity)
            self._mode = "degraded_speed"
            self._resulting_speed = pump.base_speed
            self._removed_controls = 0

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "mechanism": getattr(self, "_mode", "unknown"),
                "resulting_base_speed": getattr(self, "_resulting_speed", None),
                "removed_controls": getattr(self, "_removed_controls", 0),
                "affected_nodes": [],
                "affected_links": [self.target_id],
            }
        )
        return info
