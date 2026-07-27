from wntr.network.controls import ControlAction, Rule, SimTimeCondition
from wntr.network.base import LinkStatus

from .base import FaultInjector


class BlockageFault(FaultInjector):
    """
    Simulates a timed blockage in a network link.

    For high-severity blockage, the target link is closed
    only during the configured fault interval.

    This gives us an EPANET-compatible timed blockage
    that can be verified against the normal baseline.
    """

    fault_type = "blockage"

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
        Ensure that the blockage target is a valid network link.
        """

        if self.target_id not in network.link_name_list:
            raise ValueError(
                f"Blockage target '{self.target_id}' "
                "is not a valid link in the network."
            )

    def _inject(self, network) -> None:
        """
        Add timed controls that close the link during
        the fault window and reopen it afterward.
        """

        link = network.get_link(
            self.target_id
        )

        start_time_s = (
            self.start_hour * 3600
        )

        if self.end_hour is None:
            raise ValueError(
                "Timed blockage requires end_hour."
            )

        end_time_s = (
            self.end_hour * 3600
        )

        # --------------------------------------------------
        # Close link at fault start
        # --------------------------------------------------

        start_condition = SimTimeCondition(
            network,
            relation=">=",
            threshold=start_time_s,
        )

        close_action = ControlAction(
            link,
            "status",
            LinkStatus.Closed,
        )

        start_rule = Rule(
            start_condition,
            close_action,
            priority=6,
            name=f"blockage_start_{self.target_id}",
        )

        network.add_control(
            f"blockage_start_{self.target_id}",
            start_rule,
        )

        # --------------------------------------------------
        # Reopen link at fault end
        # --------------------------------------------------

        end_condition = SimTimeCondition(
            network,
            relation=">=",
            threshold=end_time_s,
        )

        open_action = ControlAction(
            link,
            "status",
            LinkStatus.Open,
        )

        end_rule = Rule(
            end_condition,
            open_action,
            priority=7,
            name=f"blockage_end_{self.target_id}",
        )

        network.add_control(
            f"blockage_end_{self.target_id}",
            end_rule,
        )

    def describe(self) -> dict:
        """
        Extend common fault metadata with blockage details.
        """

        info = super().describe()

        info.update(
            {
                "mechanism": "timed_link_closure",
                "affected_nodes": [],
                "affected_links": [self.target_id],
            }
        )

        return info