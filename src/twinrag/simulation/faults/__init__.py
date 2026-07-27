"""
Fault-injection package.

Each injector mutates a WNTR ``WaterNetworkModel`` in place to produce a
labelled fault condition for the digital twin. Use :func:`create_fault` to build
one from a fault-type name (as read from the YAML config), or import a concrete
class directly.
"""

from .base import FaultInjector
from .leak import LeakFault
from .pump_failure import PumpFailureFault
from .valve_blockage import BlockageFault


#: Maps fault-type names (used in configs) to their injector classes.
FAULT_REGISTRY = {
    LeakFault.fault_type: LeakFault,
    PumpFailureFault.fault_type: PumpFailureFault,
    BlockageFault.fault_type: BlockageFault,
}


def create_fault(
    fault_type: str,
    target_id: str,
    severity: float = 0.5,
    start_hour: int = 0,
    end_hour: int | None = None,
    **params,
) -> FaultInjector:
    """
    Build a fault injector from a fault-type name.

    Extra keyword arguments are forwarded to the injector, e.g.
    ``max_emitter_coefficient`` for ``leak`` or ``max_minor_loss`` for
    ``blockage``.

    Raises ``ValueError`` if ``fault_type`` is not registered.
    """

    try:
        fault_cls = FAULT_REGISTRY[fault_type]
    except KeyError:
        valid = ", ".join(sorted(FAULT_REGISTRY))
        raise ValueError(
            f"Unknown fault type '{fault_type}'. Valid types: {valid}."
        )

    return fault_cls(
    target_id=target_id,
    severity=severity,
    start_hour=start_hour,
    end_hour=end_hour,
    **params,
)

__all__ = [
    "FaultInjector",
    "LeakFault",
    "PumpFailureFault",
    "BlockageFault",
    "FAULT_REGISTRY",
    "create_fault",
]
