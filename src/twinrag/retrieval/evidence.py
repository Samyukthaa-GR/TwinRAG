"""
The evidence packet: everything the LLM is allowed to know about one
incident, and nothing else.

Every fact carries the ID of the graph entity it is about, and the packet
lists those IDs in ``allowed_ids``. Phase 5 rejects any diagnosis that
cites an ID outside that list -- that check is what makes the reasoning
"graph-constrained" rather than merely graph-flavoured.

What must never be in a packet: the scenario name (it encodes the fault
type, the target and the window -- ``leak_F3A-KIT_sev100_t06_12``), the
temporal ``state`` labels, or anything from the metadata answer key.
``assert_no_leakage`` enforces that on the serialised packet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


#: Ground-truth vocabulary that has no business in an evidence packet.
LEAKY_TOKENS = ("fault_active", "recovery", "scenario", "severity", "sev100", "injected")


@dataclass
class EvidencePacket:
    incident: dict
    building: dict
    observations: list = field(default_factory=list)
    assets: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    topology_facts: dict = field(default_factory=dict)
    allowed_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def estimated_tokens(self) -> int:
        """
        Rough LLM token count (about four characters per token for
        compact JSON), for budgeting prompts.
        """

        return len(self.to_json(indent=None)) // 4

    def assert_no_leakage(self, forbidden=()) -> None:
        """
        Raise if the packet mentions any ground-truth string.

        Args:
            forbidden: Extra strings to reject -- pass the scenario name
                and anything else from the answer key.
        """

        text = self.to_json(indent=None).lower()

        for token in tuple(LEAKY_TOKENS) + tuple(forbidden):
            if token and str(token).lower() in text:
                raise ValueError(
                    f"Evidence packet leaks ground truth: contains {token!r}."
                )
