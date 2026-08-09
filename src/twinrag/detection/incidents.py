"""
The detection contract (dataclass pipeline).

Not to be confused with ``events.py``, which aggregates point anomalies
into event *rows* for the DataFrame pipeline. Both live here on purpose;
see the package docstring for which to use.

Two levels, because Phases 4 and 5 need different things:

``AnomalyEvent``
    One reading that does not match expectation — a single
    (timestamp, asset, parameter) cell. Fine-grained evidence.

``Incident``
    A contiguous stretch of time during which the network as a whole
    looks wrong, with its member events rolled up into a *ranked list of
    candidate assets*. This is what Phase 4 breadth-first-searches from.

The candidate list is ranked rather than reduced to a single epicenter on
purpose. A fault propagates network-wide within one hydraulic timestep,
and the largest deviation does not reliably sit on the faulted asset --
for ``blockage_103`` the worst-hit node is a different node entirely.
Handing Phase 4 a ranked shortlist lets topology arbitrate; handing it
one argmax would bake in an error the graph could have caught.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class AnomalyEvent:
    """
    One reading that deviates from expectation.
    """

    timestamp_s: int
    asset_id: str
    asset_type: str
    parameter: str
    observed: float
    expected: float
    residual: float
    score: float
    scenario: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Incident:
    """
    A contiguous window in which the network deviates from expectation.
    """

    scenario: str
    detected_at_s: int
    last_seen_s: int
    candidates: list = field(default_factory=list)
    event_count: int = 0
    parameters: list = field(default_factory=list)
    detector: str = ""

    @property
    def epicenter(self):
        """
        Highest-ranked candidate asset, or ``None`` if there are none.

        Convenience for reporting only. Phase 4 should walk
        ``candidates``, not trust this.
        """

        return self.candidates[0]["asset_id"] if self.candidates else None

    def top_candidates(self, limit: int = 5) -> list:
        return self.candidates[:limit]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["epicenter"] = self.epicenter
        return payload


@dataclass
class AnomalyReport:
    """
    Everything one detector found in one scenario.
    """

    scenario: str
    detector: str
    events: list = field(default_factory=list)
    incidents: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return bool(self.incidents)

    @property
    def first_detection_s(self):
        """
        When the first incident opened, or ``None`` if nothing fired.
        """

        if not self.incidents:
            return None

        return min(incident.detected_at_s for incident in self.incidents)

    def events_dataframe(self):
        """
        Events as a dataframe, ordered by time then descending score.

        Imported lazily so the dataclasses stay usable in contexts that
        have no need for pandas.
        """

        import pandas as pd

        columns = [
            "scenario",
            "timestamp_s",
            "asset_id",
            "asset_type",
            "parameter",
            "observed",
            "expected",
            "residual",
            "score",
        ]

        if not self.events:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame([event.to_dict() for event in self.events])

        return frame.sort_values(
            ["timestamp_s", "score"],
            ascending=[True, False],
        ).reset_index(drop=True)[columns]

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "detector": self.detector,
            "config": self.config,
            "detected": self.detected,
            "first_detection_s": self.first_detection_s,
            "event_count": len(self.events),
            "incidents": [incident.to_dict() for incident in self.incidents],
        }
