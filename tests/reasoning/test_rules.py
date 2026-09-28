"""
Rule-based diagnosis on evidence packets built by the real retriever from
hand-made incidents and events -- no simulation. Each test pins one
physical pattern and the principle the rules use to resolve it.
"""

import pytest

from twinrag.building import BuildingNetworkGenerator
from twinrag.detection import AnomalyEvent, Incident
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.reasoning.rules import RuleBasedDiagnoser, format_message
from twinrag.retrieval import SubgraphRetriever


@pytest.fixture(scope="module")
def retriever():
    network, layout = BuildingNetworkGenerator().build()
    return SubgraphRetriever(BuildingKnowledgeGraph(network, layout))


def _event(hour, asset, parameter, observed, expected):
    sigma = 0.25 if parameter == "pressure" else 2e-6 + 0.02 * abs(expected)
    residual = observed - expected
    return AnomalyEvent(timestamp_s=hour * 3600, asset_id=asset,
                        asset_type="link" if parameter == "flowrate" else "node",
                        parameter=parameter, observed=observed, expected=expected,
                        residual=residual, score=abs(residual) / sigma)


def _diagnose(retriever, events):
    """Build the incident the detector would, retrieve, diagnose."""
    best = {}
    for e in events:
        c = best.get(e.asset_id)
        if c is None or e.score > c["score"]:
            best[e.asset_id] = {"asset_id": e.asset_id, "asset_type": e.asset_type,
                                "parameter": e.parameter, "score": e.score,
                                "residual": e.residual, "peak_at_s": e.timestamp_s,
                                "first_seen_s": min(x.timestamp_s for x in events if x.asset_id == e.asset_id)}
    hours = sorted({e.timestamp_s for e in events})
    incident = Incident(scenario="hidden", detected_at_s=hours[0], last_seen_s=hours[-1],
                        candidates=sorted(best.values(), key=lambda c: -c["score"]), detector="residual")
    packet = retriever.retrieve(incident, events=events).to_dict()
    return RuleBasedDiagnoser().diagnose(packet)


def test_leak_is_named_at_the_tap_of_the_deepest_extra_flow(retriever):
    events = []
    for h in (7, 8, 9):
        events += [
            _event(h, "F3A-MBATH-BR", "flowrate", 2.05e-4, 5e-6),
            _event(h, "F3A-SUPPLY", "flowrate", 2.3e-4, 3e-5),
            _event(h, "OHT-OUTLET", "flowrate", 6e-4, 4e-4),
        ]

    result = _diagnose(retriever, events)
    d = result.diagnosis

    assert result.grounded
    assert (d["root_cause_asset"], d["fault_type"]) == ("F3A-MBATH", "leak")
    assert d["message"]["severity"] == "critical"          # 0.2 L/s is a burst
    assert "master bathroom" in d["message"]["title"]
    assert d["confidence"] >= 0.8


def test_small_leak_is_a_warning_not_a_burst(retriever):
    events = [_event(h, "F2B-UTL-BR", "flowrate", 2.4e-5, 3e-6) for h in (7, 8)]
    events += [_event(h, "F2B-SUPPLY", "flowrate", 5e-5, 2.9e-5) for h in (7, 8)]

    d = _diagnose(retriever, events).diagnosis

    assert d["root_cause_asset"] == "F2B-UTL"
    assert d["message"]["severity"] == "warning"


def test_blocked_riser_is_bracketed_by_the_healthy_flat_above(retriever):
    events = []
    for h in (15, 16):
        for floor, normal in enumerate((22.8, 19.8, 16.8, 13.8, 10.8)):
            events.append(_event(h, f"F{floor}A-IN", "pressure", 0.1, normal))

    d = _diagnose(retriever, events).diagnosis

    assert (d["root_cause_asset"], d["fault_type"]) == ("DTA-5-4", "blockage")
    assert d["message"]["title"] == "No water: Flats 0A, 1A, 2A, 3A, 4A"
    assert d["confidence"] >= 0.8


def test_top_floor_going_dry_is_never_a_minor_loss(retriever):
    # Roof main to stack B closed: the top flat only had ~5 m to lose, but
    # it is dry -- it must stay in the group, so the cause is the roof main.
    events = []
    for h in (8, 9):
        for floor, normal in enumerate((22.9, 19.9, 16.9, 13.9, 10.9, 7.9, 4.9)):
            events.append(_event(h, f"F{floor}B-IN", "pressure", 0.1, normal))

    d = _diagnose(retriever, events).diagnosis

    assert d["root_cause_asset"] == "ROOF-B"


def test_pump_failure_needs_sustained_evidence(retriever):
    base = [_event(11, "PUMP1", "flowrate", 0.0, 1.34e-3)]
    tank = [_event(h, "OHT", "pressure", 0.3, 1.6) for h in (12, 13, 14)]

    sustained = _diagnose(retriever, base + tank).diagnosis
    assert (sustained["root_cause_asset"], sustained["fault_type"]) == ("PUMP1", "pump_failure")
    assert sustained["confidence"] >= 0.5

    # One hour of "pump short, tank low" is also what a normal shift in the
    # pump's cycle looks like: reported, but flagged for review.
    blip = [_event(24, "PUMP1", "flowrate", 0.0, 1.12e-3), _event(24, "OHT", "pressure", 0.4, 1.8)]
    d = _diagnose(retriever, blip).diagnosis
    assert d["confidence"] < 0.5
    assert d["message"]["severity"] == "review"
    assert d["message"]["title"].startswith("Unconfirmed")


def test_every_message_renders_and_every_citation_is_grounded(retriever):
    events = [_event(h, "F1B-IN", "pressure", 0.2, 19.8) for h in (7, 8, 9)]
    result = _diagnose(retriever, events)

    assert result.grounded and not result.violations
    text = format_message(result.diagnosis["message"])
    assert "Flat 1B" in text and "Action:" in text
