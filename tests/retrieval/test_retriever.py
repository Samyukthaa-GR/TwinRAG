"""
Subgraph retrieval: the topology facts it derives, the negative evidence
it keeps, and the guarantee that no ground truth reaches the packet.

Incidents are built by hand, shaped like the detector's output, so each
test pins one physical pattern without running a simulation.
"""

import json

import pytest

from twinrag.building import BuildingNetworkGenerator
from twinrag.detection import Incident
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.retrieval import SubgraphRetriever


@pytest.fixture(scope="module")
def retriever():
    network, layout = BuildingNetworkGenerator().build()
    return SubgraphRetriever(BuildingKnowledgeGraph(network, layout))


def _pressure(asset, residual_m, sigma=0.25):
    return {"asset_id": asset, "asset_type": "node", "parameter": "pressure",
            "residual": residual_m, "score": abs(residual_m) / sigma, "first_seen_s": 8 * 3600}


def _flow(asset, residual_m3s, sigma=2e-6):
    return {"asset_id": asset, "asset_type": "link", "parameter": "flowrate",
            "residual": residual_m3s, "score": abs(residual_m3s) / sigma, "first_seen_s": 8 * 3600}


def _incident(candidates, scenario="leak_F3A-KIT_sev100_t06_12"):
    return Incident(scenario=scenario, detected_at_s=8 * 3600, last_seen_s=11 * 3600,
                    candidates=candidates, detector="residual")


def _observed(packet, status):
    return {o["asset"] for o in packet.observations if o["status"] == status and "asset" in o}


def test_extra_flow_points_at_the_deepest_pipe_and_its_tap(retriever):
    packet = retriever.retrieve(_incident([
        _flow("OHT-OUTLET", 2e-4),
        _flow("F3A-SUPPLY", 2e-4),
        _flow("F3A-KIT-BR", 2e-4),
        _flow("PUMP1", 1e-3),
    ]))

    extra = packet.topology_facts["extra_flow"]
    assert extra["deepest_assets_with_extra_flow"] == ["F3A-KIT-BR"]
    assert extra["supplied_through_them"] == ["F3A-KIT"]

    # the sibling rooms' meters stayed quiet -- that is evidence too
    assert {"F3A-UTL-BR", "F3A-MBATH-BR"} <= _observed(packet, "normal")


def test_closed_riser_is_bracketed_by_the_healthy_flat_above(retriever):
    packet = retriever.retrieve(_incident(
        [_pressure(f"F{f}A-IN", -10.0 - 3 * f) for f in range(5)],
        scenario="blockage_DTA-5-4_sev100_t07_13",
    ))

    loss = packet.topology_facts["pressure_loss"]
    assert loss["common_supply_point"] == "DTA-4"
    assert loss["suspect_span_upward"] == ["DTA-4", "DTA-5-4"]
    assert loss["nearest_upstream_point_still_feeding_a_healthy_sensor"] == "DTA-5"
    assert "F5A-IN" in _observed(packet, "normal")


def test_uniform_pressure_drop_is_one_building_wide_fact(retriever):
    inlets = [f"F{f}{x}-IN" for f in range(7) for x in "AB"]
    packet = retriever.retrieve(_incident(
        [_pressure(a, -1.4) for a in inlets] + [_pressure("OHT", -1.2)],
        scenario="pump_failure_PUMP1_sev100_t06_16",
    ))

    facts = packet.topology_facts
    assert facts["building_wide_pressure_shift"]["sensors_shifted"] == 14
    assert "pressure_loss" not in facts
    # the low roof tank is traced up its own supply to the pump
    assert facts["low_tank_supply_chain"]["OHT"] == ["OHT", "RISING-MAIN", "PUMP-DEL", "PUMP1"]


def test_a_flat_that_stands_out_from_the_shift_stays_local(retriever):
    inlets = [f"F{f}{x}-IN" for f in range(7) for x in "AB"]
    candidates = [_pressure(a, -1.4) for a in inlets if a != "F1B-IN"]
    candidates.append(_pressure("F1B-IN", -18.0))

    packet = retriever.retrieve(_incident(candidates))

    loss = packet.topology_facts["pressure_loss"]
    assert loss["assets_with_pressure_loss"] == ["F1B-IN"]
    assert loss["suspect_span_upward"] == ["F1B-IN", "F1B-SUPPLY"]


def test_every_cited_id_is_allowed(retriever):
    packet = retriever.retrieve(_incident([_flow("F3A-KIT-BR", 2e-4), _pressure("F0A-IN", -20.0)]))
    allowed = set(packet.allowed_ids)

    for subject, _, obj in packet.relations:
        assert subject in allowed and obj in allowed
    for obs in packet.observations:
        assert obs["asset"] in allowed
        assert obs["sensor"] is None or obs["sensor"] in allowed
    for asset in packet.assets:
        assert asset["id"] in allowed


def test_packet_carries_no_ground_truth(retriever):
    scenario = "blockage_DTA-5-4_sev100_t07_13"
    packet = retriever.retrieve(_incident([_pressure("F0A-IN", -20.0)], scenario=scenario))

    packet.assert_no_leakage(forbidden=[scenario])
    assert scenario not in json.dumps(packet.to_dict())

    packet.incident["note"] = scenario
    with pytest.raises(ValueError, match="leaks ground truth"):
        packet.assert_no_leakage(forbidden=[scenario])
