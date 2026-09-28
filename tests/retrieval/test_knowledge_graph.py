"""
The building knowledge graph: entities, typed relations, and the supply
queries retrieval is built on. Built in memory from the generator -- no
simulation needed.
"""

import pytest

from twinrag.building import BuildingNetworkGenerator
from twinrag.graph.knowledge import FEEDS, LOCATED_IN, MONITORS, PART_OF, BuildingKnowledgeGraph


@pytest.fixture(scope="module")
def kg():
    network, layout = BuildingNetworkGenerator().build()
    return BuildingKnowledgeGraph(network, layout)


def test_entities_and_relations(kg):
    summary = kg.summary()

    assert summary["source"] == "CITY"
    assert summary["entities"] == {
        "building": 1, "floor": 7, "flat": 14, "room": 140, "asset": 261, "sensor": 103,
    }
    # every link is fed once and feeds once
    assert summary["relations"][FEEDS] == 2 * 130
    assert summary["relations"][LOCATED_IN] == 261
    assert summary["relations"][MONITORS] == 103


def test_supply_path_runs_from_tap_to_street(kg):
    path = kg.upstream("F3A-KIT")

    assert path[:7] == ["F3A-KIT", "F3A-KIT-BR", "F3A-TK", "F3A-MAIN-K", "F3A-IN", "F3A-SUPPLY", "DTA-3"]
    assert "OHT" in path and "PUMP1" in path and "SUMP" in path
    assert path[-1] == "CITY"


def test_downstream_of_a_riser_segment_is_the_floors_below(kg):
    below = kg.downstream("DTA-5-4")

    assert {"F4A-KIT", "F0A-KIT", "F0A-IN"} <= below
    assert "F5A-KIT" not in below
    assert "F4B-KIT" not in below


def test_common_supply_point(kg):
    assert kg.common_supply_point(["F0A-IN", "F4A-IN"]) == "DTA-4"
    assert kg.common_supply_point(["F3A-KIT", "F3A-MBATH"]) == "F3A-IN"
    assert kg.common_supply_point(["F3A-KIT-BR"]) == "F3A-KIT-BR"
    # one asset supplying the other is itself the common point
    assert kg.common_supply_point(["F3A-SUPPLY", "F3A-KIT"]) == "F3A-SUPPLY"


def test_triples_carry_place_and_sensor_context(kg):
    triples = set(kg.triples())

    assert ("F3A-TK", FEEDS, "F3A-KIT-BR") in triples
    assert ("F3A-KIT-BR", FEEDS, "F3A-KIT") in triples
    assert ("F3A-KIT", LOCATED_IN, "room:F3A-KIT") in triples
    assert ("room:F3A-KIT", PART_OF, "flat:F3A") in triples
    assert ("flat:F3A", PART_OF, "floor:3") in triples
    assert ("sensor:FM-F3A-KIT-BR", MONITORS, "F3A-KIT-BR") in triples


def test_neighbourhood_counts_pipes_as_hops(kg):
    one = kg.neighbourhood("F3A-TK", 1)

    assert {"F3A-KIT-BR", "F3A-KIT", "F3A-UTL-BR", "F3A-MAIN-K", "F3A-IN"} <= one
    assert "F3A-SUPPLY" not in one


def test_loops_are_rejected():
    network, layout = BuildingNetworkGenerator().build()
    network.add_pipe("LOOP", "F3A-KIT", "F3A-UTL", length=1.0, diameter=0.015, roughness=140)
    layout["links"]["LOOP"] = {"kind": "pipe", "role": "room_branch", "label": "loop"}

    with pytest.raises(ValueError, match="loop"):
        BuildingKnowledgeGraph(network, layout)
