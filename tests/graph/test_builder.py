from pathlib import Path

import networkx as nx
import pytest

from twinrag.graph import NetworkGraphBuilder
from twinrag.graph.constants import GRAPH_SCHEMA_VERSION
from twinrag.simulation.network_loader import WaterNetworkLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Net3.inp"


@pytest.fixture(scope="module")
def network():
    return WaterNetworkLoader(NETWORK_PATH).load()


@pytest.fixture(scope="module")
def builder(network):
    return NetworkGraphBuilder(
        network,
        source=NETWORK_PATH,
    )


@pytest.fixture(scope="module")
def graph(builder):
    return builder.build()


def test_builder_returns_undirected_multigraph(graph):
    assert isinstance(graph, nx.MultiGraph)
    assert graph.is_multigraph()
    assert not graph.is_directed()


def test_graph_matches_network_composition(builder, graph):
    summary = builder.summary(graph)

    assert summary["nodes_total"] == 97
    assert summary["edges_total"] == 119

    assert summary["node_types"] == {
        "junction": 92,
        "tank": 3,
        "reservoir": 2,
    }

    assert summary["link_types"] == {
        "pipe": 117,
        "pump": 2,
    }

    assert summary["multigraph"] is True
    assert summary["directed"] is False


def test_network_is_one_connected_component(builder, graph):
    """A disconnected island would indicate a model or build error."""

    summary = builder.summary(graph)

    assert summary["connected"] is True
    assert summary["components"] == 1


def test_graph_carries_schema_metadata(graph):
    assert graph.graph["name"] == "Net3"
    assert graph.graph["schema_version"] == GRAPH_SCHEMA_VERSION
    assert graph.graph["graph_type"] == (
        "water_distribution_topology"
    )
    assert graph.graph["multigraph"] is True
    assert graph.graph["directed"] is False
    assert graph.graph["runtime_timestamp_s"] is None


def test_junction_has_standard_attribute_containers(graph):
    node = graph.nodes["101"]

    assert node["asset_id"] == "101"
    assert node["asset_type"] == "junction"
    assert node["display_name"] == "Junction 101"
    assert node["is_node_asset"] is True
    assert node["is_link_asset"] is False

    assert isinstance(node["static"], dict)
    assert node["runtime"] == {}
    assert node["anomaly"] == {}

    assert node["static"]["elevation"] is not None
    assert node["static"]["base_demand"] is not None

    coordinates = node["static"]["coordinates"]

    assert isinstance(coordinates["x"], float)
    assert isinstance(coordinates["y"], float)


def test_node_compatibility_attributes_are_preserved(graph):
    node = graph.nodes["101"]

    assert node["node_type"] == "junction"
    assert node["elevation"] is not None
    assert node["base_demand"] is not None
    assert isinstance(node["x"], float)
    assert isinstance(node["y"], float)


def test_reservoir_preserves_base_head_semantics(graph):
    reservoir = graph.nodes["Lake"]

    assert reservoir["asset_type"] == "reservoir"
    assert reservoir["static"]["base_head"] is not None
    assert "elevation" not in reservoir["static"]

    # Compatibility field retained until snapshot/viewer refactoring.
    assert reservoir["elevation"] == reservoir["static"]["base_head"]


def test_tank_contains_operational_limits(graph):
    tanks = [
        data
        for _, data in graph.nodes(data=True)
        if data["asset_type"] == "tank"
    ]

    assert tanks

    for tank in tanks:
        static = tank["static"]

        assert static["elevation"] is not None
        assert static["initial_level"] is not None
        assert static["minimum_level"] is not None
        assert static["maximum_level"] is not None
        assert static["diameter"] is not None


def test_link_is_addressable_by_epanet_identifier(graph):
    edge = graph.edges["10", "101", "101"]

    assert edge["asset_id"] == "101"
    assert edge["asset_type"] == "pipe"
    assert edge["link_name"] == "101"
    assert edge["link_type"] == "pipe"

    assert edge["start_node"] == "10"
    assert edge["end_node"] == "101"

    assert edge["is_node_asset"] is False
    assert edge["is_link_asset"] is True

    assert edge["runtime"] == {}
    assert edge["anomaly"] == {}


def test_pipe_contains_static_engineering_metadata(graph):
    edge = graph.edges["10", "101", "101"]
    static = edge["static"]

    assert static["diameter"] is not None
    assert static["length"] is not None
    assert static["roughness"] is not None
    assert static["minor_loss"] is not None
    assert isinstance(static["check_valve"], bool)


def test_every_graph_edge_key_matches_its_asset_id(graph):
    for _, _, key, data in graph.edges(
        keys=True,
        data=True,
    ):
        assert key == data["asset_id"]
        assert key == data["link_name"]


def test_all_link_endpoints_exist_in_graph(graph):
    for start, end, _, data in graph.edges(
        keys=True,
        data=True,
    ):
        assert start in graph
        assert end in graph

        assert {
            data["start_node"],
            data["end_node"],
        } == {start, end}


def test_builder_rejects_invalid_network_object():
    with pytest.raises(
        TypeError,
        match="required WNTR interface",
    ):
        NetworkGraphBuilder(object())