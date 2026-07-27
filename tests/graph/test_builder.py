from pathlib import Path

from twinrag.graph import NetworkGraphBuilder
from twinrag.simulation.network_loader import WaterNetworkLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Net3.inp"


def _build():
    network = WaterNetworkLoader(NETWORK_PATH).load()
    builder = NetworkGraphBuilder(network)
    return builder, builder.build()


def test_graph_matches_network_composition():
    builder, graph = _build()

    summary = builder.summary(graph)

    assert summary["nodes_total"] == 97
    assert summary["edges_total"] == 119
    assert summary["node_types"] == {
        "junction": 92,
        "tank": 3,
        "reservoir": 2,
    }
    assert summary["link_types"] == {"pipe": 117, "pump": 2}


def test_network_is_one_connected_component():
    """A water network with an isolated island would be a modelling bug."""

    builder, graph = _build()

    assert builder.summary(graph)["components"] == 1


def test_nodes_carry_position_and_elevation():
    _, graph = _build()

    node = graph.nodes["101"]

    assert node["node_type"] == "junction"
    assert node["elevation"] is not None
    assert isinstance(node["x"], float)
    assert isinstance(node["y"], float)


def test_reservoir_elevation_falls_back_to_head():
    """Reservoirs have base_head, not elevation; both map to `elevation`."""

    _, graph = _build()

    assert graph.nodes["Lake"]["node_type"] == "reservoir"
    assert graph.nodes["Lake"]["elevation"] is not None


def test_edges_retain_link_identity_and_endpoints():
    _, graph = _build()

    edge = graph.edges["10", "101"]

    assert edge["link_name"] == "101"
    assert edge["link_type"] == "pipe"
    assert {edge["start_node"], edge["end_node"]} == {"10", "101"}
    assert edge["diameter"] is not None
