from pathlib import Path

import networkx as nx
import pytest

from twinrag.graph.builder import NetworkGraphBuilder
from twinrag.graph.validation import (
    GraphValidationError,
    GraphValidator,
)
from twinrag.simulation.network_loader import WaterNetworkLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = (
    PROJECT_ROOT
    / "data"
    / "networks"
    / "Net3.inp"
)


@pytest.fixture(scope="module")
def valid_graph() -> nx.MultiGraph:
    """
    Build one valid Net3 graph for validator tests.
    """

    network = WaterNetworkLoader(
        NETWORK_PATH
    ).load()

    return NetworkGraphBuilder(
        network,
        source=NETWORK_PATH,
    ).build()


@pytest.fixture()
def graph(valid_graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Return an isolated graph copy for each corruption test.
    """

    return valid_graph.copy()


@pytest.fixture()
def validator() -> GraphValidator:
    """
    Return a graph validator.
    """

    return GraphValidator()


def first_node(
    graph: nx.MultiGraph,
) -> tuple[str, dict]:
    """
    Return the first graph node and its attributes.
    """

    node_id = next(iter(graph.nodes))
    return node_id, graph.nodes[node_id]


def first_edge(
    graph: nx.MultiGraph,
) -> tuple[str, str, str, dict]:
    """
    Return the first keyed graph edge and its attributes.
    """

    return next(
        iter(
            graph.edges(
                keys=True,
                data=True,
            )
        )
    )


def test_valid_graph_passes_validation(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    validator.validate(graph)


def test_validator_rejects_non_graph(
    validator: GraphValidator,
) -> None:
    with pytest.raises(
        TypeError,
        match="NetworkX graph",
    ):
        validator.validate({})


def test_validator_rejects_simple_graph(
    validator: GraphValidator,
) -> None:
    graph = nx.Graph()

    with pytest.raises(
        GraphValidationError,
        match="MultiGraph",
    ):
        validator.validate(graph)


def test_validator_rejects_directed_multigraph(
    validator: GraphValidator,
) -> None:
    graph = nx.MultiDiGraph()

    with pytest.raises(
        GraphValidationError,
        match="undirected",
    ):
        validator.validate(graph)


def test_validator_rejects_missing_graph_metadata(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    del graph.graph["schema_version"]

    with pytest.raises(
        GraphValidationError,
        match="missing required field",
    ):
        validator.validate(graph)


def test_validator_rejects_wrong_graph_type(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    graph.graph["graph_type"] = "invalid_graph"

    with pytest.raises(
        GraphValidationError,
        match="Invalid graph_type",
    ):
        validator.validate(graph)


def test_validator_rejects_wrong_schema_version(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    graph.graph["schema_version"] = "999.0.0"

    with pytest.raises(
        GraphValidationError,
        match="Unsupported graph schema version",
    ):
        validator.validate(graph)


def test_validator_rejects_invalid_runtime_timestamp(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    graph.graph["runtime_timestamp_s"] = "3600"

    with pytest.raises(
        GraphValidationError,
        match="integer or None",
    ):
        validator.validate(graph)


def test_validator_rejects_missing_node_attribute(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    del attributes["runtime"]

    with pytest.raises(
        GraphValidationError,
        match="missing required attribute",
    ):
        validator.validate(graph)


def test_validator_rejects_node_key_asset_id_mismatch(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    attributes["asset_id"] = "different_node"

    with pytest.raises(
        GraphValidationError,
        match="Node keys must match asset IDs",
    ):
        validator.validate(graph)


def test_validator_rejects_invalid_node_asset_type(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    attributes["asset_type"] = "pipe"

    with pytest.raises(
        GraphValidationError,
        match="unsupported asset_type",
    ):
        validator.validate(graph)


def test_validator_rejects_node_identity_flags(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    attributes["is_node_asset"] = False

    with pytest.raises(
        GraphValidationError,
        match="is_node_asset=True",
    ):
        validator.validate(graph)


def test_validator_rejects_invalid_node_display_name(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    attributes["display_name"] = "Invalid Name"

    with pytest.raises(
        GraphValidationError,
        match="display_name",
    ):
        validator.validate(graph)


def test_validator_rejects_non_mapping_node_static(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    attributes["static"] = []

    with pytest.raises(
        GraphValidationError,
        match="'static' must be a mapping",
    ):
        validator.validate(graph)


def test_validator_rejects_non_mapping_node_runtime(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    attributes["runtime"] = []

    with pytest.raises(
        GraphValidationError,
        match="'runtime' must be a mapping",
    ):
        validator.validate(graph)


def test_validator_rejects_non_mapping_node_anomaly(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, attributes = first_node(graph)
    attributes["anomaly"] = []

    with pytest.raises(
        GraphValidationError,
        match="'anomaly' must be a mapping",
    ):
        validator.validate(graph)


def test_validator_rejects_missing_edge_attribute(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    del attributes["start_node"]

    with pytest.raises(
        GraphValidationError,
        match="missing required attribute",
    ):
        validator.validate(graph)


def test_validator_rejects_edge_key_asset_id_mismatch(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    attributes["asset_id"] = "different_link"

    with pytest.raises(
        GraphValidationError,
        match="Edge keys must match link asset IDs",
    ):
        validator.validate(graph)


def test_validator_rejects_invalid_link_asset_type(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    attributes["asset_type"] = "junction"

    with pytest.raises(
        GraphValidationError,
        match="unsupported asset_type",
    ):
        validator.validate(graph)


def test_validator_rejects_edge_identity_flags(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    attributes["is_link_asset"] = False

    with pytest.raises(
        GraphValidationError,
        match="is_link_asset=True",
    ):
        validator.validate(graph)


def test_validator_rejects_invalid_link_display_name(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    attributes["display_name"] = "Invalid Name"

    with pytest.raises(
        GraphValidationError,
        match="display_name",
    ):
        validator.validate(graph)


def test_validator_rejects_missing_declared_start_node(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    attributes["start_node"] = "missing_node"

    with pytest.raises(
        GraphValidationError,
        match="missing start_node",
    ):
        validator.validate(graph)


def test_validator_rejects_endpoint_mismatch(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)

    replacement = next(
        node_id
        for node_id in graph.nodes
        if node_id not in {
            attributes["start_node"],
            attributes["end_node"],
        }
    )

    attributes["end_node"] = replacement

    with pytest.raises(
        GraphValidationError,
        match="do not match the graph edge endpoints",
    ):
        validator.validate(graph)


def test_validator_rejects_non_mapping_edge_runtime(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    attributes["runtime"] = []

    with pytest.raises(
        GraphValidationError,
        match="'runtime' must be a mapping",
    ):
        validator.validate(graph)


def test_validator_rejects_non_mapping_edge_anomaly(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    _, _, _, attributes = first_edge(graph)
    attributes["anomaly"] = []

    with pytest.raises(
        GraphValidationError,
        match="'anomaly' must be a mapping",
    ):
        validator.validate(graph)


def test_validator_rejects_disconnected_graph(
    validator: GraphValidator,
    graph: nx.MultiGraph,
) -> None:
    graph.add_node(
        "isolated",
        asset_id="isolated",
        asset_type="junction",
        display_name="Junction isolated",
        is_node_asset=True,
        is_link_asset=False,
        static={},
        runtime={},
        anomaly={},
    )

    with pytest.raises(
        GraphValidationError,
        match="must be connected",
    ):
        validator.validate(graph)