from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from twinrag.graph import GraphSnapshot, NetworkGraphBuilder
from twinrag.simulation.network_loader import WaterNetworkLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Net3.inp"
BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "baseline.csv"


@pytest.fixture(scope="module")
def graph():
    network = WaterNetworkLoader(NETWORK_PATH).load()

    return NetworkGraphBuilder(
        network,
        source=NETWORK_PATH,
    ).build()


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE_PATH.exists():
        pytest.skip(
            "baseline.csv not generated; run the simulation pipeline."
        )

    return pd.read_csv(BASELINE_PATH)


def test_snapshot_preserves_multigraph(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    assert isinstance(snapshot.graph, nx.MultiGraph)
    assert snapshot.graph.is_multigraph()
    assert not snapshot.graph.is_directed()


def test_snapshot_attaches_node_runtime_values(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    node = snapshot.graph.nodes["101"]

    assert node["runtime"]["timestamp_s"] == 3600
    assert node["runtime"]["pressure"] is not None
    assert node["runtime"]["demand"] is not None

    assert node["pressure"] == node["runtime"]["pressure"]
    assert node["demand"] == node["runtime"]["demand"]


def test_snapshot_attaches_link_runtime_values(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    edge = snapshot.graph.edges["10", "101", "101"]

    assert edge["runtime"]["timestamp_s"] == 3600
    assert edge["runtime"]["flowrate"] is not None
    assert edge["flowrate"] == edge["runtime"]["flowrate"]


def test_snapshot_sets_graph_runtime_metadata(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    assert snapshot.graph.graph["runtime_timestamp_s"] == 3600
    assert snapshot.graph.graph["runtime_scenario"] == (
        snapshot.scenario
    )
    assert snapshot.graph.graph["runtime_state"] == (
        snapshot.state
    )


def test_snapshot_rejects_unknown_timestamp(graph, baseline):
    with pytest.raises(
        ValueError,
        match="No rows at timestamp",
    ):
        GraphSnapshot(
            graph,
            baseline,
            timestamp_s=12345,
        )


def test_snapshot_rejects_non_dataframe(graph):
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        GraphSnapshot(
            graph,
            dataset=[],
            timestamp_s=0,
        )


def test_snapshot_rejects_missing_required_columns(graph):
    frame = pd.DataFrame(
        {
            "timestamp_s": [0],
            "asset_id": ["101"],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        GraphSnapshot(
            graph,
            frame,
            timestamp_s=0,
        )


def test_snapshot_rejects_duplicate_asset_parameter_rows(
    graph,
    baseline,
):
    frame = baseline.loc[
        baseline["timestamp_s"] == 0
    ].copy()

    frame = pd.concat(
        [
            frame,
            frame.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate asset/parameter",
    ):
        GraphSnapshot(
            graph,
            frame,
            timestamp_s=0,
        )


def test_snapshot_rejects_unknown_node_asset(
    graph,
    baseline,
):
    frame = baseline.loc[
        baseline["timestamp_s"] == 0
    ].copy()

    extra = frame.loc[
        frame["asset_type"] == "node"
    ].iloc[[0]].copy()

    extra["asset_id"] = "unknown_node"

    frame = pd.concat(
        [
            frame,
            extra,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="unknown node IDs",
    ):
        GraphSnapshot(
            graph,
            frame,
            timestamp_s=0,
        )


def test_snapshot_rejects_unknown_link_asset(
    graph,
    baseline,
):
    frame = baseline.loc[
        baseline["timestamp_s"] == 0
    ].copy()

    extra = frame.loc[
        frame["asset_type"] == "link"
    ].iloc[[0]].copy()

    extra["asset_id"] = "unknown_link"

    frame = pd.concat(
        [
            frame,
            extra,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="unknown link IDs",
    ):
        GraphSnapshot(
            graph,
            frame,
            timestamp_s=0,
        )


def test_snapshot_rejects_unsupported_asset_type(
    graph,
    baseline,
):
    frame = baseline.loc[
        baseline["timestamp_s"] == 0
    ].copy()

    frame.loc[
        frame.index[0],
        "asset_type",
    ] = "unknown"

    with pytest.raises(
        ValueError,
        match="unsupported asset_type",
    ):
        GraphSnapshot(
            graph,
            frame,
            timestamp_s=0,
        )


def test_node_and_link_with_same_id_are_resolved_separately(
    graph,
    baseline,
):
    """
    EPANET node and link namespaces may reuse the same identifier.

    Junction 101 and Pipe 101 must receive their own measurements
    without being confused with each other.
    """

    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    node = snapshot.graph.nodes["101"]
    link = snapshot.graph.edges["10", "101", "101"]

    assert node["asset_id"] == "101"
    assert link["asset_id"] == "101"

    assert node["runtime"]["pressure"] is not None
    assert node["runtime"]["demand"] is not None
    assert node["runtime"].get("flowrate") is None

    assert link["runtime"]["flowrate"] is not None
    assert link["runtime"].get("pressure") is None
    assert link["runtime"].get("demand") is None


def test_duplicate_validation_respects_asset_namespace(
    graph,
):
    """
    The same identifier is valid once as a node and once as a link.
    """

    frame = pd.DataFrame(
        [
            {
                "timestamp_s": 0,
                "asset_id": "101",
                "asset_type": "node",
                "parameter": "pressure",
                "value": 40.0,
            },
            {
                "timestamp_s": 0,
                "asset_id": "101",
                "asset_type": "link",
                "parameter": "flowrate",
                "value": 1.5,
            },
        ]
    )

    snapshot = GraphSnapshot(
        graph,
        frame,
        timestamp_s=0,
    )

    assert snapshot.graph.nodes["101"]["pressure"] == 40.0

    link_matches = [
        data
        for _, _, key, data in snapshot.graph.edges(
            keys=True,
            data=True,
        )
        if key == "101"
    ]

    assert len(link_matches) == 1
    assert link_matches[0]["flowrate"] == 1.5


def test_flow_direction_follows_sign(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=0,
    )

    for _, _, _, data in snapshot.graph.edges(
        keys=True,
        data=True,
    ):
        flowrate = data["runtime"]["flowrate"]

        if flowrate is None:
            continue

        if flowrate >= 0:
            assert data["runtime"]["flow_from"] == (
                data["start_node"]
            )
            assert data["runtime"]["flow_to"] == (
                data["end_node"]
            )
        else:
            assert data["runtime"]["flow_from"] == (
                data["end_node"]
            )
            assert data["runtime"]["flow_to"] == (
                data["start_node"]
            )


def test_direction_actually_reverses_across_the_day(
    graph,
    baseline,
):
    """
    Link 105 changes direction during normal operation.

    This verifies that runtime flow direction is not assumed from the
    static EPANET start/end orientation.
    """

    morning = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600 * 4,
    )

    evening = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600 * 16,
    )

    def orientation(snapshot):
        for _, _, key, data in snapshot.graph.edges(
            keys=True,
            data=True,
        ):
            if key == "105":
                return data["runtime"]["flow_from"]

        raise AssertionError("link 105 not found")

    assert orientation(morning) != orientation(evening)


def test_directed_snapshot_is_keyed_multidigraph(
    graph,
    baseline,
):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    directed = snapshot.directed()

    assert isinstance(directed, nx.MultiDiGraph)
    assert directed.is_directed()
    assert directed.is_multigraph()

    for _, _, key, data in directed.edges(
        keys=True,
        data=True,
    ):
        assert key == data["asset_id"]
        assert data["flow_from"] is not None
        assert data["flow_to"] is not None


def test_neighbors_are_ordered_by_distance(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    neighbours = snapshot.neighbors(
        "101",
        hops=2,
    )

    assert neighbours

    assert "101" not in {
        item["asset_id"]
        for item in neighbours
    }

    hops = [
        item["hops"]
        for item in neighbours
    ]

    assert hops == sorted(hops)
    assert max(hops) <= 2

    one_hop = {
        item["asset_id"]
        for item in neighbours
        if item["hops"] == 1
    }

    assert one_hop == set(
        graph.neighbors("101")
    )


def test_neighbors_reject_unknown_asset(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    with pytest.raises(
        ValueError,
        match="not a node",
    ):
        snapshot.neighbors(
            "does_not_exist"
        )


def test_neighbors_reject_negative_hops(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600,
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        snapshot.neighbors(
            "101",
            hops=-1,
        )


def test_upstream_reaches_a_source(graph, baseline):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600 * 6,
    )

    upstream = snapshot.upstream("101")

    sources = {
        name
        for name in upstream
        if graph.nodes[name]["asset_type"]
        in {
            "reservoir",
            "tank",
        }
    }

    assert sources, (
        "No source upstream of 101. "
        f"First results: {upstream[:10]}"
    )


def test_bounded_upstream_respects_max_hops(
    graph,
    baseline,
):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600 * 6,
    )

    upstream = snapshot.upstream(
        "101",
        max_hops=2,
    )

    reversed_graph = snapshot.directed().reverse(
        copy=False,
    )

    distances = nx.single_source_shortest_path_length(
        reversed_graph,
        "101",
        cutoff=2,
    )

    assert set(upstream) == (
        set(distances) - {"101"}
    )


def test_downstream_is_inverse_of_upstream(
    graph,
    baseline,
):
    snapshot = GraphSnapshot(
        graph,
        baseline,
        timestamp_s=3600 * 6,
    )

    downstream = snapshot.downstream("101")

    for name in downstream[:5]:
        assert "101" in snapshot.upstream(name)