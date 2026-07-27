from pathlib import Path

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
    return NetworkGraphBuilder(network).build()


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE_PATH.exists():
        pytest.skip("baseline.csv not generated; run run_fault_simulation.py")
    return pd.read_csv(BASELINE_PATH)


def test_snapshot_attaches_values(graph, baseline):
    snapshot = GraphSnapshot(graph, baseline, timestamp_s=3600)

    assert snapshot.graph.nodes["101"]["pressure"] is not None
    assert snapshot.graph.edges["10", "101"]["flowrate"] is not None


def test_snapshot_rejects_unknown_timestamp(graph, baseline):
    with pytest.raises(ValueError, match="No rows at timestamp"):
        GraphSnapshot(graph, baseline, timestamp_s=12345)


def test_flow_direction_follows_sign(graph, baseline):
    """Negative flowrate must orient the edge end -> start, not start -> end."""

    snapshot = GraphSnapshot(graph, baseline, timestamp_s=0)

    for _, _, data in snapshot.graph.edges(data=True):
        if data["flowrate"] is None:
            continue

        if data["flowrate"] >= 0:
            assert data["flow_from"] == data["start_node"]
        else:
            assert data["flow_from"] == data["end_node"]


def test_direction_actually_reverses_across_the_day(graph, baseline):
    """
    Link 105 runs forward mid-morning and backward late in the day. This
    is why the static graph is undirected.
    """

    morning = GraphSnapshot(graph, baseline, timestamp_s=3600 * 4)
    evening = GraphSnapshot(graph, baseline, timestamp_s=3600 * 16)

    def orientation(snapshot):
        for _, _, data in snapshot.graph.edges(data=True):
            if data["link_name"] == "105":
                return data["flow_from"]
        raise AssertionError("link 105 not found")

    assert orientation(morning) != orientation(evening)


def test_neighbors_are_ordered_by_distance(graph, baseline):
    snapshot = GraphSnapshot(graph, baseline, timestamp_s=3600)

    neighbours = snapshot.neighbors("101", hops=2)

    assert neighbours, "expected at least one neighbour"
    assert "101" not in [n["asset_id"] for n in neighbours]

    hops = [n["hops"] for n in neighbours]
    assert hops == sorted(hops)
    assert max(hops) <= 2

    one_hop = {n["asset_id"] for n in neighbours if n["hops"] == 1}
    assert one_hop == set(graph.neighbors("101"))


def test_neighbors_rejects_unknown_asset(graph, baseline):
    snapshot = GraphSnapshot(graph, baseline, timestamp_s=3600)

    with pytest.raises(ValueError, match="not a node"):
        snapshot.neighbors("does_not_exist")


def test_upstream_reaches_a_source(graph, baseline):
    """Every junction must trace back to a reservoir or tank."""

    snapshot = GraphSnapshot(graph, baseline, timestamp_s=3600 * 6)

    upstream = snapshot.upstream("101")

    sources = {
        name
        for name in upstream
        if graph.nodes[name]["node_type"] in ("reservoir", "tank")
    }

    assert sources, f"no source upstream of 101, got {upstream[:10]}"


def test_downstream_is_the_inverse_of_upstream(graph, baseline):
    snapshot = GraphSnapshot(graph, baseline, timestamp_s=3600 * 6)

    downstream = snapshot.downstream("101")

    for name in downstream[:5]:
        assert "101" in snapshot.upstream(name)
