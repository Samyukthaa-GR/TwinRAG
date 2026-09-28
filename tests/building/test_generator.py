"""
The generated building network: structure, layout and physics.

The structural tests build the model in memory and need no simulation.
The hydraulic tests run EPANET on the generated model (well under a
second each) because the claims worth pinning -- pressure falls a storey
at a time, the level switches cycle the pump, a leak shows up on its own
room's branch and nowhere else -- are properties of the simulated water,
not of the file.
"""

import json

import networkx as nx
import pytest

from twinrag.building import BuildingNetworkGenerator, BuildingSpec, load_layout
from twinrag.graph import NetworkGraphBuilder
from twinrag.simulation.config import SimulationConfig
from twinrag.simulation.faults import create_fault
from twinrag.simulation.network_loader import WaterNetworkLoader
from twinrag.simulation.scenario import FaultScenarioRunner


@pytest.fixture(scope="module")
def built():
    return BuildingNetworkGenerator().build()


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    folder = tmp_path_factory.mktemp("building")
    inp = folder / "building.inp"
    layout = folder / "building.layout.json"
    BuildingNetworkGenerator().write(inp, layout)
    return inp, layout


@pytest.fixture(scope="module")
def runner(written, tmp_path_factory, monkeypatch_module):
    # EPANET writes its scratch files into the working directory.
    monkeypatch_module.chdir(tmp_path_factory.mktemp("epanet"))
    return FaultScenarioRunner(
        written[0],
        SimulationConfig(24, 900, 3600, pressure_floor_m=0.0),
    )


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def baseline(runner):
    dataset, _ = runner.run_baseline()
    return dataset


def _wide(dataset, parameter):
    frame = dataset[dataset["parameter"] == parameter]
    return frame.pivot(index="timestamp_s", columns="asset_id", values="value")


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


def test_default_building_composition(built):
    wn, layout = built

    assert len(layout["floors"]) == 7
    assert len(layout["flats"]) == 14
    assert sum(room["wet"] for room in layout["rooms"]) == 70
    assert wn.num_junctions == 128
    assert wn.num_tanks == 2
    assert wn.num_reservoirs == 1
    assert wn.num_pipes == 129
    assert wn.num_pumps == 1


def test_node_and_link_ids_never_collide(built):
    wn, _ = built

    assert not set(wn.node_name_list) & set(wn.link_name_list)


def test_layout_covers_every_asset(built):
    wn, layout = built

    assert set(layout["nodes"]) == set(wn.node_name_list)
    assert set(layout["links"]) == set(wn.link_name_list)


def test_every_fixture_is_located_in_a_room(built):
    _, layout = built

    rooms = {room["id"]: room for room in layout["rooms"]}

    for node_id, node in layout["nodes"].items():
        if node["role"] != "fixture":
            continue

        room = rooms[node["room"]]
        x0, y0, x1, y1 = room["rect"]

        assert room["wet"] and room["fixture_node"] == node_id
        assert x0 <= node["x"] <= x1 and y0 <= node["y"] <= y1
        assert room["floor"] == node["floor"]


def test_pipe_paths_join_their_endpoints(built):
    _, layout = built

    nodes = layout["nodes"]

    for link_id, link in layout["links"].items():
        if link["kind"] != "pipe" or link["role"] in (
            "municipal_inlet",
            "rising_main",
        ):
            continue

        start = nodes[link["start"]]
        end = nodes[link["end"]]

        assert link["path"][0] == [start["x"], start["y"], start["z"]], link_id
        assert link["path"][-1] == [end["x"], end["y"], end["z"]], link_id


def test_pipe_routes_are_orthogonal(built):
    _, layout = built

    for link_id, link in layout["links"].items():
        path = link["path"]

        for a, b in zip(path, path[1:]):
            changed = sum(1 for i in range(3) if a[i] != b[i])
            assert changed == 1 or link["kind"] == "pump", link_id


def test_topology_is_one_tree_fed_from_the_roof_tank(built):
    wn, _ = built

    graph = NetworkGraphBuilder(wn).build()

    assert nx.is_connected(graph)
    # A building supply is a tree: every link is a cut edge, which is
    # exactly why a closed riser starves the floors below it.
    assert graph.number_of_edges() == graph.number_of_nodes() - 1


def test_flat_b_mirrors_flat_a(built):
    _, layout = built

    width = layout["building"]["footprint"][2]
    a = layout["nodes"]["F2A-KIT"]
    b = layout["nodes"]["F2B-KIT"]

    assert a["x"] + b["x"] == pytest.approx(width)
    assert (a["y"], a["z"]) == (b["y"], b["z"])


def test_every_wet_room_branch_is_metered(built):
    _, layout = built

    metered = {
        sensor["asset_id"]
        for sensor in layout["sensors"]
        if sensor["parameter"] == "flowrate"
    }

    branches = {
        link_id
        for link_id, link in layout["links"].items()
        if link["role"] == "room_branch"
    }

    assert len(branches) == 70
    assert branches <= metered


def test_demand_shares_must_sum_to_one():
    rooms = BuildingSpec().rooms
    broken = tuple(
        room if room.code != "KIT" else type(room)(**{
            **room.__dict__, "demand_share": 0.5,
        })
        for room in rooms
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        BuildingNetworkGenerator(BuildingSpec(rooms=broken))


def test_round_trip_through_inp(written):
    inp, layout_path = written

    wn = WaterNetworkLoader(inp).load()
    layout = load_layout(layout_path)

    assert set(layout["nodes"]) == set(wn.node_name_list)
    assert {"PUMP1-START", "PUMP1-STOP"} <= set(wn.control_name_list)
    assert wn.options.hydraulic.demand_model == "PDA"


# ----------------------------------------------------------------------
# Hydraulics
# ----------------------------------------------------------------------


def test_pressure_falls_one_storey_per_floor(baseline, built):
    _, layout = built

    pressure = _wide(baseline, "pressure")
    kitchens = [f"F{floor}A-KIT" for floor in range(7)]
    drops = pressure[kitchens].diff(axis=1).iloc[:, 1:]

    # 3 m slab-to-slab; friction adds almost nothing at domestic flows.
    assert drops.to_numpy() == pytest.approx(-3.0, abs=0.05)


def test_top_floor_keeps_serviceable_pressure(baseline, built):
    _, layout = built

    pressure = _wide(baseline, "pressure")
    # Taps, not the ceiling-level distribution pipes 1.8 m above them.
    top = [
        node_id
        for node_id, node in layout["nodes"].items()
        if node["role"] == "fixture" and node["floor"] == 6
    ]

    assert pressure[top].min().min() > 5.0


def test_level_switches_cycle_the_pump(baseline):
    flow = _wide(baseline, "flowrate")["PUMP1"]
    level = _wide(baseline, "pressure")["OHT"]

    assert (flow > 0).any() and (flow == 0).any()
    assert level.min() > 0.5 and level.max() <= 2.0


def test_leak_appears_on_its_own_room_branch_only(runner, baseline):
    fault = create_fault(
        "leak", "F3A-KIT", 1.0, 6, 12, max_leak_demand=0.0002,
    )
    dataset, _ = runner.run_fault(fault)

    delta = (_wide(dataset, "flowrate") - _wide(baseline, "flowrate"))
    hour_8 = delta.loc[8 * 3600]
    branches = [c for c in delta if c.endswith("-BR")]

    assert hour_8["F3A-KIT-BR"] == pytest.approx(0.0002, rel=0.01)
    assert hour_8["F3A-SUPPLY"] == pytest.approx(0.0002, rel=0.01)
    others = hour_8[[c for c in branches if c != "F3A-KIT-BR"]]
    assert others.abs().max() < 1e-6


def test_closed_riser_starves_the_floors_below(runner):
    fault = create_fault("blockage", "DTA-5-4", 1.0, 7, 13)
    dataset, _ = runner.run_fault(fault)

    pressure = _wide(dataset, "pressure").loc[10 * 3600]
    demand = _wide(dataset, "demand").loc[10 * 3600]

    below = [f"F{floor}A-KIT" for floor in range(5)]

    assert pressure[below].min() >= 0.0            # floored, not negative
    assert pressure[below].max() < 0.05            # drained
    assert demand[below].abs().max() < 1e-6
    assert pressure["F5A-KIT"] > 5.0               # above the closure
    assert pressure["F2B-KIT"] > 5.0               # the other stack


def test_pump_outage_drains_the_roof_tank(runner, baseline):
    fault = create_fault(
        "pump_failure", "PUMP1", 1.0, 6, 16, restore_after_outage=False,
    )
    dataset, _ = runner.run_fault(fault)

    flow = _wide(dataset, "flowrate")["PUMP1"]
    level = _wide(dataset, "pressure")["OHT"]
    base_level = _wide(baseline, "pressure")["OHT"]

    window = flow.loc[6 * 3600 : 15 * 3600]
    assert (window == 0).all()
    assert level.loc[15 * 3600] < base_level.loc[15 * 3600] - 0.5
    assert (flow.loc[16 * 3600 :] > 0).any()       # level switch restarts it


def test_pump_outage_leaves_level_switches_in_charge(written):
    # WNTR's default after-outage rule would hold the pump open for the
    # rest of the run, outranking the roof-tank level switches.
    wn = WaterNetworkLoader(written[0]).load()
    before = set(wn.control_name_list)

    create_fault(
        "pump_failure", "PUMP1", 1.0, 6, 16, restore_after_outage=False,
    ).apply(wn)

    assert len(set(wn.control_name_list) - before) == 1


def test_live_replay_never_cites_the_future(runner, baseline, written):
    # What the dashboard shows at hour h must come only from readings up to
    # h: every time a live diagnosis mentions must be <= h.
    import re

    from twinrag.building import load_layout
    from twinrag.building.pipeline import BuildingDiagnosisPipeline
    from twinrag.graph.knowledge import BuildingKnowledgeGraph

    fault = create_fault("leak", "F3A-MBATH", 1.0, 6, 12, max_leak_demand=0.0002)
    dataset, _ = runner.run_fault(fault)

    kg = BuildingKnowledgeGraph.from_files(*written)
    live = BuildingDiagnosisPipeline(kg, baseline).run_live(dataset, seed=3)

    assert live, "the leak should be detected"
    assert live[min(live)][0]["diagnosis"]["root_cause_asset"] == "F3A-MBATH"

    for hour, incidents in live.items():
        for incident in incidents:
            text = json.dumps(incident["diagnosis"]["message"]) if incident["diagnosis"] else ""
            cited = [int(h) for h in re.findall(r"(\d{2}):00", text)]
            assert all(h <= hour for h in cited), (hour, cited)
