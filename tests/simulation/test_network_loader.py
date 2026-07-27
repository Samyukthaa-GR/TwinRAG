from pathlib import Path

import pytest

from twinrag.simulation.network_loader import WaterNetworkLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Net3.inp"


def test_net3_loads_successfully():
    loader = WaterNetworkLoader(NETWORK_PATH)

    network = loader.load()

    assert network is not None


def test_net3_component_counts():
    loader = WaterNetworkLoader(NETWORK_PATH)

    loader.load()

    summary = loader.get_summary()

    assert summary["junctions"] == 92
    assert summary["reservoirs"] == 2
    assert summary["tanks"] == 3
    assert summary["pipes"] == 117
    assert summary["pumps"] == 2
    assert summary["nodes_total"] == 97
    assert summary["links_total"] == 119


def test_loader_rejects_missing_file():
    missing_path = (
        PROJECT_ROOT
        / "data"
        / "networks"
        / "does_not_exist.inp"
    )

    loader = WaterNetworkLoader(
        missing_path
    )

    with pytest.raises(FileNotFoundError):
        loader.load()