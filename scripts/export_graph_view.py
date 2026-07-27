"""
Export the twin as a single JSON payload for the visual viewer.

Bundles the static topology plus every timestep of a handful of
scenarios so the viewer can scrub through time and switch between normal
and fault conditions without a backend.
"""

import json
import sys
from pathlib import Path


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# --------------------------------------------------
# TwinRAG imports
# --------------------------------------------------

import pandas as pd

from twinrag.graph import NetworkGraphBuilder
from twinrag.simulation.network_loader import WaterNetworkLoader


OUTPUT_PATH = PROJECT_ROOT / "data" / "generated" / "graph_view.json"

#: Scenarios bundled into the viewer: the baseline plus one clear
#: example of each fault type.
SCENARIOS = [
    ("normal", "data/processed/baseline.csv", None),
    ("leak_101_sev90_t10_16", "data/generated/processed/leak_101_sev90_t10_16.csv", "101"),
    ("pump_failure_10_sev100_t08_14", "data/generated/processed/pump_failure_10_sev100_t08_14.csv", "10"),
    ("blockage_103_sev100_t08_14", "data/generated/processed/blockage_103_sev100_t08_14.csv", "103"),
]


def _round(value, digits=3):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _load_scenario(csv_path: Path):
    """
    Reduce one scenario CSV to per-timestep lookup tables.
    """

    frame = pd.read_csv(csv_path)

    timestamps = sorted(frame["timestamp_s"].unique().tolist())

    series = {"pressure": {}, "demand": {}, "flowrate": {}}
    states = []

    for timestamp in timestamps:
        step = frame[frame["timestamp_s"] == timestamp]

        for parameter in series:
            rows = step[step["parameter"] == parameter]

            series[parameter][str(timestamp)] = {
                str(asset): _round(value)
                for asset, value in zip(rows["asset_id"], rows["value"])
            }

        if "state" in step.columns:
            states.append(str(step["state"].iloc[0]))
        else:
            states.append("normal")

    return timestamps, series, states


def main() -> None:
    network_path = PROJECT_ROOT / "data" / "networks" / "Net3.inp"

    network = WaterNetworkLoader(network_path).load()

    builder = NetworkGraphBuilder(network)
    graph = builder.build()

    print("Graph summary:", builder.summary(graph))

    nodes = [
        {
            "id": name,
            "type": data["node_type"],
            "x": data["x"],
            "y": data["y"],
            "elevation": _round(data["elevation"]),
            "base_demand": _round(data["base_demand"], 6),
            "degree": graph.degree(name),
        }
        for name, data in graph.nodes(data=True)
    ]

    edges = [
        {
            "id": data["link_name"],
            "source": data["start_node"],
            "target": data["end_node"],
            "type": data["link_type"],
            "diameter": _round(data["diameter"]),
            "length": _round(data["length"], 1),
        }
        for _, _, data in graph.edges(data=True)
    ]

    scenarios = {}

    for name, relative_path, target in SCENARIOS:
        csv_path = PROJECT_ROOT / relative_path

        if not csv_path.exists():
            print(f"  SKIP {name} (missing {relative_path})")
            continue

        timestamps, series, states = _load_scenario(csv_path)

        scenarios[name] = {
            "label": name,
            "target": target,
            "timestamps": timestamps,
            "states": states,
            "series": series,
        }

        print(f"  packed {name} ({len(timestamps)} timesteps)")

    payload = {
        "network": network_path.name,
        "nodes": nodes,
        "edges": edges,
        "scenarios": scenarios,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024

    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
