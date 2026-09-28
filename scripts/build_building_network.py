"""
Generate the building digital twin's water network.

Writes two files from one design (``twinrag.building.BuildingSpec``):

    data/networks/Building_G6.inp           EPANET hydraulic model
    data/networks/Building_G6.layout.json   3D layout + building semantics

The .inp is a normal EPANET file: every other script (simulation, fault
injection, detection, graph) consumes it exactly as it consumes Net3.
"""

import argparse
import sys
from collections import Counter
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

from twinrag.building import BuildingNetworkGenerator, BuildingSpec


NETWORKS_DIR = PROJECT_ROOT / "data" / "networks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--name",
        default="Building_G6",
        help="Output file stem under data/networks/.",
    )
    args = parser.parse_args()

    inp_path = NETWORKS_DIR / f"{args.name}.inp"
    layout_path = NETWORKS_DIR / f"{args.name}.layout.json"

    spec = BuildingSpec()
    wn, layout = BuildingNetworkGenerator(spec).write(inp_path, layout_path)

    node_roles = Counter(node["role"] for node in layout["nodes"].values())
    link_roles = Counter(link["role"] for link in layout["links"].values())
    sensors = Counter(sensor["parameter"] for sensor in layout["sensors"])

    print(spec.name)
    print(
        f"  {spec.floors} floors x {len(spec.flat_letters)} flats = "
        f"{len(layout['flats'])} flats, "
        f"{sum(room['wet'] for room in layout['rooms'])} wet rooms"
    )
    print(
        f"  {wn.num_nodes} nodes "
        f"({wn.num_junctions} junctions, {wn.num_tanks} tanks, "
        f"{wn.num_reservoirs} reservoir), {wn.num_links} links "
        f"({wn.num_pipes} pipes, {wn.num_pumps} pump)"
    )
    print("  node roles: " + ", ".join(f"{k}={v}" for k, v in sorted(node_roles.items())))
    print("  link roles: " + ", ".join(f"{k}={v}" for k, v in sorted(link_roles.items())))
    print("  sensors:    " + ", ".join(f"{k}={v}" for k, v in sorted(sensors.items())))
    print(f"  daily design demand: {spec.flat_daily_demand_m3 * len(layout['flats']):.2f} m3")
    print(f"\n-> {inp_path.relative_to(PROJECT_ROOT)}")
    print(f"-> {layout_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
