"""
Building-scale digital twin.

Generates a real, constructible residential plumbing system -- storeys,
flats, rooms, a sump, a transfer pump, a roof tank and gravity down-take
risers -- as an EPANET model plus a 3D layout. The rest of the pipeline
(simulation, fault injection, graph, detection) runs on it unchanged.

    BuildingSpec              -- the design: massing, plan, demand, sizing
    BuildingNetworkGenerator  -- design -> WaterNetworkModel + 3D layout
    load_layout               -- read a saved layout back
"""

from .generator import LAYOUT_SCHEMA, BuildingNetworkGenerator, load_layout
from .spec import BuildingSpec, RoomSpec, TankSpec

__all__ = [
    "BuildingSpec",
    "RoomSpec",
    "TankSpec",
    "BuildingNetworkGenerator",
    "LAYOUT_SCHEMA",
    "load_layout",
]
