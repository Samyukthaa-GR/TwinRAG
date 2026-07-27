"""
Knowledge-graph layer (Module 3).

Turns the simulated network into a queryable structure:

    NetworkGraphBuilder  -- static connectivity (what is joined to what)
    GraphSnapshot        -- one timestep of hydraulics laid on top,
                            including flow direction and neighbour lookup
"""

from .builder import NetworkGraphBuilder
from .snapshot import GraphSnapshot

__all__ = [
    "NetworkGraphBuilder",
    "GraphSnapshot",
]
