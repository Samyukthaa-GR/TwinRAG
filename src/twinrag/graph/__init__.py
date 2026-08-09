"""
Knowledge-graph layer (Module 3).

Turns the simulated network into a queryable structure:

    NetworkGraphBuilder  -- static connectivity (what is joined to what)
    GraphSnapshot        -- one timestep of hydraulics laid on top,
                            including flow direction and neighbour lookup
    GraphValidator       -- structural checks on a built graph
"""

from .builder import NetworkGraphBuilder
from .snapshot import GraphSnapshot
from .validation import GraphValidationError, GraphValidator

__all__ = [
    "NetworkGraphBuilder",
    "GraphSnapshot",
    "GraphValidator",
    "GraphValidationError",
]
