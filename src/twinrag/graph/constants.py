"""
Controlled vocabulary for the TwinRAG graph layer.

Keeping graph labels and metadata keys in one module prevents spelling
differences from propagating into traversal, anomaly annotation, and
GraphRAG serialization code.
"""

from typing import Final


GRAPH_SCHEMA_VERSION: Final[str] = "1.0.0"
GRAPH_TYPE: Final[str] = "water_distribution_topology"

JUNCTION: Final[str] = "junction"
TANK: Final[str] = "tank"
RESERVOIR: Final[str] = "reservoir"

PIPE: Final[str] = "pipe"
PUMP: Final[str] = "pump"
VALVE: Final[str] = "valve"

NODE_ASSET_TYPES: Final[frozenset[str]] = frozenset(
    {
        JUNCTION,
        TANK,
        RESERVOIR,
    }
)

LINK_ASSET_TYPES: Final[frozenset[str]] = frozenset(
    {
        PIPE,
        PUMP,
        VALVE,
    }
)

ASSET_TYPES: Final[frozenset[str]] = (
    NODE_ASSET_TYPES | LINK_ASSET_TYPES
)

RUNTIME_PARAMETERS: Final[frozenset[str]] = frozenset(
    {
        "pressure",
        "demand",
        "flowrate",
        "head",
        "velocity",
        "status",
    }
)