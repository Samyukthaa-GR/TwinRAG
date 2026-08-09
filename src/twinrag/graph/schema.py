"""
Schema helpers for the TwinRAG physical-network graph.

The graph separates three categories of attributes:

``static``
    Permanent or configuration-level information read from EPANET.

``runtime``
    Time-dependent hydraulic state such as pressure or flowrate.

``anomaly``
    Detection evidence and event annotations attached in later stages.

The helpers in this module produce consistent, JSON-friendly graph
attribute dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    ASSET_TYPES,
    GRAPH_SCHEMA_VERSION,
    GRAPH_TYPE,
    LINK_ASSET_TYPES,
    NODE_ASSET_TYPES,
)


def to_serializable(value: Any) -> Any:
    """
    Convert a value into a graph- and JSON-friendly Python value.

    Args:
        value: Value obtained from WNTR, NumPy, or regular Python code.

    Returns:
        A value composed only of ordinary Python scalar, list, tuple, or
        dictionary types.
    """

    if value is None:
        return None

    if isinstance(value, Enum):
        return value.name

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(to_serializable(item) for item in value)

    if isinstance(value, list):
        return [to_serializable(item) for item in value]

    if isinstance(value, set):
        return sorted(to_serializable(item) for item in value)

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def validate_asset_type(asset_type: str) -> str:
    """
    Validate and normalize a graph asset type.

    Args:
        asset_type: Candidate asset type.

    Returns:
        The normalized lowercase asset type.

    Raises:
        TypeError: If ``asset_type`` is not a string.
        ValueError: If the type is unsupported.
    """

    if not isinstance(asset_type, str):
        raise TypeError(
            "asset_type must be a string, "
            f"received {type(asset_type).__name__}."
        )

    normalized = asset_type.strip().lower()

    if normalized not in ASSET_TYPES:
        supported = ", ".join(sorted(ASSET_TYPES))
        raise ValueError(
            f"Unsupported asset type '{asset_type}'. "
            f"Expected one of: {supported}."
        )

    return normalized


def make_node_attributes(
    *,
    asset_id: str,
    asset_type: str,
    static: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Construct the standard attributes for a graph node.

    Args:
        asset_id: Original EPANET node identifier.
        asset_type: Junction, tank, or reservoir.
        static: Permanent physical metadata.

    Returns:
        A graph node attribute dictionary.

    Raises:
        ValueError: If the asset ID is empty or the asset type is not a
            node asset type.
    """

    normalized_id = str(asset_id).strip()

    if not normalized_id:
        raise ValueError("Node asset_id cannot be empty.")

    normalized_type = validate_asset_type(asset_type)

    if normalized_type not in NODE_ASSET_TYPES:
        raise ValueError(
            f"Asset type '{normalized_type}' cannot be represented as "
            "a topology node."
        )

    static_data = to_serializable(dict(static))

    coordinates = static_data.get("coordinates") or {}
    x = coordinates.get("x", 0.0)
    y = coordinates.get("y", 0.0)

    elevation = static_data.get("elevation")
    if normalized_type == "reservoir":
        elevation = static_data.get("base_head")

    return {
        "asset_id": normalized_id,
        "asset_type": normalized_type,
        "display_name": f"{normalized_type.title()} {normalized_id}",
        "is_node_asset": True,
        "is_link_asset": False,
        "static": static_data,
        "runtime": {},
        "anomaly": {},
        # Compatibility fields retained until GraphSnapshot is refactored.
        "node_type": normalized_type,
        "elevation": elevation,
        "base_demand": static_data.get("base_demand"),
        "x": float(x),
        "y": float(y),
    }


def make_link_attributes(
    *,
    asset_id: str,
    asset_type: str,
    start_node: str,
    end_node: str,
    static: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Construct the standard attributes for a graph edge.

    Args:
        asset_id: Original EPANET link identifier.
        asset_type: Pipe, pump, or valve.
        start_node: Declared EPANET start node.
        end_node: Declared EPANET end node.
        static: Permanent physical metadata.

    Returns:
        A graph edge attribute dictionary.

    Raises:
        ValueError: If identifiers are empty, endpoints are equal, or the
            asset type is not a link asset type.
    """

    normalized_id = str(asset_id).strip()
    normalized_start = str(start_node).strip()
    normalized_end = str(end_node).strip()

    if not normalized_id:
        raise ValueError("Link asset_id cannot be empty.")

    if not normalized_start or not normalized_end:
        raise ValueError(
            f"Link '{normalized_id}' must have valid start and end nodes."
        )

    if normalized_start == normalized_end:
        raise ValueError(
            f"Link '{normalized_id}' cannot connect a node to itself."
        )

    normalized_type = validate_asset_type(asset_type)

    if normalized_type not in LINK_ASSET_TYPES:
        raise ValueError(
            f"Asset type '{normalized_type}' cannot be represented as "
            "a topology link."
        )

    static_data = to_serializable(dict(static))

    return {
        "asset_id": normalized_id,
        "asset_type": normalized_type,
        "display_name": f"{normalized_type.title()} {normalized_id}",
        "is_node_asset": False,
        "is_link_asset": True,
        "start_node": normalized_start,
        "end_node": normalized_end,
        "static": static_data,
        "runtime": {},
        "anomaly": {},
        # Compatibility fields retained until GraphSnapshot is refactored.
        "link_name": normalized_id,
        "link_type": normalized_type,
        "diameter": static_data.get("diameter"),
        "length": static_data.get("length"),
        "roughness": static_data.get("roughness"),
        "minor_loss": static_data.get("minor_loss"),
        "initial_status": static_data.get("initial_status"),
    }


def make_graph_metadata(
    *,
    graph_name: str,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """
    Construct metadata stored in ``graph.graph``.

    Args:
        graph_name: Human-readable network name.
        source: Optional EPANET input source.

    Returns:
        Graph-level metadata.
    """

    normalized_name = str(graph_name).strip()

    if not normalized_name:
        raise ValueError("graph_name cannot be empty.")

    return {
        "name": normalized_name,
        "graph_type": GRAPH_TYPE,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "directed": False,
        "multigraph": True,
        "source": to_serializable(source),
        "runtime_timestamp_s": None,
    }