"""
Static physical topology builder for water distribution networks.

``NetworkGraphBuilder`` converts a WNTR ``WaterNetworkModel`` into a
``networkx.MultiGraph``.

EPANET nodes are represented as graph nodes:

- junctions
- tanks
- reservoirs

EPANET links are represented as keyed graph edges:

- pipes
- pumps
- valves

The graph is deliberately undirected. Declared EPANET start and end
nodes are retained as metadata, but actual hydraulic direction is a
runtime property derived from flowrate in ``GraphSnapshot``.

A ``MultiGraph`` is required because a valid water network may contain
multiple links between the same pair of nodes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from .constants import (
    JUNCTION,
    PIPE,
    PUMP,
    RESERVOIR,
    TANK,
    VALVE,
)
from .schema import (
    make_graph_metadata,
    make_link_attributes,
    make_node_attributes,
    to_serializable,
)


_NODE_TYPES: tuple[tuple[str, str], ...] = (
    ("junction_name_list", JUNCTION),
    ("tank_name_list", TANK),
    ("reservoir_name_list", RESERVOIR),
)

_LINK_TYPES: tuple[tuple[str, str], ...] = (
    ("pipe_name_list", PIPE),
    ("pump_name_list", PUMP),
    ("valve_name_list", VALVE),
)


class NetworkGraphBuilder:
    """
    Build a static, metadata-rich water-network topology graph.

    Args:
        network: Loaded WNTR ``WaterNetworkModel``.
        source: Optional path or description of the source EPANET model.

    Raises:
        TypeError: If ``network`` does not provide the required WNTR
            interface.
    """

    def __init__(
        self,
        network: Any,
        source: str | Path | None = None,
    ) -> None:
        self.network = network
        self.source = source

        self._validate_network_interface()

    def _validate_network_interface(self) -> None:
        """
        Validate the minimum WNTR interface required by the builder.

        Raises:
            TypeError: If required attributes or methods are absent.
        """

        required = (
            "get_node",
            "get_link",
            "junction_name_list",
            "tank_name_list",
            "reservoir_name_list",
            "pipe_name_list",
            "pump_name_list",
            "valve_name_list",
        )

        missing = [
            name
            for name in required
            if not hasattr(self.network, name)
        ]

        if missing:
            raise TypeError(
                "network does not provide the required WNTR interface. "
                f"Missing: {', '.join(sorted(missing))}."
            )

    @staticmethod
    def _coordinates(node: Any) -> dict[str, float]:
        """
        Normalize WNTR coordinates into numeric x/y values.

        Args:
            node: WNTR network node.

        Returns:
            Dictionary containing ``x`` and ``y`` floats.
        """

        coordinates = getattr(node, "coordinates", None)

        if not coordinates:
            return {"x": 0.0, "y": 0.0}

        if len(coordinates) != 2:
            raise ValueError(
                "Node coordinates must contain exactly two values."
            )

        return {
            "x": float(coordinates[0]),
            "y": float(coordinates[1]),
        }

    @staticmethod
    def _pattern_name(pattern: Any) -> str | None:
        """
        Convert a WNTR pattern reference into a stable string.

        Args:
            pattern: Pattern object, name, or ``None``.

        Returns:
            Pattern name when available.
        """

        if pattern is None:
            return None

        name = getattr(pattern, "name", None)

        return str(name if name is not None else pattern)

    def _node_static_attributes(
        self,
        name: str,
        node_type: str,
    ) -> dict[str, Any]:
        """
        Collect type-specific static metadata for a network node.

        Args:
            name: EPANET node identifier.
            node_type: Normalized node type.

        Returns:
            Static metadata dictionary.
        """

        node = self.network.get_node(name)

        common: dict[str, Any] = {
            "coordinates": self._coordinates(node),
        }

        if node_type == JUNCTION:
            demand_pattern = getattr(
                node,
                "demand_pattern_name",
                None,
            )

            common.update(
                {
                    "elevation": getattr(node, "elevation", None),
                    "base_demand": getattr(node, "base_demand", None),
                    "demand_pattern": self._pattern_name(
                        demand_pattern
                    ),
                }
            )

        elif node_type == TANK:
            common.update(
                {
                    "elevation": getattr(node, "elevation", None),
                    "initial_level": getattr(
                        node,
                        "init_level",
                        None,
                    ),
                    "minimum_level": getattr(
                        node,
                        "min_level",
                        None,
                    ),
                    "maximum_level": getattr(
                        node,
                        "max_level",
                        None,
                    ),
                    "diameter": getattr(node, "diameter", None),
                    "minimum_volume": getattr(
                        node,
                        "min_vol",
                        None,
                    ),
                    "volume_curve": self._pattern_name(
                        getattr(node, "vol_curve_name", None)
                    ),
                }
            )

        elif node_type == RESERVOIR:
            common.update(
                {
                    "base_head": getattr(node, "base_head", None),
                    "head_pattern": self._pattern_name(
                        getattr(node, "head_pattern_name", None)
                    ),
                }
            )

        else:
            raise ValueError(
                f"Unsupported node type '{node_type}'."
            )

        return to_serializable(common)

    def _link_static_attributes(
        self,
        name: str,
        link_type: str,
    ) -> dict[str, Any]:
        """
        Collect type-specific static metadata for a network link.

        Args:
            name: EPANET link identifier.
            link_type: Normalized link type.

        Returns:
            Static metadata dictionary.
        """

        link = self.network.get_link(name)
        status = getattr(link, "initial_status", None)

        common: dict[str, Any] = {
            "initial_status": (
                to_serializable(status)
                if status is not None
                else None
            ),
            "diameter": getattr(link, "diameter", None),
            "minor_loss": getattr(link, "minor_loss", None),
        }

        if link_type == PIPE:
            common.update(
                {
                    "length": getattr(link, "length", None),
                    "roughness": getattr(link, "roughness", None),
                    "check_valve": bool(
                        getattr(link, "check_valve", False)
                    ),
                }
            )

        elif link_type == PUMP:
            common.update(
                {
                    "pump_type": to_serializable(
                        getattr(link, "pump_type", None)
                    ),
                    "pump_parameter": to_serializable(
                        getattr(link, "pump_parameter", None)
                    ),
                    "base_speed": getattr(link, "base_speed", None),
                    "speed_pattern": self._pattern_name(
                        getattr(link, "speed_pattern_name", None)
                    ),
                }
            )

        elif link_type == VALVE:
            common.update(
                {
                    "valve_type": to_serializable(
                        getattr(link, "valve_type", None)
                    ),
                    "initial_setting": to_serializable(
                        getattr(link, "initial_setting", None)
                    ),
                }
            )

        else:
            raise ValueError(
                f"Unsupported link type '{link_type}'."
            )

        return to_serializable(common)

    def _graph_name(self) -> str:
        """
        Derive a stable graph name from the source or network object.

        Returns:
            Human-readable graph name.
        """

        if self.source is not None:
            source_path = Path(str(self.source))
            return source_path.stem or "water_network"

        network_name = getattr(self.network, "name", None)

        if network_name:
            return str(network_name)

        return "water_network"

    def build(self) -> nx.MultiGraph:
        """
        Build the static physical topology graph.

        Returns:
            A metadata-rich ``networkx.MultiGraph``.

        Raises:
            ValueError: If link endpoints are missing from the graph or a
                duplicate link identifier is encountered.
        """

        graph = nx.MultiGraph()

        graph.graph.update(
            make_graph_metadata(
                graph_name=self._graph_name(),
                source=self.source,
            )
        )

        for list_attribute, node_type in _NODE_TYPES:
            names = getattr(self.network, list_attribute)

            for name in names:
                normalized_name = str(name)

                if normalized_name in graph:
                    raise ValueError(
                        f"Duplicate node identifier '{normalized_name}'."
                    )

                graph.add_node(
                    normalized_name,
                    **make_node_attributes(
                        asset_id=normalized_name,
                        asset_type=node_type,
                        static=self._node_static_attributes(
                            normalized_name,
                            node_type,
                        ),
                    ),
                )

        known_link_ids: set[str] = set()

        for list_attribute, link_type in _LINK_TYPES:
            names = getattr(self.network, list_attribute)

            for name in names:
                normalized_name = str(name)

                if normalized_name in known_link_ids:
                    raise ValueError(
                        f"Duplicate link identifier '{normalized_name}'."
                    )

                known_link_ids.add(normalized_name)

                link = self.network.get_link(normalized_name)
                start_node = str(link.start_node_name)
                end_node = str(link.end_node_name)

                missing_endpoints = [
                    endpoint
                    for endpoint in (start_node, end_node)
                    if endpoint not in graph
                ]

                if missing_endpoints:
                    raise ValueError(
                        f"Link '{normalized_name}' references missing "
                        f"endpoint(s): {', '.join(missing_endpoints)}."
                    )

                graph.add_edge(
                    start_node,
                    end_node,
                    key=normalized_name,
                    **make_link_attributes(
                        asset_id=normalized_name,
                        asset_type=link_type,
                        start_node=start_node,
                        end_node=end_node,
                        static=self._link_static_attributes(
                            normalized_name,
                            link_type,
                        ),
                    ),
                )

        return graph

    def summary(self, graph: nx.Graph) -> dict[str, Any]:
        """
        Return the composition and connectivity of a graph.

        Args:
            graph: Graph produced by this builder.

        Returns:
            Dictionary containing node counts, link counts, component
            count, and graph implementation details.

        Raises:
            TypeError: If ``graph`` is directed.
        """

        if graph.is_directed():
            raise TypeError(
                "Static graph summary expects an undirected graph."
            )

        node_types = Counter(
            data["asset_type"]
            for _, data in graph.nodes(data=True)
        )

        link_types = Counter(
            data["asset_type"]
            for _, _, data in graph.edges(data=True)
        )

        return {
            "nodes_total": graph.number_of_nodes(),
            "edges_total": graph.number_of_edges(),
            "node_types": dict(node_types),
            "link_types": dict(link_types),
            "connected": (
                nx.is_connected(graph)
                if graph.number_of_nodes()
                else False
            ),
            "components": nx.number_connected_components(graph),
            "multigraph": graph.is_multigraph(),
            "directed": graph.is_directed(),
            "schema_version": graph.graph.get("schema_version"),
        }