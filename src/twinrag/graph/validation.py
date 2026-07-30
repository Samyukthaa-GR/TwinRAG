"""
Validation utilities for TwinRAG water-distribution graphs.

The validator checks whether a graph follows the structural invariants
defined by the TwinRAG graph schema.

It is intentionally independent of WNTR. This allows graphs to be
validated after construction, serialization, deserialization, runtime
annotation, or retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from .constants import (
    GRAPH_SCHEMA_VERSION,
    GRAPH_TYPE,
    LINK_ASSET_TYPES,
    NODE_ASSET_TYPES,
)


_REQUIRED_GRAPH_METADATA: frozenset[str] = frozenset(
    {
        "name",
        "graph_type",
        "schema_version",
        "directed",
        "multigraph",
        "source",
        "runtime_timestamp_s",
    }
)

_REQUIRED_NODE_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "asset_id",
        "asset_type",
        "display_name",
        "is_node_asset",
        "is_link_asset",
        "static",
        "runtime",
        "anomaly",
    }
)

_REQUIRED_EDGE_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "asset_id",
        "asset_type",
        "display_name",
        "is_node_asset",
        "is_link_asset",
        "start_node",
        "end_node",
        "static",
        "runtime",
        "anomaly",
    }
)


class GraphValidationError(ValueError):
    """
    Raised when a graph violates the TwinRAG graph schema.
    """


class GraphValidator:
    """
    Validate TwinRAG water-distribution topology graphs.

    The validator currently targets the undirected static topology graph
    produced by ``NetworkGraphBuilder``.

    Runtime values may already be attached by ``GraphSnapshot`` because
    snapshot graphs retain the same undirected ``MultiGraph`` structure.

    Example:
        >>> validator = GraphValidator()
        >>> validator.validate(graph)
    """

    def validate(self, graph: nx.Graph) -> None:
        """
        Validate a complete TwinRAG topology graph.

        Args:
            graph: Candidate graph.

        Raises:
            TypeError: If ``graph`` is not a NetworkX graph.
            GraphValidationError: If the graph violates the TwinRAG
                schema or topology invariants.
        """

        self._validate_graph_type(graph)
        self._validate_graph_metadata(graph)
        self._validate_nodes(graph)
        self._validate_edges(graph)
        self._validate_connectivity(graph)

    @staticmethod
    def _validate_graph_type(graph: nx.Graph) -> None:
        """
        Validate the graph implementation.

        Args:
            graph: Candidate graph.

        Raises:
            TypeError: If the object is not a NetworkX graph.
            GraphValidationError: If it is not an undirected MultiGraph.
        """

        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "graph must be a NetworkX graph, "
                f"received {type(graph).__name__}."
            )

        if not isinstance(graph, nx.MultiGraph):
            raise GraphValidationError(
                "TwinRAG topology graphs must use "
                "networkx.MultiGraph."
            )

        if graph.is_directed():
            raise GraphValidationError(
                "TwinRAG topology graphs must be undirected."
            )

        if not graph.is_multigraph():
            raise GraphValidationError(
                "TwinRAG topology graphs must support parallel links."
            )

    @staticmethod
    def _validate_graph_metadata(graph: nx.MultiGraph) -> None:
        """
        Validate graph-level metadata.

        Args:
            graph: Candidate topology graph.

        Raises:
            GraphValidationError: If required metadata is missing or
                inconsistent.
        """

        missing = sorted(
            _REQUIRED_GRAPH_METADATA - set(graph.graph)
        )

        if missing:
            raise GraphValidationError(
                "Graph metadata is missing required field(s): "
                f"{', '.join(missing)}."
            )

        name = graph.graph["name"]

        if not isinstance(name, str) or not name.strip():
            raise GraphValidationError(
                "Graph metadata field 'name' must be a non-empty string."
            )

        graph_type = graph.graph["graph_type"]

        if graph_type != GRAPH_TYPE:
            raise GraphValidationError(
                "Invalid graph_type metadata. "
                f"Expected '{GRAPH_TYPE}', received '{graph_type}'."
            )

        schema_version = graph.graph["schema_version"]

        if schema_version != GRAPH_SCHEMA_VERSION:
            raise GraphValidationError(
                "Unsupported graph schema version. "
                f"Expected '{GRAPH_SCHEMA_VERSION}', "
                f"received '{schema_version}'."
            )

        directed = graph.graph["directed"]

        if directed is not False:
            raise GraphValidationError(
                "Graph metadata field 'directed' must be False."
            )

        multigraph = graph.graph["multigraph"]

        if multigraph is not True:
            raise GraphValidationError(
                "Graph metadata field 'multigraph' must be True."
            )

        runtime_timestamp = graph.graph["runtime_timestamp_s"]

        if (
            runtime_timestamp is not None
            and not isinstance(runtime_timestamp, int)
        ):
            raise GraphValidationError(
                "Graph metadata field 'runtime_timestamp_s' must be "
                "an integer or None."
            )

    def _validate_nodes(self, graph: nx.MultiGraph) -> None:
        """
        Validate every topology node.

        Args:
            graph: Candidate topology graph.

        Raises:
            GraphValidationError: If a node violates the schema.
        """

        if graph.number_of_nodes() == 0:
            raise GraphValidationError(
                "Topology graph cannot be empty."
            )

        for node_id, attributes in graph.nodes(data=True):
            normalized_node_id = str(node_id)

            self._validate_required_attributes(
                attributes=attributes,
                required=_REQUIRED_NODE_ATTRIBUTES,
                asset_description=f"Node '{normalized_node_id}'",
            )

            asset_id = attributes["asset_id"]

            if not isinstance(asset_id, str) or not asset_id.strip():
                raise GraphValidationError(
                    f"Node '{normalized_node_id}' has an invalid "
                    "asset_id."
                )

            if asset_id != normalized_node_id:
                raise GraphValidationError(
                    f"Node '{normalized_node_id}' has asset_id "
                    f"'{asset_id}'. Node keys must match asset IDs."
                )

            asset_type = attributes["asset_type"]

            if asset_type not in NODE_ASSET_TYPES:
                expected = ", ".join(
                    sorted(NODE_ASSET_TYPES)
                )

                raise GraphValidationError(
                    f"Node '{normalized_node_id}' has unsupported "
                    f"asset_type '{asset_type}'. Expected one of: "
                    f"{expected}."
                )

            if attributes["is_node_asset"] is not True:
                raise GraphValidationError(
                    f"Node '{normalized_node_id}' must have "
                    "is_node_asset=True."
                )

            if attributes["is_link_asset"] is not False:
                raise GraphValidationError(
                    f"Node '{normalized_node_id}' must have "
                    "is_link_asset=False."
                )

            self._validate_display_name(
                asset_id=asset_id,
                asset_type=asset_type,
                display_name=attributes["display_name"],
                asset_description=f"Node '{normalized_node_id}'",
            )

            self._validate_container(
                attributes["static"],
                container_name="static",
                asset_description=f"Node '{normalized_node_id}'",
            )

            self._validate_container(
                attributes["runtime"],
                container_name="runtime",
                asset_description=f"Node '{normalized_node_id}'",
            )

            self._validate_container(
                attributes["anomaly"],
                container_name="anomaly",
                asset_description=f"Node '{normalized_node_id}'",
            )

    def _validate_edges(self, graph: nx.MultiGraph) -> None:
        """
        Validate every topology edge.

        Args:
            graph: Candidate topology graph.

        Raises:
            GraphValidationError: If an edge violates the schema.
        """

        if graph.number_of_edges() == 0:
            raise GraphValidationError(
                "Topology graph must contain at least one link."
            )

        observed_link_ids: set[str] = set()

        for start, end, key, attributes in graph.edges(
            keys=True,
            data=True,
        ):
            normalized_key = str(key)
            edge_description = (
                f"Edge '{normalized_key}' "
                f"between '{start}' and '{end}'"
            )

            self._validate_required_attributes(
                attributes=attributes,
                required=_REQUIRED_EDGE_ATTRIBUTES,
                asset_description=edge_description,
            )

            asset_id = attributes["asset_id"]

            if not isinstance(asset_id, str) or not asset_id.strip():
                raise GraphValidationError(
                    f"{edge_description} has an invalid asset_id."
                )

            if asset_id != normalized_key:
                raise GraphValidationError(
                    f"{edge_description} has asset_id '{asset_id}'. "
                    "Edge keys must match link asset IDs."
                )

            if asset_id in observed_link_ids:
                raise GraphValidationError(
                    f"Duplicate link asset_id '{asset_id}' detected."
                )

            observed_link_ids.add(asset_id)

            asset_type = attributes["asset_type"]

            if asset_type not in LINK_ASSET_TYPES:
                expected = ", ".join(
                    sorted(LINK_ASSET_TYPES)
                )

                raise GraphValidationError(
                    f"{edge_description} has unsupported asset_type "
                    f"'{asset_type}'. Expected one of: {expected}."
                )

            if attributes["is_node_asset"] is not False:
                raise GraphValidationError(
                    f"{edge_description} must have "
                    "is_node_asset=False."
                )

            if attributes["is_link_asset"] is not True:
                raise GraphValidationError(
                    f"{edge_description} must have "
                    "is_link_asset=True."
                )

            self._validate_display_name(
                asset_id=asset_id,
                asset_type=asset_type,
                display_name=attributes["display_name"],
                asset_description=edge_description,
            )

            self._validate_edge_endpoints(
                graph=graph,
                start=start,
                end=end,
                attributes=attributes,
                edge_description=edge_description,
            )

            self._validate_container(
                attributes["static"],
                container_name="static",
                asset_description=edge_description,
            )

            self._validate_container(
                attributes["runtime"],
                container_name="runtime",
                asset_description=edge_description,
            )

            self._validate_container(
                attributes["anomaly"],
                container_name="anomaly",
                asset_description=edge_description,
            )

    @staticmethod
    def _validate_required_attributes(
        *,
        attributes: Mapping[str, Any],
        required: frozenset[str],
        asset_description: str,
    ) -> None:
        """
        Validate required attributes for one graph asset.

        Args:
            attributes: Node or edge attributes.
            required: Required attribute names.
            asset_description: Human-readable asset label.

        Raises:
            GraphValidationError: If required attributes are missing.
        """

        missing = sorted(
            required - set(attributes)
        )

        if missing:
            raise GraphValidationError(
                f"{asset_description} is missing required "
                f"attribute(s): {', '.join(missing)}."
            )

    @staticmethod
    def _validate_display_name(
        *,
        asset_id: str,
        asset_type: str,
        display_name: Any,
        asset_description: str,
    ) -> None:
        """
        Validate a graph asset's display name.

        Args:
            asset_id: Canonical asset identifier.
            asset_type: Canonical asset type.
            display_name: Candidate display name.
            asset_description: Human-readable asset label.

        Raises:
            GraphValidationError: If the display name is invalid.
        """

        expected = f"{asset_type.title()} {asset_id}"

        if not isinstance(display_name, str):
            raise GraphValidationError(
                f"{asset_description} has a non-string display_name."
            )

        if display_name != expected:
            raise GraphValidationError(
                f"{asset_description} has display_name "
                f"'{display_name}'. Expected '{expected}'."
            )

    @staticmethod
    def _validate_container(
        value: Any,
        *,
        container_name: str,
        asset_description: str,
    ) -> None:
        """
        Validate a static, runtime, or anomaly container.

        Args:
            value: Candidate container.
            container_name: Container field name.
            asset_description: Human-readable asset label.

        Raises:
            GraphValidationError: If the value is not mapping-like.
        """

        if not isinstance(value, Mapping):
            raise GraphValidationError(
                f"{asset_description} attribute '{container_name}' "
                "must be a mapping."
            )

    @staticmethod
    def _validate_edge_endpoints(
        *,
        graph: nx.MultiGraph,
        start: Any,
        end: Any,
        attributes: Mapping[str, Any],
        edge_description: str,
    ) -> None:
        """
        Validate declared edge endpoints.

        Since the graph is undirected, NetworkX may expose an edge as
        either ``(u, v)`` or ``(v, u)``. The declared EPANET
        ``start_node`` and ``end_node`` must therefore match the same
        unordered endpoint pair.

        Args:
            graph: Candidate graph.
            start: NetworkX edge start.
            end: NetworkX edge end.
            attributes: Edge metadata.
            edge_description: Human-readable edge label.

        Raises:
            GraphValidationError: If endpoints are invalid or
                inconsistent.
        """

        declared_start = str(attributes["start_node"]).strip()
        declared_end = str(attributes["end_node"]).strip()

        if not declared_start or not declared_end:
            raise GraphValidationError(
                f"{edge_description} has empty declared endpoints."
            )

        if declared_start == declared_end:
            raise GraphValidationError(
                f"{edge_description} cannot connect a node to itself."
            )

        if declared_start not in graph:
            raise GraphValidationError(
                f"{edge_description} references missing start_node "
                f"'{declared_start}'."
            )

        if declared_end not in graph:
            raise GraphValidationError(
                f"{edge_description} references missing end_node "
                f"'{declared_end}'."
            )

        networkx_endpoints = {
            str(start),
            str(end),
        }

        declared_endpoints = {
            declared_start,
            declared_end,
        }

        if networkx_endpoints != declared_endpoints:
            raise GraphValidationError(
                f"{edge_description} declares endpoints "
                f"'{declared_start}' and '{declared_end}', which do "
                "not match the graph edge endpoints."
            )

    @staticmethod
    def _validate_connectivity(graph: nx.MultiGraph) -> None:
        """
        Validate physical topology connectivity.

        Args:
            graph: Candidate topology graph.

        Raises:
            GraphValidationError: If the graph is disconnected.
        """

        if not nx.is_connected(graph):
            components = nx.number_connected_components(graph)

            raise GraphValidationError(
                "Topology graph must be connected. "
                f"Detected {components} connected components."
            )