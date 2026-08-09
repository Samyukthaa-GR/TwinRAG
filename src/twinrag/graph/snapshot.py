"""
Time-resolved hydraulic view of a water-distribution topology.

``GraphSnapshot`` copies the static ``networkx.MultiGraph`` produced by
``NetworkGraphBuilder`` and attaches one timestamp of hydraulic values.

Static topology remains undirected because EPANET start/end orientation
does not necessarily equal the direction in which water is flowing.
Runtime flow direction is derived from the sign of each link's
flowrate.

The snapshot can also materialize a ``networkx.MultiDiGraph`` oriented
according to the actual water movement at the selected timestamp.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import networkx as nx
import pandas as pd

from .constants import RUNTIME_PARAMETERS
from .schema import to_serializable


_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "timestamp_s",
        "asset_id",
        "asset_type",
        "parameter",
        "value",
    }
)

_DATASET_ASSET_TYPES: frozenset[str] = frozenset(
    {
        "node",
        "link",
    }
)

_NODE_RUNTIME_PARAMETERS: frozenset[str] = frozenset(
    {
        "pressure",
        "demand",
        "head",
    }
)

_LINK_RUNTIME_PARAMETERS: frozenset[str] = frozenset(
    {
        "flowrate",
        "velocity",
        "status",
    }
)


class GraphSnapshot:
    """
    Represent one hydraulic timestep over a static topology graph.

    Args:
        graph: Static topology from ``NetworkGraphBuilder.build()``.
        dataset: Long-format hydraulic dataset.
        timestamp_s: Timestamp to materialize, in seconds.

    Raises:
        TypeError: If the graph, dataset, or timestamp has an invalid
            type.
        ValueError: If the dataset schema or selected timestamp is
            invalid.
    """

    def __init__(
        self,
        graph: nx.MultiGraph,
        dataset: pd.DataFrame,
        timestamp_s: int,
    ) -> None:
        self._validate_graph(graph)
        self._validate_dataset(dataset)

        if not isinstance(timestamp_s, Integral):
            raise TypeError(
                "timestamp_s must be an integer number of seconds, "
                f"received {type(timestamp_s).__name__}."
            )

        self.timestamp_s = int(timestamp_s)
        self.graph = graph.copy()

        frame = dataset.loc[
            dataset["timestamp_s"] == self.timestamp_s
        ].copy()

        if frame.empty:
            available = sorted(
                int(value)
                for value in dataset["timestamp_s"].unique()
            )

            raise ValueError(
                f"No rows at timestamp {self.timestamp_s}. "
                f"Available range: {available[0]}..{available[-1]}."
            )

        self._normalize_selected_frame(frame)
        self._validate_selected_frame(frame)

        self.scenario = self._single_optional_value(
            frame,
            "scenario",
        )

        self.state = self._single_optional_value(
            frame,
            "state",
        )

        self.graph.graph["runtime_timestamp_s"] = self.timestamp_s
        self.graph.graph["runtime_scenario"] = self.scenario
        self.graph.graph["runtime_state"] = self.state

        self._apply(frame)

    @staticmethod
    def _validate_graph(graph: nx.Graph) -> None:
        """
        Validate the static graph supplied to the snapshot.

        Args:
            graph: Candidate NetworkX graph.

        Raises:
            TypeError: If the graph is not an undirected MultiGraph.
        """

        if not isinstance(graph, nx.MultiGraph):
            raise TypeError(
                "graph must be an undirected networkx.MultiGraph "
                "produced by NetworkGraphBuilder."
            )

        if graph.is_directed():
            raise TypeError(
                "graph must be undirected before runtime flow "
                "orientation is applied."
            )

    @staticmethod
    def _validate_dataset(dataset: pd.DataFrame) -> None:
        """
        Validate the general hydraulic dataset structure.

        Args:
            dataset: Candidate dataframe.

        Raises:
            TypeError: If ``dataset`` is not a pandas dataframe.
            ValueError: If required columns or usable rows are missing.
        """

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                "dataset must be a pandas DataFrame, "
                f"received {type(dataset).__name__}."
            )

        if dataset.empty:
            raise ValueError("dataset cannot be empty.")

        missing = sorted(
            _REQUIRED_COLUMNS - set(dataset.columns)
        )

        if missing:
            raise ValueError(
                "dataset is missing required column(s): "
                f"{', '.join(missing)}."
            )

        required_non_null_columns = (
            "timestamp_s",
            "asset_id",
            "asset_type",
            "parameter",
        )

        for column in required_non_null_columns:
            if dataset[column].isna().any():
                raise ValueError(
                    f"dataset column '{column}' contains missing values."
                )

    @staticmethod
    def _normalize_selected_frame(
        frame: pd.DataFrame,
    ) -> None:
        """
        Normalize identifiers and controlled string values in place.

        Args:
            frame: Selected timestamp frame.
        """

        frame["asset_id"] = (
            frame["asset_id"]
            .astype(str)
            .str.strip()
        )

        frame["asset_type"] = (
            frame["asset_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        frame["parameter"] = (
            frame["parameter"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    @staticmethod
    def _single_optional_value(
        frame: pd.DataFrame,
        column: str,
    ) -> Any:
        """
        Return one consistent optional metadata value.

        Args:
            frame: Selected timestamp frame.
            column: Metadata column name.

        Returns:
            The single non-null value or ``None``.

        Raises:
            ValueError: If multiple distinct values occur.
        """

        if column not in frame.columns:
            return None

        values = frame[column].dropna().unique()

        if len(values) == 0:
            return None

        if len(values) > 1:
            rendered = ", ".join(
                sorted(str(value) for value in values)
            )

            raise ValueError(
                f"Timestamp frame contains multiple '{column}' values: "
                f"{rendered}."
            )

        return to_serializable(values[0])

    def _validate_selected_frame(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """
        Validate rows belonging to the requested timestamp.

        EPANET node and link identifiers are not globally unique. A node
        and a link may both use an identifier such as ``101``. Rows are
        therefore resolved using both ``asset_type`` and ``asset_id``.

        Args:
            frame: Dataset rows at the selected timestamp.

        Raises:
            ValueError: If rows are duplicated, unsupported, or reference
                unknown graph assets.
        """

        unsupported_asset_types = sorted(
            set(frame["asset_type"]) - _DATASET_ASSET_TYPES
        )

        if unsupported_asset_types:
            raise ValueError(
                "Timestamp frame contains unsupported asset_type "
                f"value(s): {', '.join(unsupported_asset_types)}."
            )

        unsupported_parameters = sorted(
            set(frame["parameter"]) - RUNTIME_PARAMETERS
        )

        if unsupported_parameters:
            raise ValueError(
                "Timestamp frame contains unsupported parameter(s): "
                f"{', '.join(unsupported_parameters)}."
            )

        empty_asset_ids = frame["asset_id"] == ""

        if empty_asset_ids.any():
            raise ValueError(
                "Timestamp frame contains empty asset identifiers."
            )

        duplicate_mask = frame.duplicated(
            subset=[
                "asset_type",
                "asset_id",
                "parameter",
            ],
            keep=False,
        )

        if duplicate_mask.any():
            duplicates = (
                frame.loc[
                    duplicate_mask,
                    [
                        "asset_type",
                        "asset_id",
                        "parameter",
                    ],
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "asset_type",
                        "asset_id",
                        "parameter",
                    ]
                )
            )

            rendered = ", ".join(
                (
                    f"{row.asset_type}:"
                    f"{row.asset_id}:"
                    f"{row.parameter}"
                )
                for row in duplicates.itertuples(index=False)
            )

            raise ValueError(
                "Timestamp frame contains duplicate "
                "asset/parameter rows: "
                f"{rendered}."
            )

        node_ids = {
            str(node_id)
            for node_id in self.graph.nodes
        }

        link_ids = {
            str(key)
            for _, _, key in self.graph.edges(keys=True)
        }

        frame_node_ids = set(
            frame.loc[
                frame["asset_type"] == "node",
                "asset_id",
            ]
        )

        frame_link_ids = set(
            frame.loc[
                frame["asset_type"] == "link",
                "asset_id",
            ]
        )

        unknown_nodes = sorted(
            frame_node_ids - node_ids
        )

        unknown_links = sorted(
            frame_link_ids - link_ids
        )

        if unknown_nodes or unknown_links:
            details: list[str] = []

            if unknown_nodes:
                preview = ", ".join(unknown_nodes[:10])
                suffix = (
                    "..."
                    if len(unknown_nodes) > 10
                    else ""
                )

                details.append(
                    f"unknown node IDs: {preview}{suffix}"
                )

            if unknown_links:
                preview = ", ".join(unknown_links[:10])
                suffix = (
                    "..."
                    if len(unknown_links) > 10
                    else ""
                )

                details.append(
                    f"unknown link IDs: {preview}{suffix}"
                )

            raise ValueError(
                "Timestamp frame references asset IDs absent from the "
                f"graph ({'; '.join(details)})."
            )

        invalid_node_parameters = sorted(
            {
                f"{row.asset_id}:{row.parameter}"
                for row in frame.loc[
                    frame["asset_type"] == "node"
                ].itertuples(index=False)
                if row.parameter not in _NODE_RUNTIME_PARAMETERS
            }
        )

        invalid_link_parameters = sorted(
            {
                f"{row.asset_id}:{row.parameter}"
                for row in frame.loc[
                    frame["asset_type"] == "link"
                ].itertuples(index=False)
                if row.parameter not in _LINK_RUNTIME_PARAMETERS
            }
        )

        if invalid_node_parameters:
            raise ValueError(
                "Node assets contain unsupported runtime parameter "
                "assignments: "
                f"{', '.join(invalid_node_parameters)}."
            )

        if invalid_link_parameters:
            raise ValueError(
                "Link assets contain unsupported runtime parameter "
                "assignments: "
                f"{', '.join(invalid_link_parameters)}."
            )

    @staticmethod
    def _parameter_values(
        frame: pd.DataFrame,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Index selected rows by asset type, parameter, and asset ID.

        Args:
            frame: Validated timestamp frame.

        Returns:
            Nested mapping:

            ``asset_type -> parameter -> asset_id -> value``.
        """

        indexed: dict[
            str,
            dict[str, dict[str, Any]],
        ] = {
            "node": {},
            "link": {},
        }

        for (
            asset_type,
            parameter,
        ), group in frame.groupby(
            [
                "asset_type",
                "parameter",
            ],
            sort=False,
        ):
            indexed[str(asset_type)][str(parameter)] = {
                str(asset_id): to_serializable(value)
                for asset_id, value in zip(
                    group["asset_id"],
                    group["value"],
                    strict=True,
                )
            }

        return indexed

    def _apply(self, frame: pd.DataFrame) -> None:
        """
        Attach one timestamp of hydraulic state to the graph.

        Args:
            frame: Validated dataset frame for one timestamp.
        """

        values = self._parameter_values(frame)

        node_values = values["node"]
        link_values = values["link"]

        for node_name, data in self.graph.nodes(data=True):
            runtime = dict(data.get("runtime", {}))

            for parameter in _NODE_RUNTIME_PARAMETERS:
                runtime[parameter] = node_values.get(
                    parameter,
                    {},
                ).get(str(node_name))

            runtime["timestamp_s"] = self.timestamp_s

            data["runtime"] = runtime

            # Compatibility fields retained for existing viewer/tests.
            data["pressure"] = runtime.get("pressure")
            data["demand"] = runtime.get("demand")
            data["head"] = runtime.get("head")

        for _, _, key, data in self.graph.edges(
            keys=True,
            data=True,
        ):
            link_id = str(key)
            runtime = dict(data.get("runtime", {}))

            for parameter in _LINK_RUNTIME_PARAMETERS:
                runtime[parameter] = link_values.get(
                    parameter,
                    {},
                ).get(link_id)

            runtime["timestamp_s"] = self.timestamp_s

            flowrate = runtime.get("flowrate")

            if flowrate is None:
                flow_from = None
                flow_to = None
            elif flowrate >= 0:
                flow_from = data["start_node"]
                flow_to = data["end_node"]
            else:
                flow_from = data["end_node"]
                flow_to = data["start_node"]

            runtime["flow_from"] = flow_from
            runtime["flow_to"] = flow_to

            data["runtime"] = runtime

            # Compatibility fields retained for existing viewer/tests.
            data["flowrate"] = flowrate
            data["velocity"] = runtime.get("velocity")
            data["status"] = runtime.get("status")
            data["flow_from"] = flow_from
            data["flow_to"] = flow_to

    def directed(self) -> nx.MultiDiGraph:
        """
        Orient links according to observed runtime flow.

        Links with missing flowrate are omitted because their direction
        cannot be established. Zero-flow links retain their declared
        start-to-end orientation but carry a flowrate of zero.

        Returns:
            A keyed ``networkx.MultiDiGraph`` preserving link identity.
        """

        oriented = nx.MultiDiGraph()

        oriented.graph.update(self.graph.graph)
        oriented.graph["directed"] = True
        oriented.graph["multigraph"] = True

        oriented.add_nodes_from(
            (
                node_name,
                dict(attributes),
            )
            for node_name, attributes in self.graph.nodes(data=True)
        )

        for _, _, key, data in self.graph.edges(
            keys=True,
            data=True,
        ):
            flow_from = data.get("flow_from")
            flow_to = data.get("flow_to")

            if flow_from is None or flow_to is None:
                continue

            oriented.add_edge(
                flow_from,
                flow_to,
                key=str(key),
                **dict(data),
            )

        return oriented

    def neighbors(
        self,
        asset_id: str,
        hops: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Return topology nodes within a bounded number of hops.

        One hop means traversal across one physical link in the static
        water-network topology.

        Args:
            asset_id: EPANET node identifier.
            hops: Maximum topology distance.

        Returns:
            Node metadata ordered by distance and then asset ID.

        Raises:
            TypeError: If ``hops`` is not an integer.
            ValueError: If the node is unknown or ``hops`` is negative.
        """

        normalized_id = str(asset_id).strip()

        if normalized_id not in self.graph:
            raise ValueError(
                f"'{normalized_id}' is not a node in the network."
            )

        if not isinstance(hops, Integral):
            raise TypeError(
                "hops must be an integer, "
                f"received {type(hops).__name__}."
            )

        if hops < 0:
            raise ValueError("hops cannot be negative.")

        distances = nx.single_source_shortest_path_length(
            self.graph,
            normalized_id,
            cutoff=int(hops),
        )

        ordered = sorted(
            distances.items(),
            key=lambda item: (
                item[1],
                str(item[0]),
            ),
        )

        return [
            {
                "asset_id": str(node_name),
                "hops": distance,
                **self.graph.nodes[node_name],
            }
            for node_name, distance in ordered
            if node_name != normalized_id
        ]

    def upstream(
        self,
        asset_id: str,
        max_hops: int | None = None,
    ) -> list[str]:
        """
        Return nodes hydraulically upstream of a selected node.

        Args:
            asset_id: EPANET node identifier.
            max_hops: Optional maximum directed path distance.

        Returns:
            Upstream node identifiers sorted by distance and asset ID.
        """

        return self._directed_reachable(
            asset_id=asset_id,
            direction="upstream",
            max_hops=max_hops,
        )

    def downstream(
        self,
        asset_id: str,
        max_hops: int | None = None,
    ) -> list[str]:
        """
        Return nodes hydraulically downstream of a selected node.

        Args:
            asset_id: EPANET node identifier.
            max_hops: Optional maximum directed path distance.

        Returns:
            Downstream node identifiers sorted by distance and asset ID.
        """

        return self._directed_reachable(
            asset_id=asset_id,
            direction="downstream",
            max_hops=max_hops,
        )

    def _directed_reachable(
        self,
        *,
        asset_id: str,
        direction: str,
        max_hops: int | None,
    ) -> list[str]:
        """
        Compute bounded reachability in the flow-oriented graph.

        Args:
            asset_id: Starting network node.
            direction: ``upstream`` or ``downstream``.
            max_hops: Optional maximum path distance.

        Returns:
            Reachable node identifiers sorted by distance.

        Raises:
            TypeError: If ``max_hops`` is not an integer or ``None``.
            ValueError: If arguments are invalid.
        """

        normalized_id = str(asset_id).strip()

        if normalized_id not in self.graph:
            raise ValueError(
                f"'{normalized_id}' is not a node in the network."
            )

        if max_hops is not None:
            if not isinstance(max_hops, Integral):
                raise TypeError(
                    "max_hops must be an integer or None."
                )

            if max_hops < 0:
                raise ValueError(
                    "max_hops cannot be negative."
                )

        oriented = self.directed()

        if direction == "upstream":
            traversal_graph = oriented.reverse(copy=False)
        elif direction == "downstream":
            traversal_graph = oriented
        else:
            raise ValueError(
                "direction must be 'upstream' or 'downstream'."
            )

        distances = nx.single_source_shortest_path_length(
            traversal_graph,
            normalized_id,
            cutoff=max_hops,
        )

        ordered = sorted(
            distances.items(),
            key=lambda item: (
                item[1],
                str(item[0]),
            ),
        )

        return [
            str(node_name)
            for node_name, _ in ordered
            if node_name != normalized_id
        ]