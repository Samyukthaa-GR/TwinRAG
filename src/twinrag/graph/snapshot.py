"""
Time-resolved view of the twin.

``GraphSnapshot`` layers one timestep of simulated hydraulics onto the
static topology from :class:`NetworkGraphBuilder`: pressure and demand
onto nodes, flowrate onto links, and — derived from the *sign* of each
flowrate — the direction water is actually moving at that moment.

This is the piece that makes neighbour retrieval useful. Asking "what is
next to node 101" is a topology question; asking "what is *upstream* of
node 101 at hour 14" is a root-cause question, and only a snapshot can
answer it.
"""

import networkx as nx


class GraphSnapshot:
    """
    One timestep of network state laid over the static topology.
    """

    def __init__(self, graph: nx.Graph, dataset, timestamp_s: int):
        """
        Parameters
        ----------
        graph
            Static topology from ``NetworkGraphBuilder.build()``.
        dataset
            Long-format dataframe with the canonical schema
            (``timestamp_s, asset_id, asset_type, parameter, value, ...``).
        timestamp_s
            The timestep to materialise.
        """

        self.timestamp_s = int(timestamp_s)
        self.graph = graph.copy()

        frame = dataset[dataset["timestamp_s"] == self.timestamp_s]

        if frame.empty:
            available = sorted(dataset["timestamp_s"].unique())
            raise ValueError(
                f"No rows at timestamp {timestamp_s}. "
                f"Available range: {available[0]}..{available[-1]}."
            )

        self.scenario = (
            frame["scenario"].iloc[0] if "scenario" in frame else None
        )

        self.state = frame["state"].iloc[0] if "state" in frame else None

        self._apply(frame)

    def _apply(self, frame) -> None:
        """
        Write this timestep's values onto nodes and edges.
        """

        values = {
            parameter: dict(
                zip(group["asset_id"].astype(str), group["value"])
            )
            for parameter, group in frame.groupby("parameter")
        }

        pressure = values.get("pressure", {})
        demand = values.get("demand", {})
        flowrate = values.get("flowrate", {})

        for name, data in self.graph.nodes(data=True):
            data["pressure"] = pressure.get(name)
            data["demand"] = demand.get(name)

        for _, _, data in self.graph.edges(data=True):
            flow = flowrate.get(data["link_name"])

            data["flowrate"] = flow

            if flow is None:
                data["flow_from"] = None
                data["flow_to"] = None
                continue

            # A negative flowrate means water is running from the link's
            # declared end node back towards its start node.
            if flow >= 0:
                data["flow_from"] = data["start_node"]
                data["flow_to"] = data["end_node"]
            else:
                data["flow_from"] = data["end_node"]
                data["flow_to"] = data["start_node"]

    def directed(self) -> nx.DiGraph:
        """
        The network oriented by the flow actually observed at this
        timestep. Links carrying no flow are omitted.
        """

        oriented = nx.DiGraph()

        oriented.add_nodes_from(self.graph.nodes(data=True))

        for _, _, data in self.graph.edges(data=True):
            if not data.get("flow_from"):
                continue

            oriented.add_edge(
                data["flow_from"],
                data["flow_to"],
                **data,
            )

        return oriented

    def neighbors(self, asset_id: str, hops: int = 1) -> list:
        """
        Every node within ``hops`` of ``asset_id``, nearest first.

        This is the retrieval primitive for fault triage: when an anomaly
        fires at a node, these are the assets whose readings are worth
        looking at.
        """

        if asset_id not in self.graph:
            raise ValueError(f"'{asset_id}' is not a node in the network.")

        distances = nx.single_source_shortest_path_length(
            self.graph,
            asset_id,
            cutoff=hops,
        )

        return [
            {"asset_id": name, "hops": distance, **self.graph.nodes[name]}
            for name, distance in sorted(distances.items(), key=lambda kv: kv[1])
            if name != asset_id
        ]

    def upstream(self, asset_id: str) -> list:
        """
        Nodes water reaches ``asset_id`` *from* at this timestep.

        A fault here is a candidate cause of a symptom seen at
        ``asset_id``.
        """

        oriented = self.directed()

        if asset_id not in oriented:
            return []

        return sorted(nx.ancestors(oriented, asset_id))

    def downstream(self, asset_id: str) -> list:
        """
        Nodes fed *by* ``asset_id`` at this timestep — the blast radius
        of a fault at this location.
        """

        oriented = self.directed()

        if asset_id not in oriented:
            return []

        return sorted(nx.descendants(oriented, asset_id))
