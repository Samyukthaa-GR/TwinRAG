"""
Static topology graph.

``NetworkGraphBuilder`` turns a WNTR ``WaterNetworkModel`` into a
``networkx`` graph describing what is physically connected to what:
junctions, tanks and reservoirs as nodes; pipes, pumps and valves as
edges.

The graph is **undirected**. Flow direction is deliberately not part of
the structure: in Net3, 56 of 119 links reverse direction over a normal
24-hour day as tanks switch between filling and draining. Direction is a
property of a moment in time, not of the network, so it belongs to
``GraphSnapshot`` instead.

Edges are keyed by link name so networks containing parallel pipes
between the same pair of nodes do not silently lose one.
"""

import networkx as nx


#: WNTR node classes mapped to the node_type recorded on the graph.
_NODE_TYPES = (
    ("junction_name_list", "junction"),
    ("tank_name_list", "tank"),
    ("reservoir_name_list", "reservoir"),
)

#: WNTR link classes mapped to the link_type recorded on the graph.
_LINK_TYPES = (
    ("pipe_name_list", "pipe"),
    ("pump_name_list", "pump"),
    ("valve_name_list", "valve"),
)


class NetworkGraphBuilder:
    """
    Builds the static connectivity graph for a water network.
    """

    def __init__(self, network):
        self.network = network

    def _node_attributes(self, name: str, node_type: str) -> dict:
        """
        Collect the attributes worth carrying onto a graph node.

        Reservoirs carry a fixed head rather than an elevation, so the
        two are normalised onto a single ``elevation`` field to keep
        downstream code from special-casing them.
        """

        node = self.network.get_node(name)

        coordinates = getattr(node, "coordinates", None) or (0.0, 0.0)

        if node_type == "reservoir":
            elevation = getattr(node, "base_head", None)
        else:
            elevation = getattr(node, "elevation", None)

        base_demand = None

        if node_type == "junction":
            base_demand = getattr(node, "base_demand", None)

        return {
            "node_type": node_type,
            "elevation": elevation,
            "base_demand": base_demand,
            "x": float(coordinates[0]),
            "y": float(coordinates[1]),
        }

    def _edge_attributes(self, name: str, link_type: str) -> dict:
        """
        Collect the attributes worth carrying onto a graph edge.

        ``start_node``/``end_node`` are retained because the sign of a
        link's flowrate is only meaningful relative to them.
        """

        link = self.network.get_link(name)

        status = getattr(link, "initial_status", None)

        return {
            "link_name": name,
            "link_type": link_type,
            "start_node": link.start_node_name,
            "end_node": link.end_node_name,
            "diameter": getattr(link, "diameter", None),
            "length": getattr(link, "length", None),
            "roughness": getattr(link, "roughness", None),
            "minor_loss": getattr(link, "minor_loss", None),
            "initial_status": str(status) if status is not None else None,
        }

    def build(self) -> nx.Graph:
        """
        Build the static topology graph.

        Returns
        -------
        networkx.Graph
            Nodes keyed by asset ID, edges keyed by link name.
        """

        graph = nx.Graph()

        for list_attr, node_type in _NODE_TYPES:
            for name in getattr(self.network, list_attr):
                graph.add_node(
                    name,
                    **self._node_attributes(name, node_type),
                )

        for list_attr, link_type in _LINK_TYPES:
            for name in getattr(self.network, list_attr):
                link = self.network.get_link(name)

                graph.add_edge(
                    link.start_node_name,
                    link.end_node_name,
                    key=name,
                    **self._edge_attributes(name, link_type),
                )

        return graph

    def summary(self, graph: nx.Graph) -> dict:
        """
        Composition of a built graph, for sanity checks.
        """

        node_types = {}
        link_types = {}

        for _, data in graph.nodes(data=True):
            node_type = data["node_type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1

        for _, _, data in graph.edges(data=True):
            link_type = data["link_type"]
            link_types[link_type] = link_types.get(link_type, 0) + 1

        return {
            "nodes_total": graph.number_of_nodes(),
            "edges_total": graph.number_of_edges(),
            "node_types": node_types,
            "link_types": link_types,
            "connected": nx.is_connected(graph) if graph.number_of_nodes() else False,
            "components": nx.number_connected_components(graph),
        }
