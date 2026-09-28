"""
The building knowledge graph.

``NetworkGraphBuilder`` answers "what is joined to what". A diagnosis
needs more: which way water is supplied, where each asset physically is,
and which instrument watches it. ``BuildingKnowledgeGraph`` adds those
three layers on top of the hydraulic topology and exposes the queries
retrieval is built on.

Entities
--------
assets    every EPANET node and link, keyed by its own ID
          (``F3A-KIT`` tap, ``F3A-KIT-BR`` its branch pipe, ``PUMP1``)
places    ``building``, ``floor:3``, ``flat:F3A``, ``room:F3A-KIT``
sensors   ``sensor:FM-F3A-KIT-BR`` and friends

Relations (typed, directed)
---------------------------
FEEDS        supply direction: node -> pipe -> node, away from the source
LOCATED_IN   asset -> the most specific place it sits in
PART_OF      room -> flat -> floor -> building
MONITORS     sensor -> the asset it measures (with the parameter)

Supply direction is *static* in a building: the network is a tree fed
from the street main, so every asset has exactly one supply path back to
the source, whatever the momentary flow. That is what makes "what do all
the failing flats have in common upstream?" a well-defined question --
and on a looped town network like Net3 it is not, which is why this
class requires a tree.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import networkx as nx

from .builder import NetworkGraphBuilder


FEEDS = "FEEDS"
LOCATED_IN = "LOCATED_IN"
PART_OF = "PART_OF"
MONITORS = "MONITORS"

RELATIONS = (FEEDS, LOCATED_IN, PART_OF, MONITORS)


class BuildingKnowledgeGraph:
    """
    Typed knowledge graph of a building's water supply.

    Args:
        network: WNTR ``WaterNetworkModel`` of the building.
        layout: Layout dict from ``BuildingNetworkGenerator`` (floors,
            flats, rooms, per-asset semantics, sensors).

    Raises:
        ValueError: If the network is not a single tree fed from one
            reservoir, or the layout does not cover every asset.
    """

    def __init__(self, network, layout: dict) -> None:
        self.layout = layout
        self.topology = NetworkGraphBuilder(network).build()

        self._check_layout()
        self._orient()
        self.graph = self._build()

    @classmethod
    def from_files(cls, inp_path, layout_path) -> "BuildingKnowledgeGraph":
        # Imported here so the graph package does not depend on the
        # simulation or building packages at import time.
        from twinrag.simulation.network_loader import WaterNetworkLoader

        network = WaterNetworkLoader(Path(inp_path)).load()
        layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        return cls(network, layout)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _check_layout(self) -> None:
        node_ids = set(self.topology.nodes)
        link_ids = {key for _, _, key in self.topology.edges(keys=True)}

        missing = (node_ids - set(self.layout["nodes"])) | (
            link_ids - set(self.layout["links"])
        )

        if missing:
            raise ValueError(
                "Layout does not describe asset(s): "
                + ", ".join(sorted(missing)[:10])
            )

        clash = node_ids & link_ids
        if clash:
            raise ValueError(
                "Node and link IDs must be distinct in a building graph; "
                "shared: " + ", ".join(sorted(clash)[:10])
            )

    def _orient(self) -> None:
        """
        Orient every asset away from the supply source.

        Breadth-first from the reservoir. Reaching an already-visited
        node over a new link means a loop, and a loop makes "the supply
        path" ambiguous -- refuse rather than guess.
        """

        roots = [
            node
            for node, data in self.topology.nodes(data=True)
            if data["asset_type"] == "reservoir"
        ]

        if len(roots) != 1:
            raise ValueError(
                "A building supply must have exactly one source "
                f"reservoir, found {len(roots)}."
            )

        self.root = roots[0]
        self.parent_node = {self.root: None}   # node -> upstream node
        self.parent_link = {self.root: None}   # node -> link feeding it
        self.link_from = {}                    # link -> upstream node
        self.link_to = {}                      # link -> downstream node
        self.children = {}                     # asset -> downstream assets
        self.depth = {self.root: 0}

        queue = deque([self.root])

        while queue:
            node = queue.popleft()

            for _, neighbour, key in self.topology.edges(node, keys=True):
                if key in self.link_from:
                    continue

                if neighbour in self.parent_node:
                    raise ValueError(
                        f"Link '{key}' closes a loop; the building "
                        "knowledge graph needs a tree-shaped supply."
                    )

                self.parent_node[neighbour] = node
                self.parent_link[neighbour] = key
                self.link_from[key] = node
                self.link_to[key] = neighbour
                self.depth[key] = self.depth[node] + 1
                self.depth[neighbour] = self.depth[node] + 2
                self.children.setdefault(node, []).append(key)
                self.children.setdefault(key, []).append(neighbour)
                queue.append(neighbour)

        unreached = set(self.topology.nodes) - set(self.parent_node)
        if unreached:
            raise ValueError(
                "Assets not connected to the source: "
                + ", ".join(sorted(unreached)[:10])
            )

    def _place_of(self, asset_id: str) -> str:
        """
        Most specific place an asset sits in.
        """

        entry = self.asset(asset_id)

        if entry.get("room"):
            return f"room:{entry['room']}"
        if entry.get("flat"):
            return f"flat:{entry['flat']}"
        if entry.get("floor") is not None:
            return f"floor:{entry['floor']}"
        return "building"

    def _build(self) -> nx.MultiDiGraph:
        layout = self.layout
        g = nx.MultiDiGraph(name=layout["building"]["name"])

        g.add_node(
            "building",
            entity="place",
            place_type="building",
            label=layout["building"]["name"],
        )

        for floor in layout["floors"]:
            fid = f"floor:{floor['index']}"
            g.add_node(fid, entity="place", place_type="floor", label=floor["label"])
            g.add_edge(fid, "building", key=PART_OF, relation=PART_OF)

        for flat in layout["flats"]:
            fid = f"flat:{flat['id']}"
            g.add_node(fid, entity="place", place_type="flat", label=flat["label"])
            g.add_edge(fid, f"floor:{flat['floor']}", key=PART_OF, relation=PART_OF)

        for room in layout["rooms"]:
            rid = f"room:{room['id']}"
            g.add_node(
                rid,
                entity="place",
                place_type="room",
                label=f"Flat {room['flat'][1:]} {room['name'].lower()}",
                wet=room["wet"],
            )
            g.add_edge(rid, f"flat:{room['flat']}", key=PART_OF, relation=PART_OF)

        for asset_id in self.asset_ids():
            entry = self.asset(asset_id)
            g.add_node(
                asset_id,
                entity="asset",
                asset_kind=self.kind(asset_id),
                role=entry["role"],
                label=entry["label"],
            )
            g.add_edge(asset_id, self._place_of(asset_id), key=LOCATED_IN, relation=LOCATED_IN)

        for link_id, upstream in self.link_from.items():
            g.add_edge(upstream, link_id, key=FEEDS, relation=FEEDS)
            g.add_edge(link_id, self.link_to[link_id], key=FEEDS, relation=FEEDS)

        for sensor in layout["sensors"]:
            sid = f"sensor:{sensor['id']}"
            g.add_node(
                sid,
                entity="sensor",
                parameter=sensor["parameter"],
                label=sensor["label"],
            )
            g.add_edge(
                sid,
                sensor["asset_id"],
                key=MONITORS,
                relation=MONITORS,
                parameter=sensor["parameter"],
            )

        return g

    # ------------------------------------------------------------------
    # Asset queries
    # ------------------------------------------------------------------

    def asset_ids(self) -> list:
        return list(self.topology.nodes) + [
            key for _, _, key in self.topology.edges(keys=True)
        ]

    def __contains__(self, asset_id) -> bool:
        return asset_id in self.parent_node or asset_id in self.link_from

    def is_link(self, asset_id: str) -> bool:
        return asset_id in self.link_from

    def asset(self, asset_id: str) -> dict:
        if asset_id in self.layout["nodes"]:
            return self.layout["nodes"][asset_id]
        return self.layout["links"][asset_id]

    def kind(self, asset_id: str) -> str:
        if self.is_link(asset_id):
            return self.layout["links"][asset_id]["kind"]
        return self.topology.nodes[asset_id]["asset_type"]

    def location(self, asset_id: str) -> dict:
        """
        Floor / flat / room of an asset, as human-readable labels.
        """

        entry = self.asset(asset_id)
        floor = entry.get("floor")
        rooms = {room["id"]: room for room in self.layout["rooms"]}

        return {
            "floor": (
                None if floor is None
                else "Ground floor" if floor == 0 else f"Floor {floor}"
            ),
            "flat": f"Flat {entry['flat'][1:]}" if entry.get("flat") else None,
            "room": rooms[entry["room"]]["name"] if entry.get("room") else None,
        }

    def sensors_on(self, asset_id: str) -> list:
        return [
            sensor
            for sensor in self.layout["sensors"]
            if sensor["asset_id"] == asset_id
        ]

    def is_metered(self, asset_id: str) -> bool:
        return bool(self.sensors_on(asset_id))

    # ------------------------------------------------------------------
    # Supply-structure queries
    # ------------------------------------------------------------------

    def upstream(self, asset_id: str) -> list:
        """
        Supply path from an asset back to the source, asset first.

        Alternates nodes and links: ``[F3A-KIT, F3A-KIT-BR, F3A-TK,
        F3A-MAIN-K, F3A-IN, F3A-SUPPLY, DTA-3, ...]``.
        """

        if asset_id not in self:
            raise KeyError(asset_id)

        path = [asset_id]
        current = asset_id

        while True:
            if self.is_link(current):
                current = self.link_from[current]
            else:
                current = self.parent_link[current]
                if current is None:
                    break
            path.append(current)

        return path

    def downstream(self, asset_id: str) -> set:
        """
        Every asset supplied through ``asset_id`` (excluding itself).
        """

        seen = set()
        stack = list(self.children.get(asset_id, []))

        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.children.get(current, []))

        return seen

    def common_supply_point(self, asset_ids) -> str:
        """
        The deepest asset every one of ``asset_ids`` is supplied through
        (one of them, if one supplies all the others).
        """

        asset_ids = list(asset_ids)
        if not asset_ids:
            raise ValueError("common_supply_point needs at least one asset.")

        shared = set(self.upstream(asset_ids[0]))
        for asset_id in asset_ids[1:]:
            shared &= set(self.upstream(asset_id))

        return max(shared, key=lambda a: self.depth[a])

    def neighbourhood(self, asset_id: str, hops: int) -> set:
        """
        Assets within ``hops`` physical links, ignoring supply direction.

        One hop is one pipe: a node's neighbourhood includes the pipes
        touching it and the nodes at their far ends. A link starts from
        both of its end nodes.
        """

        if self.is_link(asset_id):
            starts = [self.link_from[asset_id], self.link_to[asset_id]]
            found = {asset_id}
        else:
            starts = [asset_id]
            found = set()

        for start in starts:
            distances = nx.single_source_shortest_path_length(
                self.topology, start, cutoff=hops
            )
            found.update(distances)

        nodes = {a for a in found if not self.is_link(a)}
        for u, v, key in self.topology.edges(nodes, keys=True):
            if u in nodes and v in nodes:
                found.add(key)

        return found

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def triples(self, entities=None) -> list:
        """
        ``(subject, RELATION, object)`` for every edge, or only for edges
        whose both ends are in ``entities``.
        """

        keep = None if entities is None else set(entities)

        out = []
        for subject, obj, data in self.graph.edges(data=True):
            if keep is not None and (subject not in keep or obj not in keep):
                continue
            out.append((subject, data["relation"], obj))

        return sorted(out)

    def places_of(self, asset_ids) -> set:
        """
        Every place entity containing any of ``asset_ids``, up to the
        building -- so retrieved triples keep their location context.
        """

        places = set()

        for asset_id in asset_ids:
            place = self._place_of(asset_id)
            while place:
                places.add(place)
                parents = [
                    obj
                    for _, obj, key in self.graph.out_edges(place, keys=True)
                    if key == PART_OF
                ]
                place = parents[0] if parents else None

        return places

    def summary(self) -> dict:
        entities = {}
        for _, data in self.graph.nodes(data=True):
            name = data.get("place_type") or data["entity"]
            entities[name] = entities.get(name, 0) + 1

        relations = {}
        for _, _, key in self.graph.edges(keys=True):
            relations[key] = relations.get(key, 0) + 1

        return {"entities": entities, "relations": relations, "source": self.root}
