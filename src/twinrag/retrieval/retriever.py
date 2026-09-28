"""
Topology-aware retrieval (Phase 4).

Turns one detected ``Incident`` into an ``EvidencePacket``: the small
piece of the building knowledge graph an LLM needs to reason out the
root cause. Nothing here decides the root cause -- every topology fact is
a deterministic graph computation over the alarms; the reasoning is
Phase 5's job.

The physics the retrieval follows
---------------------------------
A building supply is a tree fed from one source, so every asset has one
supply path. Two signatures then localise differently:

**Extra flow** (leaks). Water escaping somewhere is drawn through every
pipe between the source and the escape point, so flow rises all along
that path. The fault is at or below the *deepest* pipe carrying extra
flow -- the excess-flow leaf. Its downstream tap is included.

**Pressure loss** (blockages, lost supply). A closure starves everything
below it. The fault is at or above the *common supply point* of the
starved assets and below the nearest point upstream that still feeds a
healthy sensor -- so retrieval walks up from the common point and stops
at the first node that also feeds a sensor that is fine. What lies
between is the suspect span. With no such bracket (a failure at the
source), it stops at the first metered asset.

**Building-wide shifts are not local evidence.** When the roof tank runs
low, every flat's pressure falls by about the same amount. Those alarms
are real but say nothing about *where*; they are summarised as one fact
and removed from localisation. A flat whose deviation stands out from
that common shift (by ``local_sigma`` of its instrument's noise) is kept
as local evidence.

Everything within ``hops`` pipes of the anchors comes along, and the
sensors in that subgraph that did *not* alarm are listed as negative
evidence: "flat 5A is fine but 4A is dry" is as diagnostic as the alarm.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from twinrag.graph.knowledge import BuildingKnowledgeGraph

from .evidence import EvidencePacket


@dataclass
class RetrievalConfig:
    max_seeds: int = 25
    hops: int = 2
    max_normal_sensors: int = 30
    local_sigma: float = 4.0


def _clock(seconds) -> str:
    return f"{int(seconds) // 3600:02d}:00"


class SubgraphRetriever:
    """
    Extract the evidence subgraph around an incident.
    """

    def __init__(self, kg: BuildingKnowledgeGraph, config: RetrievalConfig | None = None):
        self.kg = kg
        self.config = config or RetrievalConfig()

        self._line_pressure_assets = {
            s["asset_id"]
            for s in kg.layout["sensors"]
            if s["parameter"] == "pressure"
            and not kg.is_link(s["asset_id"])
            and kg.asset(s["asset_id"])["role"] not in ("roof_tank", "sump")
        }

    # ------------------------------------------------------------------
    # Alarm classification
    # ------------------------------------------------------------------

    def _alarms(self, incident) -> dict:
        """
        Strongest candidate per building asset, capped at ``max_seeds``.
        """

        alarmed = {}
        for candidate in incident.candidates:
            asset_id = candidate["asset_id"]
            if asset_id in self.kg and asset_id not in alarmed:
                alarmed[asset_id] = candidate
            if len(alarmed) >= self.config.max_seeds:
                break
        return alarmed

    def _common_shift(self, alarmed: dict):
        """
        Median pressure deviation across every line-pressure sensor, and
        the alarmed ones that do *not* stand out from it.

        Unalarmed sensors count as zero deviation: they read within noise
        of the twin.
        """

        residuals = [
            alarmed[a]["residual"] if a in alarmed and alarmed[a]["parameter"] == "pressure" else 0.0
            for a in self._line_pressure_assets
        ]

        if not residuals:
            return 0.0, set()

        shift = statistics.median(residuals)
        common = set()

        for asset_id in self._line_pressure_assets:
            c = alarmed.get(asset_id)
            if not c or c["parameter"] != "pressure":
                continue
            sigma = abs(c["residual"]) / max(c["score"], 1e-9)
            if abs(c["residual"] - shift) < self.config.local_sigma * sigma:
                common.add(asset_id)

        return shift, common

    def _deepest(self, assets) -> list:
        """
        Assets with no other member of ``assets`` downstream of them.
        """

        kg = self.kg
        assets = set(assets)
        return sorted(
            a for a in assets
            if not (kg.downstream(a) & (assets - {a}))
        )

    def _feeds_healthy_sensor(self, node: str, starved: set, local: set) -> bool:
        """
        Does ``node`` also supply a pressure sensor outside the starved
        subtree that is not locally alarmed?
        """

        for asset_id in self._line_pressure_assets:
            if asset_id in starved or asset_id in local:
                continue
            if asset_id in self.kg.downstream(node):
                return True
        return False

    def _suspect_span(self, anchor: str, local_pressure: set):
        """
        Walk up from the pressure-loss anchor to the first node that also
        feeds a healthy sensor (the healthy bound), or failing that to the
        first metered asset. Returns ``(span, bound)``.
        """

        kg = self.kg
        starved = kg.downstream(anchor) | {anchor}
        span = [anchor]

        for asset in kg.upstream(anchor)[1:]:
            if not kg.is_link(asset) and self._feeds_healthy_sensor(asset, starved, local_pressure):
                return span, asset
            span.append(asset)
            if kg.is_metered(asset):
                return span, None

        return span, None

    # ------------------------------------------------------------------

    def retrieve(self, incident, incident_id: str = "INC-1") -> EvidencePacket:
        kg = self.kg
        cfg = self.config

        alarmed = self._alarms(incident)
        if not alarmed:
            raise ValueError("Incident has no candidates located in the building.")

        shift, common_mode = self._common_shift(alarmed)
        local = {a: c for a, c in alarmed.items() if a not in common_mode}

        excess_flow = [a for a, c in local.items() if c["parameter"] == "flowrate" and c["residual"] > 0]

        # Tank "pressure" is water level: a separate signal about the
        # supply *into* the tank, so it never joins the line-pressure
        # cluster (a low roof tank would drag the common point to the roof).
        tanks = {
            a: c for a, c in local.items()
            if c["parameter"] == "pressure" and kg.asset(a)["role"] in ("roof_tank", "sump")
        }
        pressure_loss = [
            a for a, c in local.items()
            if c["parameter"] == "pressure" and c["residual"] < 0 and a not in tanks
        ]

        anchors = []
        facts = {}

        # low tank -> walk up its supply to the first metered asset
        low_tanks = sorted(a for a, c in tanks.items() if c["residual"] < 0)
        if tanks:
            facts["tank_level_change"] = {
                a: f"{abs(c['residual']):.2f} m {'above' if c['residual'] > 0 else 'below'} the twin"
                for a, c in sorted(tanks.items())
            }
        for tank in low_tanks:
            supply = [tank]
            for asset in kg.upstream(tank)[1:]:
                supply.append(asset)
                if kg.is_metered(asset):
                    break
            facts.setdefault("low_tank_supply_chain", {})[tank] = supply
            anchors += supply

        # extra flow -> deepest pipe carrying it
        if excess_flow:
            leaves = self._deepest(excess_flow)
            taps = [kg.link_to[a] for a in leaves if kg.is_link(a)]
            facts["extra_flow"] = {
                "deepest_assets_with_extra_flow": leaves,
                "supplied_through_them": taps,
                "meaning": "water beyond the twin's prediction is drawn at or below these assets",
            }
            anchors += leaves + taps

        # pressure loss -> common point, bracketed from above
        span, bound = [], None
        if pressure_loss:
            point = kg.common_supply_point(pressure_loss)
            span, bound = self._suspect_span(point, set(pressure_loss))
            facts["pressure_loss"] = {
                "assets_with_pressure_loss": sorted(pressure_loss),
                "common_supply_point": point,
                "suspect_span_upward": span,
                "nearest_upstream_point_still_feeding_a_healthy_sensor": bound,
                "meaning": (
                    "every asset with lost pressure is supplied through the "
                    "common point; a restriction there or in the suspect span "
                    "would starve all of them"
                ),
            }
            anchors += span + ([bound] if bound else [])

        if common_mode:
            deviations = [alarmed[a]["residual"] for a in common_mode]
            facts["building_wide_pressure_shift"] = {
                "sensors_shifted": len(common_mode),
                "of_line_pressure_sensors": len(self._line_pressure_assets),
                "median_shift_m": round(shift, 2),
                "range_m": [round(min(deviations), 2), round(max(deviations), 2)],
                "meaning": "a near-uniform pressure change across the building, not local to one flat",
            }

        if not anchors:
            # nothing localised: fall back to the common point of all alarms
            anchors = [kg.common_supply_point(self._deepest(alarmed))]

        # supply path from the anchors back to the source
        root_anchor = kg.common_supply_point(anchors)
        supply_path = kg.upstream(root_anchor)
        facts["supply_path_to_source"] = supply_path

        context = set(local) | set(supply_path) | set(span)
        for anchor in anchors:
            context |= set(kg.upstream(anchor)[: kg.upstream(anchor).index(root_anchor) + 1]) \
                if root_anchor in kg.upstream(anchor) else {anchor}
            context |= kg.neighbourhood(anchor, cfg.hops)

        # The rest of an anchor's flat: "the other rooms of 3A are fine"
        # is what pins a fault to one room, and those rooms sit on the
        # flat's other tee, beyond a short BFS.
        flats = {kg.asset(a).get("flat") for a in anchors} - {None}
        context |= {a for a in kg.asset_ids() if kg.asset(a).get("flat") in flats}

        context |= set(common_mode)

        observations = [self._alarm(a, c) for a, c in local.items()]

        if common_mode:
            observations.append(
                {
                    "sensors": sorted(f"sensor:{kg.sensors_on(a)[0]['id']}" for a in common_mode),
                    "assets": sorted(common_mode),
                    "parameter": "pressure",
                    "status": "alarm (building-wide shift)",
                    "reading": f"about {abs(shift):.2f} m {'above' if shift > 0 else 'below'} the twin at each",
                }
            )

        normal = [
            sensor
            for asset_id in sorted(context, key=lambda a: (kg.depth[a], a))
            if asset_id not in alarmed
            for sensor in kg.sensors_on(asset_id)
        ][: cfg.max_normal_sensors]

        observations += [
            {
                "sensor": f"sensor:{s['id']}",
                "asset": s["asset_id"],
                "parameter": s["parameter"],
                "status": "normal",
                "reading": "within sensor noise of the twin",
            }
            for s in normal
        ]

        sensor_ids = set()
        for obs in observations:
            if obs.get("sensor"):
                sensor_ids.add(obs["sensor"])
            sensor_ids.update(obs.get("sensors", []))

        order = sorted(context, key=lambda a: (kg.depth[a], a))
        entities = context | kg.places_of(context) | sensor_ids
        building = kg.layout["building"]

        return EvidencePacket(
            incident={
                "id": incident_id,
                "first_alarm": _clock(incident.detected_at_s),
                "last_alarm": _clock(incident.last_seen_s),
                "alarmed_sensor_count": len(alarmed),
                "method": (
                    "Each sensor reading is compared with the digital twin's "
                    "prediction for the same hour; score is the deviation in "
                    "units of that instrument's noise (alarm at >= 4)."
                ),
            },
            building={
                "name": building["name"],
                "floors": building["floors"],
                "flats_per_floor": building["flats_per_floor"],
                "supply_scheme": building["supply_scheme"],
                "relation_meanings": {
                    "FEEDS": "water is supplied from subject to object",
                    "LOCATED_IN": "asset sits in that place",
                    "PART_OF": "place is part of the larger place",
                    "MONITORS": "sensor measures that asset",
                },
            },
            observations=observations,
            assets=[self._asset(a) for a in order],
            relations=[list(t) for t in kg.triples(entities)],
            topology_facts=facts,
            allowed_ids=sorted(entities),
        )

    # ------------------------------------------------------------------

    def _alarm(self, asset_id: str, candidate: dict) -> dict:
        sensors = self.kg.sensors_on(asset_id)
        residual = candidate["residual"]

        if candidate["parameter"] == "flowrate":
            amount = f"{abs(residual) * 1000:.3f} L/s"
        else:
            amount = f"{abs(residual):.2f} m"

        return {
            "sensor": f"sensor:{sensors[0]['id']}" if sensors else None,
            "asset": asset_id,
            "parameter": candidate["parameter"],
            "status": "alarm",
            "reading": f"{amount} {'above' if residual > 0 else 'below'} the twin",
            "score": round(float(candidate["score"]), 1),
            "first_seen": _clock(candidate["first_seen_s"]),
        }

    def _asset(self, asset_id: str) -> dict:
        kg = self.kg
        entry = kg.asset(asset_id)

        record = {
            "id": asset_id,
            "kind": kg.kind(asset_id),
            "role": entry["role"],
            "label": entry["label"],
            "location": {k: v for k, v in kg.location(asset_id).items() if v},
            "metered": kg.is_metered(asset_id),
        }

        if kg.is_link(asset_id) and entry.get("diameter"):
            record["diameter_mm"] = round(entry["diameter"] * 1000)

        return record
