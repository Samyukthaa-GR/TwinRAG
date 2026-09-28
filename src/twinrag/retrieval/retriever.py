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
    #: Hours before the incident opened in which lone readings (too few
    #: assets at once to open an incident) are kept as early evidence.
    lookback_hours: int = 3
    #: A pressure loss smaller than this fraction of the largest one is
    #: reported but not used to locate the fault -- a dry flat (-20 m) and
    #: a knock-on wobble elsewhere (-1.2 m) must not be averaged together.
    minor_fraction: float = 0.25


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

        path = kg.upstream(anchor)[1:]

        for i, asset in enumerate(path):
            if not kg.is_link(asset) and self._feeds_healthy_sensor(asset, starved, local_pressure):
                return span, asset
            span.append(asset)
            if kg.is_metered(asset):
                # Stop the span at the meter, but still look one node up:
                # if it feeds a healthy sensor, the fault is bracketed
                # (flat 1B dry while flat 0B, on the same tap-off, is fine).
                above = next((a for a in path[i + 1:] if not kg.is_link(a)), None)
                if above and self._feeds_healthy_sensor(above, starved, local_pressure):
                    return span, above
                return span, None

        return span, None

    # ------------------------------------------------------------------

    def retrieve(self, incident, incident_id: str = "INC-1", events=None) -> EvidencePacket:
        """
        Args:
            incident: A detector ``Incident``.
            incident_id: Neutral label for the packet (never the scenario).
            events: The detector's point events (``AnomalyReport.events``).
                Optional; with them every alarm carries its hourly
                deviation and the true time it first went off-twin.
        """

        kg = self.kg
        cfg = self.config

        alarmed = self._alarms(incident)
        if not alarmed:
            raise ValueError("Incident has no candidates located in the building.")

        # Early readings: an asset can go off-twin alone, before enough
        # others agree to open an incident -- often the first sign of the
        # cause (a pump that should have run and did not).
        window_start = incident.detected_at_s - cfg.lookback_hours * 3600
        early_only = set()
        for event in events or []:
            if not (window_start <= event.timestamp_s < incident.detected_at_s):
                continue
            if event.asset_id not in kg or event.asset_id in alarmed and event.asset_id not in early_only:
                continue
            best = alarmed.get(event.asset_id)
            if best is None or event.score > best["score"]:
                first = best["first_seen_s"] if best else event.timestamp_s
                alarmed[event.asset_id] = {
                    "asset_id": event.asset_id,
                    "asset_type": event.asset_type,
                    "parameter": event.parameter,
                    "score": event.score,
                    "residual": event.residual,
                    "peak_at_s": event.timestamp_s,
                    "first_seen_s": min(first, event.timestamp_s),
                }
                early_only.add(event.asset_id)
            else:
                best["first_seen_s"] = min(best["first_seen_s"], event.timestamp_s)

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
        all_loss = [
            a for a, c in local.items()
            if c["parameter"] == "pressure" and c["residual"] < 0 and a not in tanks
        ]
        largest = max((abs(local[a]["residual"]) for a in all_loss), default=0.0)

        # A sensor that went dry is a major loss however small its normal
        # pressure: a top-floor flat has only ~5 m to lose, and must not be
        # ranked "minor" next to a ground-floor flat that lost 23 m.
        dry = {
            e.asset_id for e in events or []
            if e.asset_id in all_loss and e.parameter == "pressure"
            and e.expected > 2.0 and e.observed <= max(1.0, 0.25 * e.expected)
        }
        pressure_loss = [
            a for a in all_loss
            if a in dry or abs(local[a]["residual"]) >= cfg.minor_fraction * largest
        ]
        minor_loss = sorted(set(all_loss) - set(pressure_loss))

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
                "minor_pressure_loss_not_used_for_localisation": minor_loss,
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
            # root_anchor supplies every anchor, so it is on each path.
            path = kg.upstream(anchor)
            context |= set(path[: path.index(root_anchor) + 1])
            context |= kg.neighbourhood(anchor, cfg.hops)

        # The rest of an anchor's flat: "the other rooms of 3A are fine"
        # is what pins a fault to one room, and those rooms sit on the
        # flat's other tee, beyond a short BFS.
        flats = {kg.asset(a).get("flat") for a in anchors} - {None}
        context |= {a for a in kg.asset_ids() if kg.asset(a).get("flat") in flats}

        context |= set(common_mode)

        timelines = self._timelines(incident, events, alarmed, window_start)
        observations = [
            self._alarm(a, c, timelines.get(a), early=a in early_only)
            for a, c in local.items()
        ]

        # When each alarmed asset first went off-twin, earliest first. The
        # earliest anomaly is usually nearest the cause; later ones are
        # often its consequences (a tank running dry, pressures collapsing).
        first = {}
        for asset_id, candidate in alarmed.items():
            series = timelines.get(asset_id) or [(candidate["first_seen_s"], candidate["residual"])]
            when, residual = series[0][0], series[0][1]
            direction = "up" if residual > 0 else "down"
            quantity = "level" if kg.asset(asset_id)["role"] in ("roof_tank", "sump") else candidate["parameter"]
            first.setdefault(when, []).append(f"{asset_id} {quantity} {direction}")
        sequence = [f"{_clock(t)}: {', '.join(sorted(items))}" for t, items in sorted(first.items())]

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
                "sequence_of_first_alarms": sequence,
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

    @staticmethod
    def _timelines(incident, events, alarmed, window_start) -> dict:
        """
        ``{asset_id: [(timestamp_s, residual, observed, expected), ...]}``
        inside the incident window (plus look-back), from the detector's
        point events. Empty without events.
        """

        if not events:
            return {}

        wanted = {a: c["parameter"] for a, c in alarmed.items()}
        series = {}

        for event in events:
            if (
                wanted.get(event.asset_id) == event.parameter
                and window_start <= event.timestamp_s <= incident.last_seen_s
            ):
                series.setdefault(event.asset_id, []).append(
                    (event.timestamp_s, event.residual, event.observed, event.expected)
                )

        return {asset: sorted(points) for asset, points in series.items()}

    @staticmethod
    def _signed(parameter: str, residual: float) -> str:
        if parameter == "flowrate":
            return f"{residual * 1000:+.3f}"
        return f"{residual:+.2f}"

    def _alarm(self, asset_id: str, candidate: dict, timeline=None, early: bool = False) -> dict:
        sensors = self.kg.sensors_on(asset_id)
        parameter = candidate["parameter"]
        residual = candidate["residual"]
        unit = "L/s" if parameter == "flowrate" else "m"
        scale = 1000.0 if parameter == "flowrate" else 1.0
        peak_at = candidate.get("peak_at_s", candidate["first_seen_s"])
        amount = f"{abs(residual) * (1000 if parameter == 'flowrate' else 1):.{3 if parameter == 'flowrate' else 2}f} {unit}"

        record = {
            "sensor": f"sensor:{sensors[0]['id']}" if sensors else None,
            "asset": asset_id,
            "parameter": parameter,
            "status": "early alarm (before the incident opened)" if early else "alarm",
            "reading": f"peak {amount} {'above' if residual > 0 else 'below'} the twin at {_clock(peak_at)}",
            "score": round(float(candidate["score"]), 1),
            "first_seen": _clock(candidate["first_seen_s"]),
            # Numbers for machine consumers (rules, dashboards), in the
            # display unit: L/s for flow, m for pressure / tank level.
            "unit": unit,
            "peak_deviation": round(residual * scale, 4),
            "peak_at": _clock(peak_at),
        }

        if timeline:
            # Hourly signed deviation from the twin. The sign can flip
            # (a pump that delivered nothing, then over-ran to refill), and
            # the peak alone would hide which came first.
            points = [f"{_clock(t)} {self._signed(parameter, r)}" for t, r, _, _ in timeline]
            if len(points) > 8:
                points = points[:5] + ["..."] + points[-2:]
            record["hourly_deviation"] = f"{'; '.join(points)} ({unit} vs twin, alarmed hours only)"

            record["hourly"] = [
                {"at": _clock(t), "deviation": round(r * scale, 4),
                 "observed": round(o * scale, 4), "expected": round(e * scale, 4)}
                for t, r, o, e in timeline
            ]

            peak = next((p for p in timeline if p[0] == peak_at), None)
            if peak:
                # What the instrument actually read at the peak -- tells a
                # dry pipe (pressure ~0) from one merely running low.
                record["observed_at_peak"] = round(peak[2] * scale, 4)
                record["expected_at_peak"] = round(peak[3] * scale, 4)

        return record

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
