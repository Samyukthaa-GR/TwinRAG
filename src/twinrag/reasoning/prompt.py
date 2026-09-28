"""
Prompts for graph-constrained root-cause reasoning.

The system prompt carries *domain knowledge* -- how a gravity-fed
building behaves and what each retrieval fact means -- and the hard rules
of grounding. It never carries anything about a particular scenario; that
arrives only through the evidence packet.
"""

import json

FAULT_TYPES = ("leak", "blockage", "pump_failure", "other")

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause_asset": {"type": "string"},
        "fault_type": {"type": "string", "enum": list(FAULT_TYPES)},
        "location": {"type": "string"},
        "affected_assets": {"type": "array", "items": {"type": "string"}},
        "reasoning": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["step", "evidence"],
            },
        },
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "why_less_likely": {"type": "string"},
                },
                "required": ["asset", "why_less_likely"],
            },
        },
        "confidence": {"type": "number"},
        "recommended_action": {"type": "string"},
    },
    "required": [
        "root_cause_asset", "fault_type", "location", "affected_assets",
        "reasoning", "confidence", "recommended_action",
    ],
}

SYSTEM_PROMPT = """You are a water-supply diagnostics engineer for a multi-storey residential building. \
An anomaly detector compared every sensor with a digital twin of the building and raised an incident. \
You receive an EVIDENCE PACKET extracted from the building's knowledge graph. Find the single root cause.

How this building works
- Supply: street main -> underground sump -> transfer pump (PUMP1) -> rising main -> roof tank (OHT) -> \
roof manifold -> one gravity down-take riser per flat stack (DTA-*, DTB-*) -> each flat's supply pipe -> \
flat inlet -> distribution tees -> one branch pipe per wet room -> the room's taps.
- The supply is a tree: every asset has exactly one supply path. Relation FEEDS points in the supply direction.
- IDs read as locations: F3A = floor 3 (F0 = ground), flat A; F3A-KIT = its kitchen plumbing point; \
F3A-KIT-BR = the branch pipe feeding it; DTA-5-4 = riser A between floors 5 and 4.

What the topology facts mean (they are exact graph computations, not guesses)
- extra_flow: water beyond the twin's prediction is drawn at or below these assets. Water escaping somewhere \
raises the flow in every pipe on the way there, so the deepest pipe with extra flow is the specific one.
- pressure_loss: every asset listed lost pressure and all are supplied through common_supply_point. A closure in \
suspect_span_upward would starve all of them; the named upstream point still feeds a healthy sensor, so the \
restriction is below it.
- building_wide_pressure_shift: most flats' pressure moved together -- a building-wide effect (for example the \
roof tank level changing), not a local fault.
- low_tank_supply_chain: a tank is below its expected level; these are the assets supplying it.
- One fault often has knock-on effects (a big leak drains the roof tank; a blocked riser changes tank level). \
Decide which fact is the cause and which are consequences.
- Sensors listed as normal agreed with the twin: negative evidence.
- Time order matters. The SEQUENCE shows when each asset first went off-twin; the earliest anomaly is usually nearest the cause, and later ones are often its consequences. A sign flip in an hourly deviation (below the twin, then above) means the behaviour changed during the incident -- read the whole series, not just the peak.

Fault types: leak (water escaping at or below an asset), blockage (a closed or choked pipe or valve), \
pump_failure (the transfer pump not delivering), other.

Rules
1. Use only the evidence packet. Do not assume sensors, pipes or readings that are not in it.
2. Every asset you name -- root_cause_asset, affected_assets, reasoning evidence, alternatives -- must be an \
exact ID that appears in the evidence packet. Never invent or modify an ID.
3. root_cause_asset is one asset: for a leak, the plumbing point or pipe where water escapes; for a blockage, \
the pipe that is closed; for a pump failure, the pump.
4. reasoning: 2-6 short steps; each step cites the IDs it relies on.
5. confidence is between 0 and 1 and should be lower when the facts are ambiguous.
6. Reply with one JSON object only, matching this shape:
{"root_cause_asset": "...", "fault_type": "leak|blockage|pump_failure|other", "location": "floor/flat/room in words", \
"affected_assets": ["..."], "reasoning": [{"step": "...", "evidence": ["..."]}], \
"alternatives": [{"asset": "...", "why_less_likely": "..."}], "confidence": 0.0, "recommended_action": "..."}"""


def _place(location: dict) -> str:
    parts = (location.get("floor"), location.get("flat"), location.get("room"))
    return ", ".join(p for p in parts if p) or "building services"


def _incident_line(packet: dict) -> str:
    inc = packet["incident"]
    return (
        f"INCIDENT {inc.get('id', '?')}: alarms from {inc.get('first_alarm', '?')} to "
        f"{inc.get('last_alarm', '?')}, {inc.get('alarmed_sensor_count', '?')} sensors alarmed. "
        f"{inc.get('method', '')}"
    ).strip()


def _building_line(packet: dict) -> str:
    b = packet["building"]
    return (
        f"BUILDING: {b.get('name', '?')}, {b.get('floors', '?')} floors x "
        f"{b.get('flats_per_floor', '?')} flats. Supply: {b.get('supply_scheme', '?')}."
    )


def render_packet(packet: dict) -> str:
    """
    Compact text rendering of an evidence packet for the prompt.

    Pretty JSON spent a large share of tokens on punctuation and repeated
    keys, and free-tier limits count prompt + answer against ~8k tokens a
    minute. Every ID stays. ``LOCATED_IN``/``PART_OF``/``MONITORS`` triples
    are dropped because each asset line already states its place and each
    alarm its sensor. Grounding is still checked against the packet's full
    ``allowed_ids``.
    """

    lines = [_incident_line(packet), _building_line(packet)]
    sequence = packet["incident"].get("sequence_of_first_alarms")
    if sequence:
        lines += ["", "SEQUENCE (when each alarmed asset first went off-twin)"] + [f"- {s}" for s in sequence]
    lines += ["", "ALARMS"]

    if "alarms" in packet:          # the no-graph ablation
        for alarm in packet["alarms"]:
            asset = alarm["asset"] if isinstance(alarm["asset"], str) else ", ".join(alarm["asset"])
            label = f" ({alarm['label']})" if alarm.get("label") else ""
            lines.append(f"- {asset}{label}, {alarm['parameter']}: {alarm['reading']}")
        return "\n".join(lines)

    normal = []
    for obs in packet["observations"]:
        if obs["status"] == "normal":
            normal.append(f"{obs['asset']} ({obs['parameter']})")
        elif obs.get("assets"):
            lines.append(
                f"- building-wide {obs['parameter']} shift at {', '.join(obs['assets'])}: {obs['reading']}"
            )
        else:
            hourly = f"; hourly {obs['hourly_deviation']}" if obs.get("hourly_deviation") else ""
            lines.append(
                f"- {'EARLY ' if obs['status'].startswith('early') else ''}"
                f"{obs['asset']} {obs['parameter']}: {obs['reading']}; score {obs.get('score')}; "
                f"first off-twin {obs.get('first_seen')}{hourly} [{obs.get('sensor')}]"
            )

    lines += ["", "SENSORS THAT STAYED NORMAL", ", ".join(normal) or "none in this subgraph"]

    lines += ["", "ASSETS (id | description | place | metered?)"]
    for asset in packet["assets"]:
        size = f", {asset['diameter_mm']} mm" if asset.get("diameter_mm") else ""
        lines.append(
            f"- {asset['id']} | {asset['label']}{size} | {_place(asset.get('location', {}))} | "
            f"{'metered' if asset.get('metered') else 'unmetered'}"
        )

    feeds = [f"{s}->{o}" for s, r, o in packet["relations"] if r == "FEEDS"]
    lines += ["", "SUPPLY (A->B: water flows from A to B)", "; ".join(feeds)]

    lines += ["", "TOPOLOGY FACTS", json.dumps(packet["topology_facts"], separators=(",", ":"))]
    return "\n".join(lines)


def user_message(packet: dict) -> str:
    return "EVIDENCE PACKET\n" + render_packet(packet)


def ungrounded_packet(packet: dict) -> dict:
    """
    The no-graph ablation: only what an alarm screen shows -- alarmed
    sensors with their asset, label and reading. No relations, no
    topology facts, no normal sensors, no asset list beyond the alarms.
    """

    labels = {a["id"]: a for a in packet["assets"]}
    alarms = []
    allowed = set()

    for obs in packet["observations"]:
        if not obs["status"].startswith("alarm"):
            continue
        assets = obs.get("assets") or [obs["asset"]]
        for asset in assets:
            allowed.add(asset)
        alarms.append({
            "asset": obs.get("asset") or assets,
            "label": labels.get(obs.get("asset"), {}).get("label"),
            "parameter": obs["parameter"],
            "reading": obs["reading"],
        })

    return {
        "incident": packet["incident"],
        "building": {k: v for k, v in packet["building"].items() if k != "relation_meanings"},
        "alarms": alarms,
        "allowed_ids": sorted(allowed),
    }
