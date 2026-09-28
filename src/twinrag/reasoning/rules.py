"""
Rule-based root-cause diagnosis and alert messages.

The primary diagnosis path (the LLM in ``diagnosis.py`` is an optional
comparison). It reads exactly the evidence packet an LLM would receive,
returns the same diagnosis schema -- so the same grounding check and the
same scoring apply -- and adds a plain-language alert message.

Two physical principles decide everything:

**The cause is where the supply structure points.** The building is a
tree fed from one source, so

  * water escaping somewhere raises the flow in every pipe on the way
    there: a leak sits at or below the *deepest* pipe carrying extra flow,
    and is corroborated when the pipes above it carry the same extra flow;
  * a closure starves everything below it: a blockage sits above a group
    of flats that lost pressure (or rooms that lost flow), below the
    nearest point that still feeds a healthy sensor;
  * a pump that stops delivering starves the roof tank, then every flat.

**The cause comes first in time.** One fault has knock-on effects -- a
big leak drains the roof tank and shifts the pump's schedule, a blocked
riser changes the tank level. Every hypothesis carries the hour its key
evidence first appeared; among credible hypotheses the earliest wins and
the rest are reported as consequences.

Every step of the reasoning cites the packet IDs it rests on, so a
diagnosis is traceable to specific graph entities and sensor readings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .diagnosis import DiagnosisResult, check_grounding


#: A line pressure at or below this (m) means the pipe is effectively dry.
DRY_M = 1.0
#: A pressure loss at least this large (m) means supply is cut, not a wobble.
MAJOR_LOSS_M = 3.0
#: Upstream extra flow must be at least this fraction of the leaf's to
#: corroborate a leak (the same water, seen further up the path).
CORROBORATE = 0.5
#: A branch whose flow fell to this fraction of normal or less has stopped.
STOPPED = 0.2
#: Hypotheses below this confidence are too weak to be "the cause".
CREDIBLE = 0.5


def _hour(clock) -> int | None:
    if not clock or clock == "?":
        return None
    return int(str(clock).split(":")[0])


def _clock(hour) -> str:
    return f"{hour:02d}:00"


def _lph(litres_per_second: float) -> str:
    """Litres per hour, rounded for a human."""
    lph = abs(litres_per_second) * 3600
    return f"{lph:,.0f}" if lph >= 100 else f"{lph:.0f}"


@dataclass
class Hypothesis:
    fault_type: str
    kind: str
    root: str
    start_hour: int
    confidence: float
    affected: list
    steps: list                      # [(text, [ids])]
    message: dict = field(default_factory=dict)


class PacketView:
    """
    Read-only helpers over an evidence packet.
    """

    def __init__(self, packet: dict):
        self.packet = packet
        self.facts = packet.get("topology_facts", {})
        self.assets = {a["id"]: a for a in packet.get("assets", [])}

        self.parent, self.children = {}, {}
        for subject, relation, obj in packet.get("relations", []):
            if relation == "FEEDS":
                self.parent[obj] = subject
                self.children.setdefault(subject, []).append(obj)

        self.alarms, self.shift, self.normal = {}, None, {}
        for obs in packet.get("observations", []):
            status = obs.get("status", "")
            if status == "normal":
                self.normal[obs["asset"]] = obs
            elif obs.get("assets"):
                self.shift = obs
            elif obs.get("asset"):
                self.alarms[obs["asset"]] = obs

    # -- assets -----------------------------------------------------------

    def kind(self, asset_id) -> str:
        return self.assets.get(asset_id, {}).get("kind", "")

    def role(self, asset_id) -> str:
        return self.assets.get(asset_id, {}).get("role", "")

    def label(self, asset_id) -> str:
        return self.assets.get(asset_id, {}).get("label", asset_id)

    def location(self, asset_id) -> dict:
        return self.assets.get(asset_id, {}).get("location", {})

    def flat(self, asset_id) -> str | None:
        return self.location(asset_id).get("flat")

    def place(self, asset_id) -> str:
        loc = self.location(asset_id)
        parts = [loc.get("floor"), loc.get("flat"), loc.get("room")]
        return ", ".join(p for p in parts if p) or "building supply"

    def is_link(self, asset_id) -> bool:
        return self.kind(asset_id) in ("pipe", "pump", "valve")

    def ancestors(self, asset_id) -> list:
        out, current = [], asset_id
        while current in self.parent:
            current = self.parent[current]
            out.append(current)
        return out

    def descendants(self, asset_id) -> set:
        seen, stack = set(), list(self.children.get(asset_id, []))
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(self.children.get(node, []))
        return seen

    def common_ancestor(self, asset_ids) -> str | None:
        paths = [[a] + self.ancestors(a) for a in asset_ids]
        if not paths:
            return None
        shared = set(paths[0])
        for path in paths[1:]:
            shared &= set(path)
        for asset in paths[0]:          # deepest first
            if asset in shared:
                return asset
        return None

    # -- readings ---------------------------------------------------------

    def first_hour(self, asset_id, sign=None) -> int | None:
        """
        First hour the asset was off-twin, optionally only counting
        deviations of one sign (-1 below the twin, +1 above).
        """

        obs = self.alarms.get(asset_id)
        if not obs:
            return None
        hourly = obs.get("hourly")
        if hourly:
            hours = [
                _hour(h["at"]) for h in hourly
                if sign is None or (h["deviation"] > 0) == (sign > 0)
            ]
            return min(hours) if hours else None
        if sign is not None and (obs.get("peak_deviation", 0) > 0) != (sign > 0):
            return None
        return _hour(obs.get("first_seen"))

    def deviation(self, asset_id) -> float:
        return float(self.alarms.get(asset_id, {}).get("peak_deviation", 0.0))

    def extreme(self, asset_id, sign) -> dict | None:
        """The hourly reading furthest from the twin in one direction."""
        obs = self.alarms.get(asset_id, {})
        hourly = [h for h in obs.get("hourly", []) if (h["deviation"] > 0) == (sign > 0)]
        if hourly:
            return max(hourly, key=lambda h: abs(h["deviation"]))
        if obs and (obs.get("peak_deviation", 0) > 0) == (sign > 0):
            return {"at": obs.get("peak_at"), "deviation": obs["peak_deviation"],
                    "observed": obs.get("observed_at_peak"), "expected": obs.get("expected_at_peak")}
        return None

    def sensor(self, asset_id) -> str | None:
        return (self.alarms.get(asset_id) or self.normal.get(asset_id) or {}).get("sensor")

    def flats_text(self, asset_ids) -> str:
        flats = sorted({self.flat(a) for a in asset_ids if self.flat(a)},
                       key=lambda f: (f.split()[-1][:-1], f.split()[-1][-1]))
        names = [f.split()[-1] for f in flats]
        if not names:
            return ""
        return ("Flat " if len(names) == 1 else "Flats ") + ", ".join(names)


# ----------------------------------------------------------------------
# Hypothesis generators
# ----------------------------------------------------------------------

def _leak_hypotheses(view: PacketView) -> list:
    extra = view.facts.get("extra_flow", {})
    out = []

    for leaf in extra.get("deepest_assets_with_extra_flow", []):
        # A pump carrying extra flow is running more (refilling), not leaking.
        if view.kind(leaf) == "pump" or leaf not in view.alarms:
            continue

        amount = view.deviation(leaf)
        if amount <= 0:
            continue

        corroborating = [
            a for a in view.ancestors(leaf)
            if a in view.alarms and view.alarms[a]["parameter"] == "flowrate"
            and view.deviation(a) >= CORROBORATE * amount
            and view.kind(a) != "pump"
        ]
        taps = [c for c in view.children.get(leaf, []) if view.role(c) == "fixture"]
        root = taps[0] if taps else leaf
        start = view.first_hour(leaf, sign=+1) or view.first_hour(leaf)

        confidence = 0.6 + (0.25 if corroborating else 0.0)
        if taps:
            confidence += 0.05

        flat = view.flat(leaf)
        siblings = [
            a for a in view.normal
            if view.role(a) == "room_branch" and view.flat(a) == flat and a != leaf
        ]

        steps = [(
            f"From {_clock(start)} the {view.label(leaf)} carried up to {amount:.3f} L/s "
            f"(about {_lph(amount)} litres per hour) more than the digital twin predicts.",
            [x for x in (leaf, view.sensor(leaf)) if x],
        )]
        if corroborating:
            steps.append((
                "The same extra flow appears further up its supply path ("
                + ", ".join(view.label(a) for a in corroborating)
                + "), so extra water is being drawn at the end of this path.",
                corroborating,
            ))
        steps.append((
            f"No pipe below it carries extra flow, so the water leaves the system at or after "
            f"the {view.label(leaf)}" + (f", i.e. at the {view.label(root)}." if taps else "."),
            [leaf] + taps,
        ))
        if siblings:
            steps.append((
                f"The other rooms of {flat} read normal, which pins it to this one room.",
                sorted(siblings),
            ))

        out.append(Hypothesis(
            fault_type="leak", kind="leak", root=root, start_hour=start,
            confidence=min(confidence, 0.95),
            affected=[root, leaf] + corroborating, steps=steps,
            message={"amount_lps": amount, "leaf": leaf, "tap": root if taps else None},
        ))

    return out


def _pressure_blockage(view: PacketView) -> list:
    loss = view.facts.get("pressure_loss")
    if not loss:
        return []

    affected = loss.get("assets_with_pressure_loss", [])
    span = loss.get("suspect_span_upward", [])
    bound = loss.get("nearest_upstream_point_still_feeding_a_healthy_sensor")
    common = loss.get("common_supply_point")
    if not affected or not span:
        return []

    links = [a for a in span if view.is_link(a)]
    root = links[0] if links else span[0]

    drops = [abs(view.deviation(a)) for a in affected if a in view.alarms]
    worst = max(drops, default=0.0)
    dry = [
        a for a in affected
        if (view.extreme(a, -1) or {}).get("observed") is not None
        and view.extreme(a, -1)["observed"] <= DRY_M
    ]
    starts = [h for h in (view.first_hour(a, sign=-1) for a in affected) if h is not None]
    start = min(starts) if starts else _hour(view.packet["incident"].get("first_alarm"))

    confidence = 0.55 + (0.2 if bound else 0.0) + (0.15 if dry else 0.0)
    if worst < MAJOR_LOSS_M and not dry:
        confidence = 0.3                # a wobble, most likely a knock-on

    where = view.flats_text(affected)
    one = len({view.flat(a) for a in affected}) == 1
    steps = [(
        f"From {_clock(start)} the pressure at {where} fell by up to {worst:.1f} m"
        + (f"; {'it reads' if one else view.flats_text(dry) + ' read'} about zero -- no water." if dry else "."),
        sorted(affected),
    )]
    steps.append((
        ("It is" if one else "All of them are") + f" supplied through the {view.label(common)}.",
        [common],
    ))
    if bound:
        healthy = [a for a in view.normal if view.role(a) == "flat_inlet" and bound in view.ancestors(a)]
        steps.append((
            f"The {view.label(bound)} still feeds sensors that read normal"
            + (f" ({view.flats_text(healthy)})" if healthy else "")
            + f", so the restriction lies between it and the {view.label(common)}: "
              f"the {view.label(root)}.",
            [bound, root] + sorted(healthy)[:4],
        ))
    else:
        steps.append((
            f"The first suspect above them is the {view.label(root)}.",
            [root],
        ))

    # A loss spanning the whole building (no healthy flat anywhere below
    # the roof) is either a closure at the roof tank's outlet or the tank
    # itself running out. A closure stops water leaving the tank at once;
    # a draining tank keeps delivering until it is empty. If the outlet
    # flow did not drop at the start, it is not a closed pipe.
    if not bound and view.role(common) in ("roof_manifold", "roof_tank", "pump_delivery"):
        outlet = next((a for a in view.alarms if view.role(a) == "tank_outlet"), None)
        outlet_down = view.first_hour(outlet, sign=-1) if outlet else None
        if outlet_down is None or outlet_down > start:
            confidence = min(confidence, 0.4)
            steps.append((
                "Water kept leaving the roof tank normally when the pressure loss began"
                + (f" (its outlet flow only fell at {_clock(outlet_down)})" if outlet_down else "")
                + ", so the roof tank was running low rather than a pipe being closed.",
                [x for x in (outlet, common) if x],
            ))

    return [Hypothesis(
        fault_type="blockage", kind="no_water", root=root, start_hour=start,
        confidence=min(confidence, 0.95), affected=sorted(set(affected) | {root}),
        steps=steps, message={"worst_m": worst, "dry": dry, "affected": affected, "bound": bound},
    )]


def _flow_blockage(view: PacketView) -> list:
    """Room branches whose flow stopped, with no pressure sensor below them."""

    stopped = []
    for asset, obs in view.alarms.items():
        if obs["parameter"] != "flowrate" or view.role(asset) != "room_branch":
            continue
        low = view.extreme(asset, -1)
        if low and low.get("expected") and low.get("observed") is not None \
                and low["expected"] > 0 and low["observed"] <= STOPPED * low["expected"]:
            stopped.append(asset)

    if not stopped:
        return []

    common = view.common_ancestor(stopped)
    if common is None:
        return []
    root = common if view.is_link(common) else view.parent.get(common, common)
    starts = [h for h in (view.first_hour(a, sign=-1) for a in stopped) if h is not None]
    start = min(starts) if starts else _hour(view.packet["incident"].get("first_alarm"))

    steps = [
        (f"From {_clock(start)} the flow through " + ", ".join(view.label(a) for a in stopped)
         + " stopped almost completely while the twin expected normal use.", sorted(stopped)),
        (f"They share one supply: the {view.label(root)}.", [root]),
    ]

    return [Hypothesis(
        fault_type="blockage", kind="no_flow", root=root, start_hour=start,
        confidence=0.6 if len(stopped) > 1 else 0.5,
        affected=sorted(set(stopped) | {root}), steps=steps,
        message={"stopped": stopped},
    )]


def _pump_failure(view: PacketView) -> list:
    out = []
    tank_low = {
        tank: text for tank, text in view.facts.get("tank_level_change", {}).items() if "below" in text
    }
    loss = view.facts.get("pressure_loss", {})
    broad = bool(view.shift) or (
        len({view.flat(a) for a in loss.get("assets_with_pressure_loss", [])}) >= 4
    )

    for pump in [a for a in view.alarms if view.kind(a) == "pump"]:
        low = view.extreme(pump, -1)
        start = view.first_hour(pump, sign=-1)
        if low is None or start is None:
            continue

        confidence = 0.5 + (0.2 if tank_low else 0.0) + (0.1 if broad else 0.0)

        # A failed pump starves the tank for hours. A single hour of "pump
        # short, tank low" is also what a normal shift in the pump's
        # on/off cycle looks like (after a day of lower water use, say), so
        # it is not enough to confirm a failure.
        tank_hours = max(
            (sum(1 for h in view.alarms[t].get("hourly", []) if h["deviation"] < 0) for t in tank_low),
            default=0,
        )
        incident = view.packet["incident"]
        span = (_hour(incident.get("last_alarm")) or 0) - (_hour(incident.get("first_alarm")) or 0) + 1
        sustained = tank_hours >= 2 or (broad and span >= 2)
        if not sustained:
            confidence = 0.35

        steps = [(
            f"At {_clock(start)} the {view.label(pump)} delivered "
            f"{abs(low['deviation']):.2f} L/s less than the twin expects -- it should have been running.",
            [x for x in (pump, view.sensor(pump)) if x],
        )]
        for tank, text in tank_low.items():
            steps.append((
                f"The {view.label(tank)} then fell {text} ({_clock(view.first_hour(tank) or start)}), "
                "because nothing was refilling it.",
                [tank],
            ))
        if broad:
            flats = sorted({view.flat(a) for a in loss.get("assets_with_pressure_loss", []) if view.flat(a)})
            steps.append((
                "Pressure then dropped across the building"
                + (f" ({len(flats)} flats)" if flats else "")
                + " -- a loss of supply head from the roof tank, not a local fault.",
                sorted(loss.get("assets_with_pressure_loss", []))[:6] or [pump],
            ))
        upstream_ok = [a for a in view.normal if a in view.ancestors(pump)]
        if upstream_ok:
            steps.append((
                "The supply to the pump (" + ", ".join(view.label(a) for a in upstream_ok)
                + ") read normal, so water was available to pump.",
                upstream_ok,
            ))
        high = view.extreme(pump, +1)
        if high and _hour(high["at"]) and _hour(high["at"]) > start:
            steps.append((
                f"At {high['at']} it ran {high['deviation']:.2f} L/s above normal: the refill once it "
                "recovered -- a consequence, not a second fault.",
                [pump],
            ))
        if not sustained:
            steps.append((
                f"The evidence lasts only {span} hour{'s' if span != 1 else ''}; a normal shift in the "
                "pump's on/off cycle looks the same, so a pump failure is not confirmed.",
                [pump],
            ))

        out.append(Hypothesis(
            fault_type="pump_failure", kind="pump", root=pump, start_hour=start,
            confidence=min(confidence, 0.95), affected=[pump] + list(tank_low), steps=steps,
            message={"tank_low": tank_low, "sustained": sustained},
        ))

    return out


def _street_supply(view: PacketView) -> list:
    sump = next((a for a in view.alarms if view.role(a) == "sump" and view.deviation(a) < 0), None)
    inlet = next((a for a in view.alarms if view.role(a) == "municipal_inlet" and view.deviation(a) < 0), None)
    if not (sump and inlet):
        return []
    start = min(h for h in (view.first_hour(inlet, -1), view.first_hour(sump, -1)) if h is not None)
    return [Hypothesis(
        fault_type="other", kind="street_supply", root=inlet, start_hour=start, confidence=0.7,
        affected=[inlet, sump],
        steps=[(f"From {_clock(start)} the {view.label(inlet)} delivered less than normal and the "
                f"{view.label(sump)} fell below its expected level.", [inlet, sump])],
    )]


# ----------------------------------------------------------------------
# Choice and message
# ----------------------------------------------------------------------

ACTIONS = {
    "leak": "Inspect the {room}: taps, WC cistern and flush valve, and concealed pipe joints. "
            "If nothing is visible, close {flat}'s inlet valve and watch the meter to confirm.",
    "no_water": "Check the {root_label} and its isolation valve; residents of {flats} have no water until it is cleared.",
    "no_flow": "Check the {root_label} for a closed valve or choke.",
    "pump": "Check the transfer pump: power supply, starter or overload trip, and the roof-tank level switch. "
            "The roof tank is draining; upper floors lose water first.",
    "street_supply": "Check the municipal connection and the sump's float valve; contact the water utility if the main is off.",
}


def _message(best: Hypothesis, view: PacketView, alternatives: list) -> dict:
    m = best.message

    if best.kind == "leak":
        amount = m["amount_lps"]
        room = view.location(m["leaf"]).get("room") or view.label(m["leaf"])
        flat = view.flat(m["leaf"]) or "the flat"
        severity = "critical" if amount >= 0.1 else "warning"
        title = f"Possible leak: {flat}, {room.lower()} ({view.location(m['leaf']).get('floor', '')})"
        summary = (
            f"About {_lph(amount)} litres per hour more water than normal is flowing to the "
            f"{room.lower()} of {flat} since {_clock(best.start_hour)}. "
            + ("It is likely a burst or a fully open fixture." if amount >= 0.1
               else "It is likely a running cistern, dripping tap or small pipe leak.")
        )
        action = ACTIONS["leak"].format(room=f"{room.lower()} of {flat}", flat=flat)
    elif best.kind == "no_water":
        flats = view.flats_text(m["affected"])
        severity = "critical"
        title = f"No water: {flats}" if m["dry"] else f"Pressure loss: {flats}"
        summary = (
            f"Since {_clock(best.start_hour)} {flats} "
            + (("has" if flats.startswith("Flat ") else "have")
               + (" lost water" if m["dry"] else f" lost up to {m['worst_m']:.1f} m of pressure"))
            + f". The likely cause is a blockage or closed valve in the {view.label(best.root)}."
        )
        action = ACTIONS["no_water"].format(root_label=view.label(best.root), flats=flats)
    elif best.kind == "no_flow":
        severity = "warning"
        title = f"Flow stopped: {view.flats_text(m['stopped'])}"
        summary = (f"Since {_clock(best.start_hour)} no water is reaching "
                   + ", ".join(view.label(a) for a in m["stopped"])
                   + f". The likely cause is a blockage in the {view.label(best.root)}.")
        action = ACTIONS["no_flow"].format(root_label=view.label(best.root))
    elif best.kind == "pump":
        severity = "critical"
        title = "Roof tank not refilling: transfer pump"
        summary = (f"At {_clock(best.start_hour)} the transfer pump did not deliver water when it normally runs"
                   + ("; the roof tank has dropped and pressure is falling across the building."
                      if m["tank_low"] else "."))
        action = ACTIONS["pump"]
    else:
        severity = "critical"
        title = "Street water supply interrupted"
        summary = f"Since {_clock(best.start_hour)} the municipal inlet is delivering less water and the sump is falling."
        action = ACTIONS["street_supply"]

    effect = {
        "leak": "extra flow",
        "no_water": "pressure loss",
        "no_flow": "flow stopped",
        "pump": "running off its normal schedule",
        "street_supply": "street supply change",
    }
    knock_on = [
        f"{view.label(a.root)}: {effect.get(a.kind, a.kind)} from {_clock(a.start_hour)}"
        for a in alternatives if a.start_hour >= best.start_hour
    ]

    if best.confidence < CREDIBLE:
        # Too weak to act on as a diagnosis: say so rather than guess.
        severity = "review"
        title = "Unconfirmed: " + title[0].lower() + title[1:]
        summary += (" The evidence is too brief or too weak to confirm this; "
                    "keep monitoring and check again if it persists.")

    return {
        "severity": severity,
        "title": title,
        "summary": summary,
        "evidence": [text for text, _ in best.steps],
        "consequences": knock_on,
        "action": action,
        "confidence": round(best.confidence, 2),
    }


class RuleBasedDiagnoser:
    """
    Deterministic, grounded diagnosis of an evidence packet.
    """

    name = "rules"

    def diagnose(self, packet: dict) -> DiagnosisResult:
        view = PacketView(packet)

        hypotheses = (
            _leak_hypotheses(view) + _pressure_blockage(view) + _flow_blockage(view)
            + _pump_failure(view) + _street_supply(view)
        )

        if not hypotheses:
            return DiagnosisResult(
                diagnosis=None, grounded=False, attempts=1,
                error="No rule matched the evidence; the incident needs manual review.",
            )

        credible = [h for h in hypotheses if h.confidence >= CREDIBLE] or hypotheses
        credible.sort(key=lambda h: (h.start_hour, -h.confidence))
        best = credible[0]

        # Two credible causes starting in the same hour: say so.
        rivals = [h for h in credible[1:] if h.start_hour == best.start_hour and h.root != best.root]
        confidence = best.confidence - (0.1 if rivals else 0.0)
        best.confidence = max(confidence, 0.1)

        alternatives = [h for h in hypotheses if h is not best]

        diagnosis = {
            "root_cause_asset": best.root,
            "fault_type": best.fault_type,
            "location": view.place(best.root),
            "affected_assets": sorted(set(best.affected)),
            "reasoning": [{"step": text, "evidence": ids} for text, ids in best.steps],
            "alternatives": [
                {
                    "asset": h.root,
                    "why_less_likely": (
                        f"its signs start at {_clock(h.start_hour)}, after the cause ({_clock(best.start_hour)}), "
                        "consistent with a knock-on effect"
                        if h.start_hour > best.start_hour
                        else f"weaker evidence (confidence {h.confidence:.2f} vs {best.confidence:.2f})"
                    ),
                }
                for h in sorted(alternatives, key=lambda h: (h.start_hour, -h.confidence))[:4]
            ],
            "confidence": round(best.confidence, 2),
            "recommended_action": "",
            "message": _message(best, view, alternatives),
        }
        diagnosis["recommended_action"] = diagnosis["message"]["action"]

        violations = check_grounding(diagnosis, packet["allowed_ids"])
        return DiagnosisResult(
            diagnosis=diagnosis, grounded=not violations, violations=violations, attempts=1,
            raw="",
        )


def format_message(message: dict) -> str:
    """Plain-text rendering of a diagnosis message, for logs and consoles."""

    lines = [f"[{message['severity'].upper()}] {message['title']}", message["summary"], "", "Why:"]
    lines += [f"  - {e}" for e in message["evidence"]]
    if message.get("consequences"):
        lines += ["Knock-on effects (not the cause):"] + [f"  - {c}" for c in message["consequences"]]
    lines += ["", f"Action: {message['action']}", f"Confidence: {message['confidence']:.0%}"]
    return "\n".join(lines)
