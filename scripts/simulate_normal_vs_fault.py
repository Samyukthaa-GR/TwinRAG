"""
Replay one simulated day hour by hour in the terminal, like a live monitor.

    python scripts/simulate_normal_vs_fault.py                  # kitchen leak (1st fault in the config)
    python scripts/simulate_normal_vs_fault.py --fault 3        # riser blockage
    python scripts/simulate_normal_vs_fault.py --type leak --target F5B-BATH2 \
        --severity 0.4 --start 14 --end 20
    python scripts/simulate_normal_vs_fault.py --normal-first   # replay a normal day before the fault day
    python scripts/simulate_normal_vs_fault.py --delay 0.3      # faster (seconds per hour)

The day is simulated first (EPANET, a few seconds), then printed one hour
at a time, 01:00 -> 24:00:

  - every hour: the building's key readings next to the digital twin's
    normal value, and how many of the 103 sensors read off-normal;
  - [SIM] lines: when the simulator switches the fault on and off
    (the answer key -- shown because this is a demo; --hide-answer hides it);
  - when sensors go off-normal: which ones, by how much (z = |reading - normal| / sensor noise);
  - an ALERT the hour the system raises it, using only the readings up to
    that hour (detect -> knowledge-graph retrieval -> rule-based diagnosis).

Sensor readings include simulated instrument noise, exactly what the
detector sees. Nothing is written under data/.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# --------------------------------------------------
# TwinRAG imports
# --------------------------------------------------

from twinrag.building import load_layout
from twinrag.building.pipeline import BuildingDiagnosisPipeline
from twinrag.building.sensors import building_sensor_model
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.reasoning.rules import format_message
from twinrag.simulation.config import load_config
from twinrag.simulation.faults import create_fault
from twinrag.simulation.network_loader import WaterNetworkLoader
from twinrag.simulation.scenario import FaultScenarioRunner


LAYOUT = PROJECT_ROOT / "data" / "networks" / "Building_G6.layout.json"

#: Parameters the building needs that the fault classes do not default to.
DEFAULT_PARAMS = {
    "leak": {"max_leak_demand": 0.0002},              # severity 1.0 = 0.2 L/s burst
    "pump_failure": {"restore_after_outage": False},  # level switches resume after
}

THRESHOLD = 4.0     # the detector's z threshold
MIN_ASSETS = 2      # sensors that must disagree together to open an incident
SHOW = 6            # off-normal sensors listed per hour

#: Readings printed every hour: (sensor id, short name).
KEY_SENSORS = [
    ("FM-PUMP1", "pump"),
    ("PT-OHT", "roof tank"),
    ("FM-OHT-OUTLET", "tank outflow"),
    ("PT-F0A-IN", "floor 0 pressure"),
    ("PT-F6A-IN", "floor 6 pressure"),
]


class Style:
    def __init__(self, enabled):
        on = enabled and sys.stdout.isatty() or enabled == "force"
        if on and os.name == "nt":
            os.system("")   # enables ANSI colours in the Windows console
        code = lambda c: (lambda s: f"\033[{c}m{s}\033[0m") if on else (lambda s: s)
        self.green, self.yellow, self.red = code("32"), code("33"), code("1;31")
        self.cyan, self.dim, self.bold = code("36"), code("2"), code("1")


def readings(dataset, layout, seed):
    """
    What every sensor reads each hour, as
    {sensor id: {hour: (reading with instrument noise, noise std, true value)}}.
    """

    keys = {(s["asset_id"], s["parameter"]): s["id"] for s in layout["sensors"]}
    noisy = building_sensor_model(layout, seed=seed).observe(dataset)
    out = {}
    for row in noisy.itertuples(index=False):
        sid = keys.get((row.asset_id, row.parameter))
        if sid:
            out.setdefault(sid, {})[int(row.timestamp_s) // 3600] = (row.value, row.noise_std, row.truth)
    return out


def twin(baseline, layout):
    """
    The digital twin's normal value for every sensor and hour (noise-free).
    """

    keys = {(s["asset_id"], s["parameter"]): s["id"] for s in layout["sensors"]}
    out = {}
    for row in baseline.itertuples(index=False):
        sid = keys.get((row.asset_id, row.parameter))
        if sid:
            out.setdefault(sid, {})[int(row.timestamp_s) // 3600] = row.value
    return out


def fmt(sensor, value):
    # Flow meters are stored in m3/s and shown in L/s; pressure and level in m.
    if sensor.startswith("FM-"):
        return f"{value * 1000 + 0.0:.3f} L/s"
    return f"{value + 0.0:.2f} m"


def replay_day(title, dataset, baseline_twin, layout, labels, live, fault, args, st):
    observed = readings(dataset, layout, args.seed)
    hours = [h for h in range(1, 25) if h in observed["FM-PUMP1"]]

    print(st.bold("=" * 100))
    print(st.bold(title))
    print(st.bold("=" * 100))

    seen = {}   # incident id -> (root, confidence, severity) last printed
    for hour in hours:
        if fault and not args.hide_answer:
            if hour == fault.start_hour:
                print(st.cyan(f"[SIM]   {hour:02d}:00  fault switched ON : {fault.fault_type} at "
                              f"{fault.target_id} ({labels.get(fault.target_id, '')})"))
            if fault.end_hour is not None and hour == fault.end_hour:
                print(st.cyan(f"[SIM]   {hour:02d}:00  fault switched OFF: {fault.target_id} restored"))

        off = []
        for sid, series in observed.items():
            reading, noise, _ = series[hour]
            expected = baseline_twin[sid][hour]
            z = abs(reading - expected) / noise
            if z >= THRESHOLD:
                off.append((z, sid, expected, reading))
        off.sort(reverse=True)

        # Headline values are the simulated hydraulics (noise-free); the "!"
        # lines below are noisy sensor readings, which is what the detector uses.
        key = "  ".join(f"{name} {fmt(sid, observed[sid][hour][2])}" for sid, name in KEY_SENSORS)
        if not off:
            status = st.green("NORMAL ")
            note = st.dim(f"0/{len(observed)} sensors off-normal")
        else:
            status = st.yellow("ANOMALY") if len(off) >= MIN_ASSETS else st.green("NORMAL ")
            note = (st.yellow(f"{len(off)}/{len(observed)} sensors off-normal") if len(off) >= MIN_ASSETS
                    else st.dim(f"1/{len(observed)} sensor off-normal (one sensor is not enough to alarm)"))
        print(f"[{hour:02d}:00] {status} | {key} | {note}")

        if len(off) >= MIN_ASSETS:
            for z, sid, expected, reading in off[:SHOW]:
                print(st.yellow(f"           ! {sid:<15} {labels.get(sid, '')[:38]:<38} "
                                f"normal {fmt(sid, expected):>11}  now {fmt(sid, reading):>11}  z={z:6.1f}"))
            if len(off) > SHOW:
                print(st.yellow(f"           ! ... and {len(off) - SHOW} more"))

        for inc in live.get(hour, []):
            d = inc["diagnosis"]
            msg = (d or {}).get("message") or {}
            state = (d or {}).get("root_cause_asset"), (d or {}).get("confidence"), msg.get("severity")
            if seen.get(inc["id"]) == state:
                continue
            first = inc["id"] not in seen
            seen[inc["id"]] = state
            head = "ALERT " if first else "UPDATE"
            print(st.red(f"[{head}] {hour:02d}:00  {inc['id']} raised from readings up to {hour:02d}:00 "
                         f"(alarms since {inc['first_alarm_hour']:02d}:00, evidence subgraph "
                         f"{len(inc['packet']['assets'])} assets)"))
            if d is None:
                print(st.red("           No rule matched; sent for manual review."))
            else:
                print(st.red("           " + format_message(d["message"]).replace("\n", "\n           ")))

        sys.stdout.flush()
        time.sleep(args.delay)

    return live


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "building.yaml")
    parser.add_argument("--fault", type=int, default=1, help="which fault under 'faults:' in the config (1 = leak, 2 = pump, 3 = blockage)")
    parser.add_argument("--type", choices=["leak", "pump_failure", "blockage"], help="custom fault type")
    parser.add_argument("--target", help="custom fault target, e.g. F3A-KIT, DTA-5-4, F1B-SUPPLY, PUMP1")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=6, help="start hour")
    parser.add_argument("--end", type=int, default=12, help="end hour")
    parser.add_argument("--seed", type=int, default=0, help="sensor-noise seed")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between hours")
    parser.add_argument("--normal-first", action="store_true", help="replay a normal day before the fault day")
    parser.add_argument("--hide-answer", action="store_true", help="hide the [SIM] fault on/off lines and the final check")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    st = Style(not args.no_color)

    config = load_config(args.config.resolve())
    network_path = (PROJECT_ROOT / config.network_file).resolve()
    layout = load_layout(LAYOUT)
    kg = BuildingKnowledgeGraph(WaterNetworkLoader(network_path).load(), layout)
    labels = {s["id"]: s["label"].split(": ", 1)[-1] for s in layout["sensors"]}
    labels.update({a: kg.asset(a).get("label", a) for a in kg.asset_ids()})

    if args.type:
        if not args.target:
            parser.error("--type needs --target")
        spec = (args.type, args.target, args.severity, args.start, args.end, DEFAULT_PARAMS.get(args.type, {}))
    else:
        f = config.faults[args.fault - 1]
        spec = (f.type, f.target_id, f.severity, f.start_hour, f.end_hour, f.params)
    fault = create_fault(*spec[:5], **spec[5])

    print(f"\nSimulating {network_path.name}: normal day and fault day "
          f"({config.simulation.duration_hours} h, {config.simulation.hydraulic_timestep_seconds} s steps) ...")

    # WNTR's EpanetSimulator writes temp.inp into the cwd; keep it out of the repo.
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            runner = FaultScenarioRunner(network_path, config.simulation)
            baseline, _ = runner.run_baseline()
            dataset, _ = runner.run_fault(fault)
        finally:
            os.chdir(here)

    pipeline = BuildingDiagnosisPipeline(kg, baseline)
    expected = twin(baseline, layout)
    # The system never sees the answer key: scrub the label columns.
    blind = lambda d: d.assign(scenario="unknown", state="normal")

    print("Simulation done. Replaying hour by hour "
          f"({len(layout['sensors'])} sensors, alarm when >= {MIN_ASSETS} read z >= {THRESHOLD:g}).")
    print("Each hour: simulated building state | sensors off-normal.  "
          "'!' = a noisy sensor reading vs the twin's normal value.\n")

    if args.normal_first:
        replay_day("NORMAL DAY -- no fault", baseline, expected, layout, labels,
                   pipeline.run_live(blind(baseline), seed=args.seed), None, args, st)
        print()

    title = ("FAULT DAY" if args.hide_answer
             else f"FAULT DAY -- {fault.scenario_name()}")
    live = replay_day(title, dataset, expected, layout, labels,
                      pipeline.run_live(blind(dataset), seed=args.seed), fault, args, st)

    final = live[max(live)] if live else []
    named = ((final[0]["diagnosis"] or {}).get("root_cause_asset") if final else None)
    print(st.bold("=" * 100))
    if not args.hide_answer:
        verdict = ("CORRECT" if named == fault.target_id
                   else "NOT DETECTED" if not final else "WRONG")
        print(st.bold(f"End of day. System named {named or '-'}; injected fault was at {fault.target_id} -> {verdict}"))
    else:
        print(st.bold(f"End of day. System named {named or 'nothing'}."))
    print(st.bold("=" * 100))


if __name__ == "__main__":
    main()
