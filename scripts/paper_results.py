"""
Results for the paper: repeated-seed accuracy, sensor ablation, false
alarms and per-scenario deviation tables for the building twin.

Runs the full detect -> retrieve -> diagnose pipeline
(``BuildingDiagnosisPipeline.run``) on every scenario of the design set
and the three held-out sets, under several sensor-noise seeds and sensor
layouts, and scores the FIRST incident's diagnosis against the answer key
(read only for scoring).

    python scripts/paper_results.py            # ~2-3 min
    python scripts/paper_results.py --seeds 5  # quicker

Outputs (data/paper/, gitignored):
    runs.csv            one row per (layout, set, scenario, seed)
    summary.json        headline metrics with 95% Wilson intervals
    layouts.csv         accuracy per sensor layout
    by_type.csv         accuracy per fault type and set (full layout)
    false_alarms.csv    incidents raised on the normal day, per seed
    deviations.json     expected vs observed values for six scenarios
"""

import argparse
import copy
import csv
import json
import math
import random
import statistics
import sys
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

import pandas as pd

from twinrag.building import load_layout
from twinrag.building.pipeline import BuildingDiagnosisPipeline
import twinrag.building.sensors as building_sensors
from twinrag.building.sensors import BUILDING_SENSOR_NOISE
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.simulation.network_loader import WaterNetworkLoader


NETWORK = PROJECT_ROOT / "data" / "networks" / "Building_G6.inp"
LAYOUT = PROJECT_ROOT / "data" / "networks" / "Building_G6.layout.json"
OUT = PROJECT_ROOT / "data" / "paper"

SETS = {
    "design": PROJECT_ROOT / "data" / "building",
    "holdout1": PROJECT_ROOT / "data" / "building_holdout",
    "holdout2": PROJECT_ROOT / "data" / "building_holdout2",
    "holdout3_blind": PROJECT_ROOT / "data" / "building_holdout3",
}

PLANT_ROLES = {"pump", "tank_outlet", "municipal_inlet", "roof_tank", "sump"}


def sensor_layouts(layout) -> dict:
    """
    Named subsets of the 103 installed instruments, from cheapest to full.
    """

    sensors = layout["sensors"]
    role = lambda s: (layout["links"].get(s["asset_id"]) or layout["nodes"].get(s["asset_id"]))["role"]

    room = [s for s in sensors if role(s) == "room_branch"]
    half_room = random.Random(7).sample(room, len(room) // 2)

    def keep(pred):
        return [s for s in sensors if pred(s)]

    return {
        "L16 pressure only": keep(lambda s: s["parameter"] == "pressure"),
        "L19 pressure + plant flow": keep(lambda s: s["parameter"] == "pressure" or role(s) in PLANT_ROLES),
        "L33 flat-level metering": keep(lambda s: role(s) != "room_branch"),
        "L68 half the room meters": keep(lambda s: role(s) != "room_branch" or s in half_room),
        "L98 no plant instruments": keep(lambda s: role(s) not in PLANT_ROLES),
        "L103 full": list(sensors),
    }


def score(predicted, target, kg) -> dict:
    if not isinstance(predicted, str) or predicted not in kg:
        return {"exact": False, "room": False, "flat": False}
    a, t = kg.asset(predicted), kg.asset(target)
    exact = predicted == target
    room = exact or (bool(a.get("room")) and a.get("room") == t.get("room"))
    flat = room or (bool(a.get("flat")) and a.get("flat") == t.get("flat"))
    return {"exact": exact, "room": room, "flat": flat}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def read(path):
    return pd.read_csv(path, dtype={"asset_id": str})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--ablation-seeds", type=int, default=5)
    parser.add_argument("--control-seeds", type=int, default=50)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    network = WaterNetworkLoader(NETWORK).load()
    layout = load_layout(LAYOUT)
    baseline = read(SETS["design"] / "processed" / "baseline.csv")

    layouts = sensor_layouts(layout)
    kgs = {}
    for name, sensors in layouts.items():
        sub = copy.deepcopy(layout)
        sub["sensors"] = sensors
        kgs[name] = BuildingKnowledgeGraph(network, sub)
    full = "L103 full"
    precise = "L103 full, 0.0005 L/s flow meters"
    standard_flow = tuple(BUILDING_SENSOR_NOISE["flowrate"])

    # Each variant: (name, knowledge graph, flow-meter noise (abs m3/s, rel), seeds).
    # The last one repeats the full layout with higher-grade flow meters
    # (noise floor 0.0005 L/s instead of 0.002 L/s).
    variants = [(name, kg, standard_flow, args.seeds if name == full else args.ablation_seeds)
                for name, kg in kgs.items()]
    variants.append((precise, kgs[full], (5e-7, 0.02), args.ablation_seeds))
    layout_sizes = {name: len(sensors) for name, sensors in layouts.items()}
    layout_sizes[precise] = len(layouts[full])

    def set_flow_noise(spec):
        # building_sensor_model reads this table at every call.
        building_sensors.BUILDING_SENSOR_NOISE["flowrate"] = spec

    scenarios = []
    for set_name, root in SETS.items():
        for row in csv.DictReader((root / "scenarios_manifest.csv").open(encoding="utf-8")):
            row["set"] = set_name
            # Manifest paths are project-relative (and carry backslashes on Windows).
            row["path"] = PROJECT_ROOT / row["dataset_file"].replace("\\", "/")
            scenarios.append(row)

    print(f"{len(scenarios)} scenarios in {len(SETS)} sets; layouts: {', '.join(f'{k} ({len(v)})' for k, v in layouts.items())}")

    runs = []
    for i, sc in enumerate(scenarios):
        data = read(sc["path"])
        start = int(sc["start_hour"])
        for lname, kg, flow_noise, n_seeds in variants:
            set_flow_noise(flow_noise)
            pipe = BuildingDiagnosisPipeline(kg, baseline)
            for seed in range(n_seeds):
                incidents = pipe.run(data, seed=1000 * seed + i)
                inc = incidents[0] if incidents else None
                d = (inc or {}).get("diagnosis") or {}
                s = score(d.get("root_cause_asset"), sc["target_id"], kg)
                packet_ids = {a["id"] for a in (inc or {}).get("packet", {}).get("assets", [])}
                epicenter = None
                if inc:
                    alarms = [o for o in inc["packet"]["observations"] if o.get("asset") and o["status"].startswith("alarm")]
                    epicenter = alarms[0]["asset"] if alarms else None
                runs.append({
                    "layout": lname, "sensors": layout_sizes[lname], "set": sc["set"],
                    "scenario": sc["scenario"], "fault_type": sc["fault_type"], "target": sc["target_id"],
                    "severity": float(sc["severity"]), "start_hour": start, "seed": seed,
                    "detected": bool(inc),
                    "latency_h": (inc["first_alarm_hour"] - start) if inc else None,
                    "predicted": d.get("root_cause_asset"), "predicted_type": d.get("fault_type"),
                    "confidence": d.get("confidence"),
                    "confirmed": bool(d) and (d.get("confidence") or 0) >= 0.5,
                    "exact": s["exact"], "room": s["room"], "flat": s["flat"],
                    "type_correct": d.get("fault_type") == sc["fault_type"],
                    "target_in_evidence": sc["target_id"] in packet_ids,
                    "detector_top1_exact": epicenter == sc["target_id"],
                    "grounded": (inc or {}).get("grounded", False),
                })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(scenarios)} scenarios")

    frame = pd.DataFrame(runs)
    frame.to_csv(OUT / "runs.csv", index=False)

    # ---- false alarms on the normal day (pure sensor noise) ----------
    fa = []
    for lname, kg, flow_noise, _ in variants:
        set_flow_noise(flow_noise)
        pipe = BuildingDiagnosisPipeline(kg, baseline)
        for seed in range(args.control_seeds):
            fa.append({"layout": lname, "seed": seed, "incidents": len(pipe.run(baseline, seed=500_000 + seed))})
    pd.DataFrame(fa).to_csv(OUT / "false_alarms.csv", index=False)
    set_flow_noise(standard_flow)

    # ---- summaries ---------------------------------------------------
    def block(df):
        n_scen = df["scenario"].nunique()
        out = {"scenarios": n_scen, "runs": len(df)}
        for m in ("detected", "exact", "room", "flat", "type_correct", "target_in_evidence", "detector_top1_exact"):
            per_seed = df.groupby("seed")[m].mean()
            mean = float(df[m].mean())
            lo, hi = wilson(round(mean * n_scen), n_scen)
            out[m] = {"mean": round(mean, 4), "seed_sd": round(float(per_seed.std(ddof=0)), 4),
                      "ci95": [round(lo, 4), round(hi, 4)]}
        conf = df[df["confirmed"]]
        out["confirmed_share"] = round(float(df["confirmed"].mean()), 4)
        out["precision_confirmed"] = round(float(conf["exact"].mean()), 4) if len(conf) else None
        lat = df.loc[df["detected"], "latency_h"].dropna()
        out["latency_h"] = {"median": float(lat.median()), "mean": round(float(lat.mean()), 2),
                            "p90": float(lat.quantile(0.9))} if len(lat) else None
        return out

    f_full = frame[frame["layout"] == full]
    summary = {
        "all_sets": block(f_full),
        "blind_holdout3": block(f_full[f_full["set"] == "holdout3_blind"]),
        "per_set": {s: block(f_full[f_full["set"] == s]) for s in SETS},
        "precise_flow_meters": block(frame[frame["layout"] == precise]),
        "precise_flow_meters_blind": block(frame[(frame["layout"] == precise) & (frame["set"] == "holdout3_blind")]),
        "false_alarms": {
            l: {"days": int(g["incidents"].size), "days_with_alarm": int((g["incidents"] > 0).sum()),
                "rate": round(float((g["incidents"] > 0).mean()), 4)}
            for l, g in pd.DataFrame(fa).groupby("layout")
        },
        "noise": BUILDING_SENSOR_NOISE,
        "seeds": args.seeds, "ablation_seeds": args.ablation_seeds, "control_seeds": args.control_seeds,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    rows = []
    for (lname, n), g in frame.groupby(["layout", "sensors"]):
        b = block(g)
        rows.append({"layout": lname, "sensors": n, **{m: b[m]["mean"] for m in ("detected", "exact", "room", "flat", "type_correct")},
                     "exact_ci_lo": b["exact"]["ci95"][0], "exact_ci_hi": b["exact"]["ci95"][1],
                     "precision_confirmed": b["precision_confirmed"],
                     "false_alarm_rate": summary["false_alarms"][lname]["rate"]})
        for ft, gt in g.groupby("fault_type"):
            rows[-1][f"room_{ft}"] = round(float(gt["room"].mean()), 4)
            rows[-1][f"detected_{ft}"] = round(float(gt["detected"].mean()), 4)
    pd.DataFrame(rows).sort_values("sensors").to_csv(OUT / "layouts.csv", index=False)

    bt = []
    for (s, ft), g in f_full.groupby(["set", "fault_type"]):
        b = block(g)
        bt.append({"set": s, "fault_type": ft, "scenarios": b["scenarios"],
                   **{m: b[m]["mean"] for m in ("detected", "exact", "room", "type_correct", "detector_top1_exact")},
                   "precision_confirmed": b["precision_confirmed"],
                   "latency_median_h": (b["latency_h"] or {}).get("median")})
    pd.DataFrame(bt).to_csv(OUT / "by_type.csv", index=False)

    # ---- six representative scenarios: expected vs observed ---------
    picks = [
        ("design", "leak_F3A-MBATH_sev100_t06_12", 8, ["F3A-MBATH-BR", "F3A-SUPPLY", "OHT-OUTLET", "F3A-CBATH-BR", "F3A-IN"]),
        ("design", "leak_F2B-UTL_sev10_t06_12", 8, ["F2B-UTL-BR", "F2B-SUPPLY", "F2B-KIT-BR", "F2B-IN"]),
        ("design", "blockage_DTA-5-4_sev100_t15_21", 17, ["F4A-IN", "F2A-IN", "F0A-IN", "F5A-IN", "F4B-IN"]),
        ("design", "blockage_F1B-SUPPLY_sev100_t15_21", 17, ["F1B-IN", "F1B-SUPPLY", "F0B-IN", "F2B-IN"]),
        ("holdout2", "blockage_ROOF-B_sev100_t06_12", 8, ["F6B-IN", "F3B-IN", "F0B-IN", "F6A-IN", "OHT-OUTLET"]),
        ("design", "pump_failure_PUMP1_sev100_t06_16", 12, ["PUMP1", "OHT", "F6A-IN", "F0A-IN", "SUMP"]),
    ]
    a_p, r_p = BUILDING_SENSOR_NOISE["pressure"]
    a_q, r_q = BUILDING_SENSOR_NOISE["flowrate"]
    dev = []
    for set_name, name, hour, assets in picks:
        data = read(SETS[set_name] / "processed" / f"{name}.csv")
        t = hour * 3600
        rows_ = []
        for a in assets:
            kind = "flowrate" if kgs[full].is_link(a) else "pressure"
            e = float(baseline[(baseline.asset_id == a) & (baseline.parameter == kind) & (baseline.timestamp_s == t)]["value"].iloc[0])
            o = float(data[(data.asset_id == a) & (data.parameter == kind) & (data.timestamp_s == t)]["value"].iloc[0])
            # As the detector computes it: from the reading's own true value.
            sigma = (a_q + r_q * abs(o)) if kind == "flowrate" else (a_p + r_p * abs(o))
            scale = 1000.0 if kind == "flowrate" else 1.0
            rows_.append({"asset": a, "label": kgs[full].asset(a)["label"], "parameter": kind,
                          "unit": "L/s" if kind == "flowrate" else "m",
                          "expected": round(e * scale, 4), "observed": round(o * scale, 4),
                          "residual": round((o - e) * scale, 4), "sigma": round(sigma * scale, 4),
                          "z": round(abs(o - e) / sigma, 1)})
        f = frame[(frame.layout == full) & (frame.scenario == name)]
        dev.append({"set": set_name, "scenario": name, "hour": hour, "rows": rows_,
                    "diagnosis": {
                        "predicted": f["predicted"].mode().iat[0] if f["predicted"].notna().any() else None,
                        "exact_rate": round(float(f["exact"].mean()), 3),
                        "confidence_median": float(f["confidence"].median()) if f["confidence"].notna().any() else None,
                        "first_alarm_after_start_h": float(f["latency_h"].median()) if f["latency_h"].notna().any() else None,
                    }})
    (OUT / "deviations.json").write_text(json.dumps(dev, indent=1), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("all_sets", "blind_holdout3", "false_alarms")}, indent=1))
    print(pd.DataFrame(rows).sort_values("sensors")[["layout", "sensors", "detected", "exact", "room", "flat", "type_correct", "precision_confirmed", "false_alarm_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
