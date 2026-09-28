"""
Build the 3D building digital-twin viewer.

Packs the building layout plus every simulated scenario into
viewer/building_template.html, with three.js inlined, producing one
self-contained file that opens offline -- no server, no CDN:

    data/building/building_view.json   the data payload (for other tools)
    data/building/building_twin.html   the viewer

Run these first:

    python scripts/build_building_network.py
    python scripts/run_generated_scenarios.py --config configs/building.yaml \
        --output-root data/building --with-baseline
"""

import csv
import json
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
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.simulation.network_loader import WaterNetworkLoader


LAYOUT_PATH = PROJECT_ROOT / "data" / "networks" / "Building_G6.layout.json"
DATA_ROOT = PROJECT_ROOT / "data" / "building"
MANIFEST_PATH = DATA_ROOT / "scenarios_manifest.csv"
BASELINE_PATH = DATA_ROOT / "processed" / "baseline.csv"

TEMPLATE_PATH = PROJECT_ROOT / "viewer" / "building_template.html"
THREE_PATH = PROJECT_ROOT / "viewer" / "vendor" / "three.min.js"
ORBIT_PATH = PROJECT_ROOT / "viewer" / "vendor" / "OrbitControls.js"

PAYLOAD_PATH = DATA_ROOT / "building_view.json"
OUTPUT_PATH = DATA_ROOT / "building_twin.html"

PAGE_TITLE = "TwinRAG — Building Digital Twin"

#: Instrument noise used to express deviations in sigma, matching the
#: building sensors: a 0-10 bar transducer at 0.25% of full scale, and a
#: flow meter at 0.002 L/s plus 2% of reading.
SENSOR_NOISE = {
    "pressure_abs": 0.25,
    "flow_abs": 0.002,
    "flow_rel": 0.02,
}

FAULT_ORDER = {"leak": 0, "pump_failure": 1, "blockage": 2}

#: Same wrapper as scripts/build_viewer.py: doctype, charset, theme toggle.
DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M16 3C16 3 6 15 6 21a10 10 0 0 0 20 0C26 15 16 3 16 3Z' fill='%230e7c86'/%3E%3C/svg%3E">
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ background: #eef1f3; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) body {{ background: #0c1418; }} }}
  :root[data-theme="dark"] body {{ background: #0c1418; }}
  :root[data-theme="light"] body {{ background: #eef1f3; }}
  #themeToggle {{
    position: fixed; top: 10px; right: 12px; z-index: 20;
    font: 500 12px/1 -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #40525c; background: #fff; border: 1px solid #cbd6dc;
    border-radius: 999px; padding: 7px 12px; cursor: pointer;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) #themeToggle {{ color: #a9bcc6; background: #121d23; border-color: #24343c; }}
  }}
  :root[data-theme="dark"] #themeToggle {{ color: #a9bcc6; background: #121d23; border-color: #24343c; }}
  :root[data-theme="light"] #themeToggle {{ color: #40525c; background: #fff; border-color: #cbd6dc; }}
</style>
</head>
<body>
<button type="button" id="themeToggle" aria-label="Switch between light and dark">Theme</button>
{fragment}
<script>
(function () {{
  var root = document.documentElement;
  var saved = null;
  try {{ saved = localStorage.getItem("twinrag-theme"); }} catch (e) {{}}
  if (saved) root.setAttribute("data-theme", saved);
  document.getElementById("themeToggle").addEventListener("click", function () {{
    var dark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    var next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {{ localStorage.setItem("twinrag-theme", next); }} catch (e) {{}}
  }});
}})();
</script>
</body>
</html>
"""


def _path(path_str: str) -> Path:
    return PROJECT_ROOT / Path(str(path_str).replace("\\", "/"))


def _series(dataset: pd.DataFrame, parameter: str, ids, scale: float, digits: int) -> dict:
    """
    ``{asset_id: [value per hour]}`` for one parameter.
    """

    frame = dataset[dataset["parameter"] == parameter]
    wide = frame.pivot(index="timestamp_s", columns="asset_id", values="value")
    wide = wide.sort_index()

    return {
        asset_id: [round(float(v) * scale, digits) for v in wide[asset_id]]
        for asset_id in ids
        if asset_id in wide.columns
    }


def _scenario_payload(dataset, layout, node_ids, link_ids) -> dict:
    # Node "demand" is deliberately left out. EPANET reports everything
    # leaving the pipes at a node, and a simulated leak *is* extra water
    # leaving at the leaking tap -- so demand there reads 80x normal and
    # hands the viewer the answer. No real building meters every tap
    # either. Water reaching a tap is derived from pressure instead.
    return {
        # pressure in m (tank "pressure" is water level)
        "p": _series(dataset, "pressure", node_ids, 1.0, 3),
        # flow converted to L/s for display
        "q": _series(dataset, "flowrate", link_ids, 1000.0, 5),
        "state": (
            dataset.drop_duplicates("timestamp_s")
            .sort_values("timestamp_s")["state"]
            .tolist()
            if "state" in dataset.columns
            else None
        ),
    }


def _describe(row: dict, metadata: dict, layout: dict) -> str:
    """
    A scenario label a person would say, e.g.
    "Leak 0.20 L/s · Flat 3A master bathroom · 06:00–12:00".
    """

    target = row["target_id"]
    asset = layout["nodes"].get(target) or layout["links"].get(target) or {}
    where = asset.get("label", target)
    start = int(row["start_hour"])
    end = row["end_hour"]
    window = f"{start:02d}:00–" + (f"{int(end):02d}:00" if end not in ("", None) else "end")

    if row["fault_type"] == "leak":
        rate = metadata.get("leak_demand")
        size = f" {rate * 1000:.2f} L/s" if rate else ""
        return f"Leak{size} · {where} · {window}"

    if row["fault_type"] == "pump_failure":
        return f"Pump outage · {where} · {window}"

    if row["fault_type"] == "blockage":
        return f"Blocked · {where} · {window}"

    return f"{row['fault_type']} · {where} · {window}"


def _asset_rows(entries: dict, drop=("path",)) -> list:
    rows = []
    for asset_id, entry in entries.items():
        row = {"id": asset_id}
        row.update({k: v for k, v in entry.items() if v is not None})
        rows.append(row)
    return rows


class SnapshotTable:
    """
    Deduplicated incident snapshots. The live replay repeats the same
    diagnosis for many consecutive hours; each distinct one is stored once
    and referenced by index.
    """

    def __init__(self):
        self.rows, self._index = [], {}

    def add(self, snapshot: dict) -> int:
        key = json.dumps(snapshot, sort_keys=True)
        if key not in self._index:
            self._index[key] = len(self.rows)
            self.rows.append(snapshot)
        return self._index[key]


def _live_payload(live: dict, table: SnapshotTable) -> dict:
    """``{hour: [snapshot index, ...]}`` -- the incidents known at each hour."""
    return {str(hour): [table.add(s) for s in _viewer_incidents(incidents)] for hour, incidents in live.items()}


def _viewer_incidents(incidents) -> list:
    """
    What the dashboard needs from each pipeline incident: when it was
    raised, which assets the retrieval pulled out, which alarmed, and the
    rule-based diagnosis with its message and cited evidence.
    """

    out = []
    for inc in incidents:
        packet, d = inc["packet"], inc["diagnosis"]
        alarmed = []
        for obs in packet["observations"]:
            if obs["status"].startswith(("alarm", "early")):
                alarmed += obs.get("assets") or [obs["asset"]]
        out.append({
            "id": inc["id"],
            "first": inc["first_alarm_hour"],
            "last": inc["last_alarm_hour"],
            "subgraph": [a["id"] for a in packet["assets"]],
            "alarmed": sorted(set(alarmed)),
            "diagnosis": None if not d else {
                "root": d["root_cause_asset"],
                "fault_type": d["fault_type"],
                "location": d["location"],
                "confidence": d["confidence"],
                "affected": d["affected_assets"],
                "reasoning": d["reasoning"],
                "message": d["message"],
            },
        })
    return out


def main() -> None:
    for path, hint in (
        (LAYOUT_PATH, "python scripts/build_building_network.py"),
        (MANIFEST_PATH, "python scripts/run_generated_scenarios.py --config configs/building.yaml --output-root data/building --with-baseline"),
        (BASELINE_PATH, "the same command, with --with-baseline"),
    ):
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(PROJECT_ROOT)}. Run: {hint}")

    layout = load_layout(LAYOUT_PATH)

    node_ids = list(layout["nodes"])
    link_ids = list(layout["links"])

    # Pressure-driven demand settings, read from the model itself so the
    # viewer's "water reaching the tap" uses exactly EPANET's rule.
    hydraulic = WaterNetworkLoader(
        LAYOUT_PATH.parent / layout["building"]["network_file"]
    ).load().options.hydraulic
    demand_model = {
        "required_pressure": float(hydraulic.required_pressure),
        "minimum_pressure": float(hydraulic.minimum_pressure),
        "exponent": float(hydraulic.pressure_exponent),
    }

    baseline = pd.read_csv(BASELINE_PATH, dtype={"asset_id": str})
    hours = sorted(int(t) // 3600 for t in baseline["timestamp_s"].unique())

    # The same detect -> retrieve -> diagnose pipeline the evaluation
    # scores, run here so the dashboard shows exactly its output.
    kg = BuildingKnowledgeGraph.from_files(LAYOUT_PATH.parent / layout["building"]["network_file"], LAYOUT_PATH)
    pipeline = BuildingDiagnosisPipeline(kg, baseline)
    table = SnapshotTable()

    scenarios = [
        {
            "name": "normal",
            "label": "Normal day — no fault",
            "fault_type": None,
            "target_id": None,
            "severity": None,
            "start_hour": None,
            "end_hour": None,
            **_scenario_payload(baseline, layout, node_ids, link_ids),
            "live": _live_payload(pipeline.run_live(baseline, seed=10_000), table),
        }
    ]

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    # Noise seeds follow manifest order, as in build_evidence_packets.py.
    seed_of = {row["scenario"]: index for index, row in enumerate(manifest)}

    manifest.sort(
        key=lambda r: (
            FAULT_ORDER.get(r["fault_type"], 9),
            r["target_id"],
            float(r["severity"]),
            int(r["start_hour"]),
        )
    )

    for row in manifest:
        dataset = pd.read_csv(_path(row["dataset_file"]), dtype={"asset_id": str})
        metadata = json.loads(_path(row["metadata_file"]).read_text(encoding="utf-8"))

        scenarios.append(
            {
                "name": row["scenario"],
                "label": _describe(row, metadata, layout),
                "fault_type": row["fault_type"],
                "target_id": row["target_id"],
                "severity": float(row["severity"]),
                "start_hour": int(row["start_hour"]),
                "end_hour": int(row["end_hour"]) if row["end_hour"] not in ("", None) else None,
                **_scenario_payload(dataset, layout, node_ids, link_ids),
                "live": _live_payload(pipeline.run_live(dataset, seed=seed_of[row["scenario"]]), table),
            }
        )

        live = scenarios[-1]["live"]
        final = [table.rows[i] for i in live[max(live, key=int)]] if live else []
        first = final[0]["diagnosis"] if final and final[0]["diagnosis"] else None
        print(f"  packed {row['scenario']:36s} incidents={len(final)}"
              + (f"  first raised {final[0]['first']:02d}:00 -> {first['root']} ({first['fault_type']}, {first['confidence']})" if first else ""))

    payload = {
        "building": layout["building"],
        "floors": layout["floors"],
        "flats": layout["flats"],
        "rooms": layout["rooms"],
        "nodes": _asset_rows(layout["nodes"]),
        "links": _asset_rows(layout["links"]),
        "sensors": layout["sensors"],
        "hours": hours,
        "noise": SENSOR_NOISE,
        "demand_model": demand_model,
        "scenarios": scenarios,
        # Incident snapshots referenced by each scenario's "live" replay.
        "snapshots": table.rows,
    }

    PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, separators=(",", ":"))
    PAYLOAD_PATH.write_text(payload_json, encoding="utf-8")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    three = THREE_PATH.read_text(encoding="utf-8")
    orbit = ORBIT_PATH.read_text(encoding="utf-8")

    for name, script in (("three.min.js", three), ("OrbitControls.js", orbit)):
        if "</script" in script.lower():
            raise SystemExit(f"{name} contains '</script' and cannot be inlined.")

    fragment = (
        template
        .replace("__THREE_JS__", three)
        .replace("__ORBIT_CONTROLS_JS__", orbit)
        # "</" inside a JSON script block would end the block early.
        .replace("__TWIN_DATA__", payload_json.replace("</", "<\\/"))
    )

    # The template opens with its own <title>; the document supplies one.
    fragment = fragment.split("</title>", 1)[1] if fragment.lstrip().startswith("<title>") else fragment

    OUTPUT_PATH.write_text(
        DOCUMENT.format(title=PAGE_TITLE, fragment=fragment),
        encoding="utf-8",
    )

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\n{len(scenarios) - 1} fault scenarios + normal day")
    print(f"-> {PAYLOAD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"-> {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
