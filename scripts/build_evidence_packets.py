"""
Phase 4 end to end on the building: detect -> retrieve -> evidence packet.

For every scenario in data/building/scenarios_manifest.csv:

    1. run the residual detector with the building's real sensors and
       instrument noise (twin baseline = data/building/processed/baseline.csv)
    2. take the first incident and retrieve its evidence subgraph from the
       building knowledge graph
    3. check the packet carries no ground truth, and save it

then score retrieval against the answer key -- the answer key is read
only here, *after* each packet is built, never passed into it.

Outputs (data/building/evidence/, gitignored):

    <scenario>.json          the evidence packet an LLM would receive
    retrieval_summary.csv    one row per scenario
"""

import argparse
import csv
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
from twinrag.building.sensors import building_sensor_model
from twinrag.detection import ResidualDetector
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.retrieval import RetrievalConfig, SubgraphRetriever


NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Building_G6.inp"
LAYOUT_PATH = PROJECT_ROOT / "data" / "networks" / "Building_G6.layout.json"
DATA_ROOT = PROJECT_ROOT / "data" / "building"
OUTPUT_DIR = DATA_ROOT / "evidence"


def _path(path_str: str) -> Path:
    return PROJECT_ROOT / Path(str(path_str).replace("\\", "/"))


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"asset_id": str})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--min-assets", type=int, default=2)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None,
                        help="Batch directory with processed/ and scenarios_manifest.csv (default data/building).")
    args = parser.parse_args()

    layout = load_layout(LAYOUT_PATH)
    kg = BuildingKnowledgeGraph.from_files(NETWORK_PATH, LAYOUT_PATH)
    retriever = SubgraphRetriever(
        kg, RetrievalConfig(max_seeds=args.max_seeds, hops=args.hops)
    )

    print("Knowledge graph:", kg.summary())

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    output_dir = data_root / "evidence"

    baseline = _read(data_root / "processed" / "baseline.csv")
    manifest = list(csv.DictReader((data_root / "scenarios_manifest.csv").open(encoding="utf-8")))

    def detector(seed):
        return ResidualDetector(
            baseline,
            sensor_model=building_sensor_model(layout, seed=seed),
            score_threshold=args.threshold,
            min_assets=args.min_assets,
        )

    control = detector(args.seed + 10_000).detect(baseline)
    print(f"Baseline control: {len(control.incidents)} incident(s) (expect 0)\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, row in enumerate(manifest):
        name = row["scenario"]
        target = row["target_id"]
        report = detector(args.seed + index).detect(_read(_path(row["dataset_file"])))

        result = {
            "scenario": name,
            "fault_type": row["fault_type"],
            "target_id": target,
            "detected": bool(report.incidents),
        }

        if report.incidents:
            incident = report.incidents[0]
            packet = retriever.retrieve(incident, incident_id=f"INC-{index + 1:03d}", events=report.events)
            packet.assert_no_leakage(forbidden=[name])

            (output_dir / f"{name}.json").write_text(packet.to_json(), encoding="utf-8")

            ids = {a["id"] for a in packet.assets}
            facts = packet.topology_facts
            extra = facts.get("extra_flow", {})
            loss = facts.get("pressure_loss", {})

            # "At the focus": the fault sits exactly where the topology
            # points -- the deepest pipe carrying extra flow (or the tap it
            # feeds), or inside the suspect span above the pressure loss.
            focus = set(extra.get("deepest_assets_with_extra_flow", []))
            focus |= set(extra.get("supplied_through_them", []))
            focus |= set(loss.get("suspect_span_upward", []))
            for chain in facts.get("low_tank_supply_chain", {}).values():
                focus |= set(chain)

            result.update(
                detected_at=f"{incident.detected_at_s // 3600:02d}:00",
                start=f"{int(row['start_hour']):02d}:00",
                detector_top1=incident.epicenter,
                target_in_packet=target in ids,
                target_at_focus=target in focus,
                focus=" ".join(sorted(focus)),
                assets=len(packet.assets),
                observations=len(packet.observations),
                relations=len(packet.relations),
                est_tokens=packet.estimated_tokens(),
            )

        rows.append(result)

        mark = (
            "not detected" if not result["detected"]
            else f"in packet={result['target_in_packet']!s:5}  focus={result['target_at_focus']!s:5}  "
                 f"{result['assets']:3d} assets  ~{result['est_tokens']:5d} tokens  focus: {result['focus']}"
        )
        print(f"{name:38s} {mark}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "retrieval_summary.csv", index=False)

    detected = summary[summary["detected"]]
    print("\nBy fault type (detected scenarios only):")
    for fault_type, group in detected.groupby("fault_type"):
        n_all = int((summary["fault_type"] == fault_type).sum())
        print(
            f"  {fault_type:13s} detected {len(group)}/{n_all}  "
            f"target in packet {group['target_in_packet'].mean():.0%}  "
            f"target at focus {group['target_at_focus'].mean():.0%}  "
            f"median packet {int(group['assets'].median())} assets / ~{int(group['est_tokens'].median())} tokens"
        )

    print(f"\n-> {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
