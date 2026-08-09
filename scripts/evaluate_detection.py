"""
Score a detection run against simulation ground truth.

Every generated scenario ships an exact answer key: the metadata JSON
names the faulted asset, and the dataset's ``state`` column marks which
timesteps were fault-active. That makes detection measurable rather than
anecdotal.

    python scripts/evaluate_detection.py
    python scripts/evaluate_detection.py --name statistical

Three things get measured:

detection
    Did an incident open at all, and how many hours after the fault
    started. Scored at timestamp level, so a detector that fires for one
    hour of a six-hour fault is not credited the same as one that tracks
    it throughout.

localization
    How close the top-ranked candidate is to the truth, in graph hops,
    plus hit@k over the ranked list. This uses the Phase 2 topology and
    is the number Phase 4 has to improve on.

false alarms
    Counted only on genuinely normal time: the baseline control run, and
    the pre-fault hours of each scenario.

    Firing during ``recovery`` is reported separately and NOT counted as
    a false alarm. After a fault clears, tank levels and pump schedules
    have genuinely drifted -- baseline deviations there reach 29 m, larger
    than during some faults. The network really is off-nominal, so an
    alarm is correct behaviour; it is simply not the fault window.
"""

import argparse
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
# These must come AFTER src is added to sys.path
# --------------------------------------------------

import networkx as nx
import pandas as pd

from twinrag.graph import NetworkGraphBuilder
from twinrag.simulation.network_loader import WaterNetworkLoader


GENERATED_ROOT = PROJECT_ROOT / "data" / "generated"
MANIFEST_PATH = GENERATED_ROOT / "scenarios_manifest.csv"
DETECTION_ROOT = GENERATED_ROOT / "detection"
NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Net3.inp"

BASELINE_SCENARIO = "normal"

HIT_AT_K = (1, 3, 5, 10)


def _resolve(path_str: str) -> Path:
    return PROJECT_ROOT / Path(str(path_str).replace("\\", "/"))


def _link_endpoints(graph) -> dict:
    """
    Map each link name to the two nodes it joins.

    Link faults (blockage, pump failure) name a link as ground truth,
    but hop distance is only defined between nodes.
    """

    endpoints = {}

    for _, _, data in graph.edges(data=True):
        endpoints[data["link_name"]] = (data["start_node"], data["end_node"])

    return endpoints


def _asset_nodes(asset_id: str, graph, endpoints: dict) -> set:
    """
    The node positions an asset occupies, whether it is a node or a link.
    """

    if asset_id in graph:
        return {asset_id}

    if asset_id in endpoints:
        return set(endpoints[asset_id])

    return set()


def _truth_nodes(metadata: dict, graph, endpoints: dict) -> set:
    """
    Node positions of the injected fault.
    """

    nodes = set()

    for asset_id in metadata.get("affected_nodes") or []:
        nodes |= _asset_nodes(str(asset_id), graph, endpoints)

    for asset_id in metadata.get("affected_links") or []:
        nodes |= _asset_nodes(str(asset_id), graph, endpoints)

    return nodes


def _hop_distance(candidate_nodes: set, truth_nodes: set, distances: dict):
    """
    Fewest hops between a candidate and any faulted asset.

    ``distances`` is a per-truth-node BFS map, precomputed once per
    scenario.
    """

    if not candidate_nodes or not truth_nodes:
        return None

    best = None

    for truth_node, reachable in distances.items():
        for node in candidate_nodes:
            hops = reachable.get(node)

            if hops is None:
                continue

            if best is None or hops < best:
                best = hops

    return best


def _evaluate_scenario(
    scenario: str,
    report: dict,
    truth: dict,
    timeline: list,
    graph,
    endpoints: dict,
) -> dict:
    """
    Score one scenario.

    ``truth`` carries ``start_s``, ``end_s`` and the ground-truth asset
    IDs. For the baseline control, ``start_s`` is ``None`` — every
    timestamp is normal time.
    """

    incidents = report.get("incidents") or []

    fired = set()

    for incident in incidents:
        first = incident["detected_at_s"]
        last = incident["last_seen_s"]

        fired |= {t for t in timeline if first <= t <= last}

    start_s = truth["start_s"]
    end_s = truth["end_s"]

    if start_s is None:
        # Negative control: all normal, so every alarm is false.
        normal_times = set(timeline)
        active_times = set()
        recovery_times = set()
    else:
        normal_times = {t for t in timeline if t < start_s}
        active_times = {t for t in timeline if start_s <= t < end_s}
        recovery_times = {t for t in timeline if t >= end_s}

    true_positive = len(fired & active_times)
    false_negative = len(active_times - fired)
    false_positive = len(fired & normal_times)
    recovery_fired = len(fired & recovery_times)

    # --------------------------------------------------
    # Latency
    # --------------------------------------------------

    latency_hours = None

    if start_s is not None:
        in_window = sorted(fired & active_times)

        if in_window:
            latency_hours = (in_window[0] - start_s) / 3600

    # --------------------------------------------------
    # Localization
    # --------------------------------------------------

    truth_nodes = truth["nodes"]
    truth_ids = truth["asset_ids"]

    distances = {
        node: nx.single_source_shortest_path_length(graph, node)
        for node in truth_nodes
        if node in graph
    }

    # Rank candidates from the incident that overlaps the fault window,
    # falling back to the first incident. Scoring the recovery incident's
    # candidates against the fault location would be measuring the wrong
    # window.
    ranked = []

    for incident in incidents:
        overlaps = active_times and any(
            incident["detected_at_s"] <= t <= incident["last_seen_s"]
            for t in active_times
        )

        if overlaps or not ranked:
            ranked = incident.get("candidates") or []

        if overlaps:
            break

    hit_at = {k: 0 for k in HIT_AT_K}
    first_hit_rank = None

    for rank, candidate in enumerate(ranked, start=1):
        if str(candidate["asset_id"]) in truth_ids:
            first_hit_rank = rank
            break

    for k in HIT_AT_K:
        hit_at[k] = int(first_hit_rank is not None and first_hit_rank <= k)

    top_hops = None

    if ranked and truth_nodes:
        top_hops = _hop_distance(
            _asset_nodes(str(ranked[0]["asset_id"]), graph, endpoints),
            truth_nodes,
            distances,
        )

    return {
        "scenario": scenario,
        "fault_type": truth["fault_type"],
        "severity": truth["severity"],
        "detected": int(bool(incidents)),
        "incident_count": len(incidents),
        "latency_hours": latency_hours,
        "true_positive_steps": true_positive,
        "false_negative_steps": false_negative,
        "false_positive_steps": false_positive,
        "recovery_fired_steps": recovery_fired,
        "active_steps": len(active_times),
        "normal_steps": len(normal_times),
        "recovery_steps": len(recovery_times),
        "epicenter": (ranked[0]["asset_id"] if ranked else None),
        "epicenter_hops": top_hops,
        "first_hit_rank": first_hit_rank,
        "candidate_count": len(ranked),
        **{f"hit_at_{k}": hit_at[k] for k in HIT_AT_K},
    }


def _percent(numerator, denominator) -> str:
    if not denominator:
        return "n/a"

    return f"{100 * numerator / denominator:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a detection run against ground truth."
    )

    parser.add_argument(
        "--name",
        default="residual",
        help="Detection run subdirectory under data/generated/detection.",
    )

    args = parser.parse_args()

    run_dir = DETECTION_ROOT / args.name
    incidents_path = run_dir / "incidents.json"

    if not incidents_path.exists():
        raise SystemExit(
            f"No detection run at {run_dir.relative_to(PROJECT_ROOT)}.\n"
            f"Run: python scripts/detect_anomalies.py --name {args.name}"
        )

    reports = json.loads(incidents_path.read_text(encoding="utf-8"))

    manifest = pd.read_csv(MANIFEST_PATH)

    network = WaterNetworkLoader(NETWORK_PATH).load()
    graph = NetworkGraphBuilder(network).build()
    endpoints = _link_endpoints(graph)

    print(f"\nDetection run: {args.name}")

    run_config = run_dir / "run.json"

    if run_config.exists():
        config = json.loads(run_config.read_text(encoding="utf-8"))
        sensors = config.get("sensors", {})
        print(
            f"  threshold={config['config']['score_threshold']} sigma  "
            f"min_assets={config['config']['min_assets']}  "
            f"noise={'on' if sensors.get('noise_enabled') else 'OFF'}  "
            f"scale x{config.get('noise_scale', 1)}"
        )

    rows = []

    # --------------------------------------------------
    # Fault scenarios
    # --------------------------------------------------

    for _, entry in manifest.iterrows():
        scenario = entry["scenario"]

        if scenario not in reports:
            print(f"  WARNING: {scenario} missing from the detection run.")
            continue

        metadata = json.loads(
            _resolve(entry["metadata_file"]).read_text(encoding="utf-8")
        )

        timeline = sorted(
            pd.read_csv(
                _resolve(entry["dataset_file"]),
                usecols=["timestamp_s"],
            )["timestamp_s"]
            .unique()
            .tolist()
        )

        asset_ids = {
            str(asset)
            for asset in (metadata.get("affected_nodes") or [])
            + (metadata.get("affected_links") or [])
        }

        truth = {
            "start_s": int(entry["start_hour"]) * 3600,
            "end_s": int(entry["end_hour"]) * 3600,
            "fault_type": entry["fault_type"],
            "severity": entry["severity"],
            "nodes": _truth_nodes(metadata, graph, endpoints),
            "asset_ids": asset_ids,
        }

        rows.append(
            _evaluate_scenario(
                scenario,
                reports[scenario],
                truth,
                timeline,
                graph,
                endpoints,
            )
        )

    # --------------------------------------------------
    # Baseline negative control
    # --------------------------------------------------

    control = None

    if BASELINE_SCENARIO in reports:
        baseline_path = PROJECT_ROOT / "data" / "processed" / "baseline.csv"

        timeline = sorted(
            pd.read_csv(baseline_path, usecols=["timestamp_s"])["timestamp_s"]
            .unique()
            .tolist()
        )

        control = _evaluate_scenario(
            BASELINE_SCENARIO,
            reports[BASELINE_SCENARIO],
            {
                "start_s": None,
                "end_s": None,
                "fault_type": "none",
                "severity": 0.0,
                "nodes": set(),
                "asset_ids": set(),
            },
            timeline,
            graph,
            endpoints,
        )

    results = pd.DataFrame(rows)

    if results.empty:
        raise SystemExit("No scenarios scored.")

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    total = len(results)

    print(f"\n{'=' * 68}\nDETECTION  ({total} fault scenarios)\n{'=' * 68}")

    print(
        f"  Scenarios with an incident : {results['detected'].sum()}/{total} "
        f"({_percent(results['detected'].sum(), total)})"
    )

    active_total = results["active_steps"].sum()
    tp_total = results["true_positive_steps"].sum()
    fp_total = results["false_positive_steps"].sum()

    control_fp = control["false_positive_steps"] if control else 0
    control_steps = control["normal_steps"] if control else 0

    normal_total = results["normal_steps"].sum() + control_steps
    fp_all = fp_total + control_fp

    print(
        f"  Fault-active recall        : {tp_total}/{active_total} "
        f"({_percent(tp_total, active_total)}) timesteps"
    )

    print(
        f"  Precision (vs normal time) : "
        f"{_percent(tp_total, tp_total + fp_all)}"
    )

    print(
        f"  False alarms on normal time: {fp_all}/{normal_total} timesteps "
        f"({_percent(fp_all, normal_total)})"
    )

    if control is not None:
        verdict = "FALSE ALARM" if control["detected"] else "clean"

        print(
            f"  Baseline control           : {verdict} "
            f"({control['false_positive_steps']}/{control['normal_steps']} "
            "timesteps flagged)"
        )

    latencies = results["latency_hours"].dropna()

    if not latencies.empty:
        print(
            f"  Detection latency (hours)  : median={latencies.median():.1f} "
            f"max={latencies.max():.1f} "
            f"(same-hour on {(latencies == 0).sum()}/{len(latencies)})"
        )

    recovery_fired = results["recovery_fired_steps"].sum()
    recovery_total = results["recovery_steps"].sum()

    print(
        f"  Recovery-window firing     : "
        f"{recovery_fired}/{recovery_total} timesteps "
        f"({_percent(recovery_fired, recovery_total)}) "
        "- not scored as false alarm"
    )

    # --------------------------------------------------
    # Localization
    # --------------------------------------------------

    print(f"\n{'=' * 68}\nLOCALIZATION\n{'=' * 68}")

    for k in HIT_AT_K:
        column = f"hit_at_{k}"
        print(
            f"  Faulted asset in top {k:<2d}     : "
            f"{results[column].sum()}/{total} "
            f"({_percent(results[column].sum(), total)})"
        )

    hops = results["epicenter_hops"].dropna()

    if not hops.empty:
        print(
            f"  Top-1 distance to fault    : median={hops.median():.0f} hops "
            f"max={hops.max():.0f} hops "
            f"(exact on {(hops == 0).sum()}/{len(hops)})"
        )

    print(
        f"  Candidates per incident    : "
        f"median={results['candidate_count'].median():.0f} "
        f"max={results['candidate_count'].max():.0f}"
    )

    # --------------------------------------------------
    # Per fault type
    # --------------------------------------------------

    print(f"\n{'=' * 68}\nBY FAULT TYPE\n{'=' * 68}")

    print(
        f"  {'type':<14s} {'n':>3s} {'detected':>9s} {'recall':>8s} "
        f"{'hit@1':>7s} {'hit@5':>7s} {'med hops':>9s} {'latency':>8s}"
    )

    for fault_type, group in results.groupby("fault_type"):
        group_hops = group["epicenter_hops"].dropna()
        group_latency = group["latency_hours"].dropna()

        print(
            f"  {fault_type:<14s} {len(group):>3d} "
            f"{group['detected'].sum():>9d} "
            f"{_percent(group['true_positive_steps'].sum(), group['active_steps'].sum()):>8s} "
            f"{_percent(group['hit_at_1'].sum(), len(group)):>7s} "
            f"{_percent(group['hit_at_5'].sum(), len(group)):>7s} "
            f"{(f'{group_hops.median():.0f}' if not group_hops.empty else 'n/a'):>9s} "
            f"{(f'{group_latency.median():.1f}h' if not group_latency.empty else 'n/a'):>8s}"
        )

    # --------------------------------------------------
    # Misses worth looking at
    # --------------------------------------------------

    missed = results[results["detected"] == 0]

    if not missed.empty:
        print(f"\nUndetected scenarios ({len(missed)}):")

        for _, row in missed.iterrows():
            print(f"  {row['scenario']}")

    mislocated = results[results["hit_at_5"] == 0]

    if not mislocated.empty:
        print(f"\nFaulted asset outside top 5 ({len(mislocated)}):")

        for _, row in mislocated.head(10).iterrows():
            distance = row["epicenter_hops"]

            # An undetected scenario has no epicenter at all, which pandas
            # stores as NaN -- a truthy float, so it cannot be defaulted
            # with `or` and cannot be formatted as a string.
            epicenter = row["epicenter"]
            epicenter = "-" if pd.isna(epicenter) else str(epicenter)

            print(
                f"  {row['scenario']:32s} top-1={epicenter:>6s} "
                f"({'?' if pd.isna(distance) else int(distance)} hops away)"
            )

    # --------------------------------------------------
    # Persist
    # --------------------------------------------------

    results.to_csv(run_dir / "evaluation.csv", index=False)

    summary = {
        "run": args.name,
        "scenarios": int(total),
        "detected": int(results["detected"].sum()),
        "fault_active_recall": (
            float(tp_total / active_total) if active_total else None
        ),
        "precision": (
            float(tp_total / (tp_total + fp_all)) if (tp_total + fp_all) else None
        ),
        "false_alarm_rate_normal_time": (
            float(fp_all / normal_total) if normal_total else None
        ),
        "baseline_control_clean": (
            bool(not control["detected"]) if control else None
        ),
        "median_latency_hours": (
            float(latencies.median()) if not latencies.empty else None
        ),
        "median_epicenter_hops": (
            float(hops.median()) if not hops.empty else None
        ),
        "recovery_firing_rate": (
            float(recovery_fired / recovery_total) if recovery_total else None
        ),
        **{
            f"hit_at_{k}": float(results[f'hit_at_{k}'].mean())
            for k in HIT_AT_K
        },
    }

    with (run_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"\nWrote {(run_dir / 'evaluation.csv').relative_to(PROJECT_ROOT)} "
        "and evaluation.json"
    )


if __name__ == "__main__":
    main()
