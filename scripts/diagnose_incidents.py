"""
Phase 5: root-cause diagnosis of the building's evidence packets.

Reads the packets written by build_evidence_packets.py, diagnoses each,
and scores it against the answer key (read only here, after the
diagnosis). Compares with the detector's own top-ranked suspect.

Two engines produce the same diagnosis schema:

    rules  (default) deterministic rule-based diagnosis + alert message;
           free, instant, offline -- the primary output
    llm    an LLM over the same packet; optional comparison, costs API quota

    python scripts/diagnose_incidents.py                       # rules, every incident
    python scripts/diagnose_incidents.py --show-messages       # ...and print the alerts
    python scripts/diagnose_incidents.py --evidence data/building_holdout/evidence
    python scripts/diagnose_incidents.py --engine llm --per-type 2 --delay 45
    python scripts/diagnose_incidents.py --engine llm --no-graph --name nograph

For --engine llm the endpoint, key and model come from the project's .env
(see twinrag/reasoning/llm.py). For Groq's free tier:

    TWINRAG_LLM_BASE_URL=https://api.groq.com/openai/v1
    TWINRAG_LLM_API_KEY=gsk_...
    TWINRAG_LLM_MODEL=openai/gpt-oss-120b

Free tiers are throttled (Groq: ~8k tokens/min, ~200k/day), so start
with --per-type and use --delay 45 for longer runs.

Outputs, <data root>/diagnosis/<name>/ (gitignored):
    <scenario>.json   diagnosis (with alert message), grounding result, score
    summary.csv       one row per scenario
    run.json          engine, model, settings, token usage
"""

import argparse
import json
import sys
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

import pandas as pd

from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.reasoning import ChatClient, Diagnoser, ungrounded_packet
from twinrag.reasoning.llm import Usage
from twinrag.reasoning.rules import RuleBasedDiagnoser, format_message


NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Building_G6.inp"
LAYOUT_PATH = PROJECT_ROOT / "data" / "networks" / "Building_G6.layout.json"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "building" / "evidence"


def _score(predicted, target, kg) -> dict:
    """
    exact  -- the named asset is the faulted one
    room   -- exact, or another asset in the same room (a leaking tap vs
              the branch pipe feeding it): room-level diagnosis
    near   -- same room, or one pipe away (a closed riser segment vs the
              node just below it)
    """

    if not isinstance(predicted, str) or predicted not in kg:
        return {"exact": False, "room": False, "near": False}

    exact = predicted == target
    room_p = kg.asset(predicted).get("room")
    room_t = kg.asset(target).get("room")
    same_room = bool(room_p) and room_p == room_t

    def touching(a, b):
        return b in {kg.link_from.get(a), kg.link_to.get(a)}

    near = exact or same_room or touching(predicted, target) or touching(target, predicted)
    return {"exact": exact, "room": exact or same_room, "near": near}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--engine", choices=("rules", "llm"), default="rules",
                        help="rules (default; deterministic, free) or llm (optional comparison).")
    parser.add_argument("--evidence", default=None,
                        help="Evidence directory (default data/building/evidence).")
    parser.add_argument("--show-messages", action="store_true", help="Print each alert message.")
    parser.add_argument("--model", default=None, help="LLM model ID (default: TWINRAG_LLM_MODEL).")
    parser.add_argument("--name", default=None, help="Run directory name.")
    parser.add_argument("--per-type", type=int, default=None, help="Only the first N scenarios of each fault type.")
    parser.add_argument("--only", default=None, help="Only scenarios whose name contains this text.")
    parser.add_argument("--no-graph", action="store_true", help="LLM ablation: alarms only, no knowledge graph.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="LLM only: seconds between incidents (free tiers: ~45 stays under 8k tokens/min).")
    args = parser.parse_args()

    evidence_dir = Path(args.evidence) if args.evidence else EVIDENCE_DIR
    if not evidence_dir.is_absolute():
        evidence_dir = PROJECT_ROOT / evidence_dir

    summary_path = evidence_dir / "retrieval_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"No evidence packets in {evidence_dir}. Run scripts/build_evidence_packets.py first.")

    kg = BuildingKnowledgeGraph.from_files(NETWORK_PATH, LAYOUT_PATH)
    retrieval = pd.read_csv(summary_path)
    retrieval = retrieval[retrieval["detected"]]

    if args.only:
        retrieval = retrieval[retrieval["scenario"].str.contains(args.only, regex=False)]
    if args.per_type:
        retrieval = retrieval.groupby("fault_type", group_keys=False).head(args.per_type)

    if args.engine == "rules":
        if args.no_graph:
            raise SystemExit("--no-graph applies to --engine llm; the rules need the graph.")
        client, diagnoser = None, RuleBasedDiagnoser()
        usage, label = Usage(), "rule-based"
        name = args.name or "rules"
    else:
        client = ChatClient.from_env(model=args.model)
        # The no-graph run gets no repair turn: citing an ID it was never
        # shown is exactly the hallucination the comparison should count.
        diagnoser = Diagnoser(client, repair_attempts=0 if args.no_graph else 1)
        usage, label = client.usage, f"{client.model} via {client.base_url}"
        name = args.name or (client.model.split("/")[-1] + ("-nograph" if args.no_graph else ""))

    out_dir = evidence_dir.parent / "diagnosis" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{label}  |  {len(retrieval)} incident(s)  |  "
          f"{'NO GRAPH (alarms only)' if args.no_graph else 'graph-constrained'}\n")

    rows = []
    started = time.time()

    for i, (_, row) in enumerate(retrieval.iterrows()):
        if i and args.delay and args.engine == "llm":
            time.sleep(args.delay)

        scenario, target = row["scenario"], row["target_id"]
        packet = json.loads((evidence_dir / f"{scenario}.json").read_text(encoding="utf-8"))
        if args.no_graph:
            packet = ungrounded_packet(packet)

        result = diagnoser.diagnose(packet)
        d = result.diagnosis or {}
        predicted = d.get("root_cause_asset")
        score = _score(predicted, target, kg)
        baseline = _score(row.get("detector_top1"), target, kg)

        record = {
            "scenario": scenario,
            "fault_type": row["fault_type"],
            "target_id": target,
            "predicted": predicted,
            "predicted_type": d.get("fault_type"),
            "exact": score["exact"],
            "room": score["room"],
            "near": score["near"],
            "type_correct": d.get("fault_type") == row["fault_type"],
            "grounded": result.grounded,
            "violations": "; ".join(result.violations),
            "attempts": result.attempts,
            "confidence": d.get("confidence"),
            # "Confirmed" = confident enough to act on (rules: >= 0.5).
            "confirmed": (d.get("confidence") or 0) >= 0.5,
            "detector_top1": row.get("detector_top1"),
            "detector_exact": baseline["exact"],
            "detector_near": baseline["near"],
            "error": result.error,
        }
        rows.append(record)

        (out_dir / f"{scenario}.json").write_text(
            json.dumps({"score": record, "result": result.to_dict()}, indent=1), encoding="utf-8"
        )

        tick = "OK " if score["exact"] else ("~  " if score["near"] else "-- ")
        print(f"{tick}{scenario:38s} -> {str(predicted):14s} {str(d.get('fault_type')):12s} "
              f"conf={d.get('confidence')}  grounded={result.grounded}"
              + (f"  ERROR {result.error[:80]}" if result.error else ""))
        if args.show_messages and d.get("message"):
            print("    " + format_message(d["message"]).replace("\n", "\n    ") + "\n")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("\n                 exact  room   near   type   grounded | detector top-1 exact  near")
    for group_label, group in [("all", summary)] + list(summary.groupby("fault_type")):
        print(f"  {group_label:13s} {group['exact'].mean():5.0%}  {group['room'].mean():5.0%}  "
              f"{group['near'].mean():5.0%}  {group['type_correct'].mean():5.0%}  "
              f"{group['grounded'].mean():6.0%}   | "
              f"{group['detector_exact'].mean():18.0%}  {group['detector_near'].mean():5.0%}   (n={len(group)})")

    confirmed = summary[summary["confirmed"]]
    print(f"\n  confirmed diagnoses: {len(confirmed)}/{len(summary)}; exact among confirmed: "
          f"{confirmed['exact'].mean() if len(confirmed) else 0:.0%}; "
          f"unconfirmed flagged for review: {len(summary) - len(confirmed)}")

    run = {
        "engine": args.engine,
        "model": client.model if client else None,
        "base_url": client.base_url if client else None,
        "graph": not args.no_graph,
        "evidence": str(evidence_dir.relative_to(PROJECT_ROOT)),
        "incidents": len(summary),
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")

    cost = (f"{usage.calls} calls, {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens, "
            if client else "no API calls, ")
    print(f"\n{cost}{run['seconds']} s  ->  {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
