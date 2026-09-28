"""
Phase 5: LLM root-cause diagnosis of the building's evidence packets.

Reads the packets written by build_evidence_packets.py, asks an LLM for a
structured, grounding-checked diagnosis of each, and scores it against
the answer key (read only here, after the diagnosis). Compares with the
detector's own top-ranked suspect.

    python scripts/diagnose_incidents.py --per-type 2          # cheap smoke run
    python scripts/diagnose_incidents.py                        # every packet
    python scripts/diagnose_incidents.py --no-graph --name nograph
    python scripts/diagnose_incidents.py --model Qwen/Qwen2.5-72B-Instruct

The endpoint, key and model come from the project's .env (see
twinrag/reasoning/llm.py). For Groq's free tier:

    TWINRAG_LLM_BASE_URL=https://api.groq.com/openai/v1
    TWINRAG_LLM_API_KEY=gsk_...
    TWINRAG_LLM_MODEL=openai/gpt-oss-120b

Free tiers are throttled (Groq: ~8k tokens/min, ~200k/day), so start
with --per-type and use --delay 40 for longer runs.

Outputs, data/building/diagnosis/<name>/ (gitignored):
    <scenario>.json   diagnosis, grounding result and score
    summary.csv       one row per scenario
    run.json          model, settings, token usage
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

from twinrag.building import load_layout
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.reasoning import ChatClient, Diagnoser, ungrounded_packet


NETWORK_PATH = PROJECT_ROOT / "data" / "networks" / "Building_G6.inp"
LAYOUT_PATH = PROJECT_ROOT / "data" / "networks" / "Building_G6.layout.json"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "building" / "evidence"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "building" / "diagnosis"


def _score(predicted, target, kg) -> dict:
    """
    exact  -- the named asset is the faulted one
    room   -- exact, or another asset in the same room (a leaking tap vs
              the branch pipe feeding it): room-level diagnosis
    near   -- same room, or one pipe away (a closed riser segment vs the
              node just below it)
    """

    if not predicted or predicted not in kg:
        return {"exact": False, "room": False, "near": False}

    exact = predicted == target
    room_p = kg.asset(predicted).get("room")
    room_t = kg.asset(target).get("room")
    same_room = bool(room_p) and room_p == room_t

    def touching(a, b):
        ends = {kg.link_from.get(a), kg.link_to.get(a)}
        return b in ends

    near = exact or same_room or touching(predicted, target) or touching(target, predicted)
    return {"exact": exact, "room": exact or same_room, "near": near}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", default=None, help="Model ID (default: TWINRAG_LLM_MODEL or Llama-3.3-70B).")
    parser.add_argument("--name", default=None, help="Run directory name (default: derived from model).")
    parser.add_argument("--per-type", type=int, default=None, help="Diagnose only the first N scenarios of each fault type.")
    parser.add_argument("--only", default=None, help="Only scenarios whose name contains this text.")
    parser.add_argument("--no-graph", action="store_true", help="Ablation: alarms only, no knowledge graph.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to pause between incidents (free tiers: ~40 keeps under 8k tokens/min).")
    args = parser.parse_args()

    summary_path = EVIDENCE_DIR / "retrieval_summary.csv"
    if not summary_path.exists():
        raise SystemExit("No evidence packets. Run: python scripts/build_evidence_packets.py")

    kg = BuildingKnowledgeGraph.from_files(NETWORK_PATH, LAYOUT_PATH)
    retrieval = pd.read_csv(summary_path)
    retrieval = retrieval[retrieval["detected"]]

    if args.only:
        retrieval = retrieval[retrieval["scenario"].str.contains(args.only, regex=False)]
    if args.per_type:
        retrieval = retrieval.groupby("fault_type", group_keys=False).head(args.per_type)

    client = ChatClient.from_env(model=args.model)
    # The no-graph run gets no repair turn: citing an ID it was never shown
    # is exactly the hallucination the comparison is meant to count.
    diagnoser = Diagnoser(client, repair_attempts=0 if args.no_graph else 1)

    name = args.name or (client.model.split("/")[-1] + ("-nograph" if args.no_graph else ""))
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model {client.model} via {client.base_url}  |  {len(retrieval)} incident(s)  |  "
          f"{'NO GRAPH (alarms only)' if args.no_graph else 'graph-constrained'}\n")

    rows = []
    started = time.time()

    for i, (_, row) in enumerate(retrieval.iterrows()):
        if i and args.delay:
            time.sleep(args.delay)
        scenario, target = row["scenario"], row["target_id"]
        packet = json.loads((EVIDENCE_DIR / f"{scenario}.json").read_text(encoding="utf-8"))
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

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("\n                 exact  room   near   type   grounded | detector top-1 exact  near")
    for label, group in [("all", summary)] + list(summary.groupby("fault_type")):
        print(f"  {label:13s} {group['exact'].mean():5.0%}  {group['room'].mean():5.0%}  {group['near'].mean():5.0%}  "
              f"{group['type_correct'].mean():5.0%}  {group['grounded'].mean():6.0%}   | "
              f"{group['detector_exact'].mean():18.0%}  {group['detector_near'].mean():5.0%}   (n={len(group)})")

    usage = client.usage
    run = {
        "model": client.model,
        "base_url": client.base_url,
        "graph": not args.no_graph,
        "incidents": len(summary),
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")

    print(f"\n{usage.calls} calls, {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens, "
          f"{run['seconds']} s  ->  {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
