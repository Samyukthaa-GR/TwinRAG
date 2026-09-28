"""
The reasoning layer without a network: a scripted fake client stands in
for the LLM, so these pin the grounding rules, the repair turn, JSON
extraction and the no-graph ablation input.
"""

import json

import pytest

from twinrag.reasoning import Diagnoser, check_grounding, ungrounded_packet
from twinrag.reasoning.llm import Usage, extract_json


PACKET = {
    "incident": {"id": "INC-1", "first_alarm": "08:00"},
    "building": {"name": "B", "relation_meanings": {"FEEDS": "..."}},
    "observations": [
        {"sensor": "sensor:FM-F3A-KIT-BR", "asset": "F3A-KIT-BR", "parameter": "flowrate",
         "status": "alarm", "reading": "0.200 L/s above the twin"},
        {"sensor": "sensor:FM-F3A-UTL-BR", "asset": "F3A-UTL-BR", "parameter": "flowrate",
         "status": "normal", "reading": "within sensor noise of the twin"},
    ],
    "assets": [{"id": "F3A-KIT-BR", "label": "Flat 3A kitchen branch"},
               {"id": "F3A-KIT", "label": "Flat 3A kitchen"},
               {"id": "F3A-UTL-BR", "label": "Flat 3A utility branch"}],
    "relations": [["F3A-KIT-BR", "FEEDS", "F3A-KIT"]],
    "topology_facts": {"extra_flow": {"deepest_assets_with_extra_flow": ["F3A-KIT-BR"]}},
    "allowed_ids": ["F3A-KIT", "F3A-KIT-BR", "F3A-UTL-BR", "sensor:FM-F3A-KIT-BR", "sensor:FM-F3A-UTL-BR"],
}


def _answer(root="F3A-KIT", evidence=("F3A-KIT-BR",), **extra):
    answer = {
        "root_cause_asset": root,
        "fault_type": "leak",
        "location": "Floor 3, flat 3A, kitchen",
        "affected_assets": [root],
        "reasoning": [{"step": "Extra flow ends at the kitchen branch.", "evidence": list(evidence)}],
        "confidence": 0.8,
        "recommended_action": "Inspect the kitchen tap and branch.",
    }
    answer.update(extra)
    return answer


class FakeClient:
    """Replays scripted replies and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []
        self.usage = Usage()

    def chat(self, messages, json_schema=None):
        self.sent.append([dict(m) for m in messages])
        return self.replies.pop(0)


def test_grounded_answer_passes():
    assert check_grounding(_answer(), PACKET["allowed_ids"]) == []


def test_invented_ids_are_violations():
    problems = check_grounding(_answer(root="F3A-TAP-99", evidence=("DTA-9",)), PACKET["allowed_ids"])

    assert any("root_cause_asset 'F3A-TAP-99'" in p for p in problems)
    assert any("DTA-9" in p for p in problems)


def test_bad_type_and_confidence_are_violations():
    problems = check_grounding(_answer(fault_type="burst", confidence=1.7), PACKET["allowed_ids"])

    assert any("fault_type" in p for p in problems)
    assert any("confidence" in p for p in problems)


def test_json_is_extracted_from_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here: {"a": {"b": 2}} done') == {"a": {"b": 2}}


def test_repair_turn_fixes_an_ungrounded_answer():
    bad = json.dumps(_answer(root="F3A-SINK"))
    good = json.dumps(_answer())
    client = FakeClient(bad, good)

    result = Diagnoser(client).diagnose(PACKET)

    assert result.grounded and result.attempts == 2
    assert result.diagnosis["root_cause_asset"] == "F3A-KIT"
    # the second request tells the model exactly what was wrong
    assert "F3A-SINK" in client.sent[1][-1]["content"]


def test_unrepaired_answer_is_kept_but_marked():
    bad = json.dumps(_answer(root="F3A-SINK"))
    result = Diagnoser(FakeClient(bad), repair_attempts=0).diagnose(PACKET)

    assert not result.grounded
    assert result.diagnosis["root_cause_asset"] == "F3A-SINK"
    assert result.violations


def test_no_graph_input_keeps_alarms_only():
    stripped = ungrounded_packet(PACKET)

    assert "relations" not in stripped and "topology_facts" not in stripped
    assert [a["asset"] for a in stripped["alarms"]] == ["F3A-KIT-BR"]
    assert stripped["allowed_ids"] == ["F3A-KIT-BR"]
    assert "relation_meanings" not in stripped["building"]
