"""
Graph-constrained diagnosis (Phase 5).

``Diagnoser.diagnose(packet)`` asks the LLM for a structured root cause,
then enforces grounding: every asset the answer names must be in the
packet's ``allowed_ids``. A reply that fails to parse or cites an unknown
ID is sent back once with the specific problems; if it still fails, the
result is kept but marked ungrounded, so evaluation can count it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .llm import ChatClient, LLMError, extract_json
from .prompt import DIAGNOSIS_SCHEMA, FAULT_TYPES, SYSTEM_PROMPT, user_message


@dataclass
class DiagnosisResult:
    diagnosis: dict | None
    grounded: bool
    violations: list = field(default_factory=list)
    attempts: int = 0
    raw: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def cited_ids(diagnosis: dict) -> list:
    ids = [diagnosis.get("root_cause_asset")]
    ids += list(diagnosis.get("affected_assets") or [])
    for step in diagnosis.get("reasoning") or []:
        ids += list(step.get("evidence") or [])
    for alt in diagnosis.get("alternatives") or []:
        ids.append(alt.get("asset"))
    return [i for i in ids if i]


def check_grounding(diagnosis: dict, allowed_ids) -> list:
    """
    Problems with a diagnosis, as human-readable strings (empty = valid).
    """

    allowed = set(allowed_ids)
    problems = []

    for key in DIAGNOSIS_SCHEMA["required"]:
        if key not in diagnosis:
            problems.append(f"missing field '{key}'")

    root = diagnosis.get("root_cause_asset")
    if root and root not in allowed:
        problems.append(f"root_cause_asset '{root}' does not appear in the evidence packet")

    unknown = sorted({i for i in cited_ids(diagnosis) if i not in allowed} - {root})
    if unknown:
        problems.append("cited IDs that do not appear in the evidence packet: " + ", ".join(unknown[:15]))

    if diagnosis.get("fault_type") not in FAULT_TYPES:
        problems.append(f"fault_type must be one of {', '.join(FAULT_TYPES)}")

    confidence = diagnosis.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        problems.append("confidence must be a number between 0 and 1")

    return problems


class Diagnoser:
    def __init__(self, client: ChatClient, repair_attempts: int = 1):
        self.client = client
        self.repair_attempts = repair_attempts

    def diagnose(self, packet: dict, system_prompt: str = SYSTEM_PROMPT) -> DiagnosisResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message(packet)},
        ]

        result = DiagnosisResult(diagnosis=None, grounded=False)

        for attempt in range(self.repair_attempts + 1):
            result.attempts = attempt + 1

            try:
                raw = self.client.chat(messages, json_schema=DIAGNOSIS_SCHEMA)
            except LLMError as error:
                result.error = str(error)
                return result

            result.raw = raw

            try:
                diagnosis = extract_json(raw)
            except ValueError as error:
                problems = [f"reply was not valid JSON ({error})"]
                diagnosis = None
            else:
                problems = check_grounding(diagnosis, packet["allowed_ids"])

            result.diagnosis = diagnosis
            result.violations = problems

            if not problems:
                result.grounded = True
                return result

            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your answer has these problems:\n- "
                        + "\n- ".join(problems)
                        + "\nUse only IDs that appear in the evidence packet. Reply with the corrected JSON object only."
                    ),
                },
            ]

        return result
