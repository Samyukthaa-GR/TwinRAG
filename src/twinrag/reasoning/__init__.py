"""
Graph-constrained LLM root-cause reasoning (Phase 5).

    ChatClient        -- OpenAI-compatible client (Hugging Face by default)
    Diagnoser         -- evidence packet -> structured, grounding-checked diagnosis
    check_grounding   -- the rule: every cited ID must be in the packet
    ungrounded_packet -- the no-graph ablation input
"""

from .diagnosis import DiagnosisResult, Diagnoser, check_grounding, cited_ids
from .llm import ChatClient, LLMError, load_dotenv
from .prompt import DIAGNOSIS_SCHEMA, FAULT_TYPES, SYSTEM_PROMPT, ungrounded_packet

__all__ = [
    "ChatClient",
    "LLMError",
    "load_dotenv",
    "Diagnoser",
    "DiagnosisResult",
    "check_grounding",
    "cited_ids",
    "DIAGNOSIS_SCHEMA",
    "FAULT_TYPES",
    "SYSTEM_PROMPT",
    "ungrounded_packet",
]
