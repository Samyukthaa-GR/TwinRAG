"""
A minimal client for any OpenAI-compatible chat-completions API.

Standard library only, so the reasoning layer adds no dependency. The
same client reaches:

    Hugging Face Inference Providers  https://router.huggingface.co/v1   (default)
    Groq (free tier)                  https://api.groq.com/openai/v1
                                      e.g. model openai/gpt-oss-120b; ~8k tokens/min
    a local Ollama                    http://localhost:11434/v1

Configuration comes from the environment, or from a ``.env`` file in the
project root (never committed -- it is gitignored):

    HF_TOKEN               API key for the default Hugging Face endpoint
    TWINRAG_LLM_API_KEY    API key for any other endpoint (wins over HF_TOKEN)
    TWINRAG_LLM_BASE_URL   endpoint base URL
    TWINRAG_LLM_MODEL      model ID, e.g. meta-llama/Llama-3.3-70B-Instruct
    TWINRAG_LLM_REASONING_EFFORT   optional: low / medium / high (reasoning models)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class LLMError(RuntimeError):
    pass


def load_dotenv(path=None) -> dict:
    """
    Read ``KEY=value`` lines from a ``.env`` file into ``os.environ``
    (existing variables win). Returns what was read.
    """

    path = Path(path) if path else Path(__file__).resolve().parents[3] / ".env"
    found = {}

    if not path.exists():
        return found

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        found[key] = value
        os.environ.setdefault(key, value)

    return found


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, usage: dict | None) -> None:
        self.calls += 1
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)


@dataclass
class ChatClient:
    """
    Chat-completions over HTTP. ``temperature`` defaults to 0 so a
    diagnosis run is as repeatable as the provider allows.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    # Reasoning models (gpt-oss, Qwen3) spend output tokens thinking
    # before the JSON, so leave room.
    max_tokens: int = 3000
    timeout: float = 180.0
    retries: int = 2
    rate_limit_waits: int = 8
    #: Sent as ``reasoning_effort`` when set (gpt-oss, some Qwen3 models);
    #: "low" thinks less and spends fewer output tokens.
    reasoning_effort: str | None = None
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_env(cls, model: str | None = None, **overrides) -> "ChatClient":
        load_dotenv()

        api_key = os.environ.get("TWINRAG_LLM_API_KEY") or os.environ.get("HF_TOKEN")
        if not api_key:
            raise LLMError(
                "No API key. Put HF_TOKEN=hf_... in the project's .env file "
                "(or set TWINRAG_LLM_API_KEY for another provider)."
            )

        return cls(
            api_key=api_key,
            model=model or os.environ.get("TWINRAG_LLM_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("TWINRAG_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            reasoning_effort=os.environ.get("TWINRAG_LLM_REASONING_EFFORT") or None,
            **overrides,
        )

    # ------------------------------------------------------------------

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Some gateways' firewalls reject Python's default
                # "Python-urllib" agent with an HTML 403 page.
                "User-Agent": "twinrag/0.1 (+https://github.com/Samyukthaa-GR/TwinRAG)",
            },
        )

        last = None
        rate_limited = 0
        attempt = 0

        while attempt <= self.retries:
            wait = 2 * (attempt + 1)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:500]
                last = LLMError(f"HTTP {error.code}: {detail}")

                if error.code == 429:
                    # Free tiers throttle by tokens per minute; the server
                    # says how long to wait. Waiting is not a failure, so it
                    # has its own, larger budget.
                    rate_limited += 1
                    if rate_limited > self.rate_limit_waits:
                        raise last from None
                    try:
                        wait = float(error.headers.get("retry-after") or 20)
                    except ValueError:
                        wait = 20.0
                    time.sleep(min(max(wait, 1.0), 90.0) + 1.0)
                    continue

                # Other 4xx (bad key, no credit, bad request) will not fix itself.
                if error.code < 500:
                    raise last from None
            except (urllib.error.URLError, TimeoutError) as error:
                last = LLMError(f"Network error: {error}")

            attempt += 1
            time.sleep(wait)

        raise last

    def chat(self, messages: list, json_schema: dict | None = None) -> str:
        """
        One completion. With ``json_schema``, asks the provider for
        schema-constrained output and falls back to plain JSON-in-text if
        the provider rejects the ``response_format``.
        """

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "diagnosis", "schema": json_schema, "strict": False},
            }

        try:
            reply = self._post(body)
        except LLMError as error:
            # 400/422 means the provider rejected response_format; retry as
            # plain text. Anything else (402 no credit, 401 bad key) will not
            # be fixed by dropping the schema.
            if json_schema and str(error).startswith(("HTTP 400", "HTTP 422")):
                body.pop("response_format")
                reply = self._post(body)
            else:
                raise

        self.usage.add(reply.get("usage"))
        return reply["choices"][0]["message"]["content"] or ""


def extract_json(text: str) -> dict:
    """
    Parse a JSON object from a model reply, tolerating code fences and
    prose around it.
    """

    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]

    return json.loads(text)
