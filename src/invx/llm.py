from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, json_schema: dict[str, Any]) -> Completion: ...


class LLMError(Exception):
    """Transport or provider failure: retry or fall back to the next model."""


class OpenAICompatibleLLM:
    """Any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, Groq, vLLM) using structured outputs."""

    def __init__(
        self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0, client: httpx.Client | None = None
    ) -> None:
        self.name = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout_s)

    def complete(self, system: str, user: str, json_schema: dict[str, Any]) -> Completion:
        body = {
            "model": self.name,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "invoice", "schema": json_schema, "strict": False},
            },
        }
        try:
            r = self._client.post(self._url, headers=self._headers, json=body)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        data = r.json()
        usage = data.get("usage", {})
        return Completion(
            data["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            self.name,
        )


def parse_json(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])
