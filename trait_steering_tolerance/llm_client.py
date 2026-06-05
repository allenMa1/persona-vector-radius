from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any


class LLMClientError(RuntimeError):
    pass


@dataclass
class OpenRouterClient:
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    app_name: str = "trait-steering-tolerance"
    max_retries: int = 5

    def _client(self):
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMClientError(
                f"Missing {self.api_key_env}. Set it in the shell; do not paste it into chat."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError("Install the openai package to use OpenRouter.") from exc
        return OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers={
                "X-Title": self.app_name,
            },
        )

    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        client = self._client()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    raise
                delay = min(30.0, 2.0 * (2**attempt))
                time.sleep(delay)
        else:
            raise LLMClientError("OpenRouter request failed.") from last_exc
        content = response.choices[0].message.content
        if content is None:
            raise LLMClientError("Model returned no text content.")
        return content

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        text = self.complete_text(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        return parse_json_object(text)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMClientError(f"Could not parse JSON object from: {text[:200]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise LLMClientError("Expected a JSON object.")
    return value
