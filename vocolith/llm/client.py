# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""LLM client using any OpenAI-compatible endpoint, or local Ollama."""
from __future__ import annotations
import json
import logging
import os
import re
import time
import random

import httpx
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

log = logging.getLogger(__name__)

# Short provider names -> default base URLs
_PROVIDER_URLS: dict[str, str] = {
    "openai":    "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama":    "http://localhost:11434/v1",
}


def _resolve_from_env() -> tuple[str | None, str | None, str | None]:
    """
    Read LLM credentials from environment variables.

    Priority order:
      1. OPENAI_API_KEY + OPENAI_API_MODEL + OPENAI_API_PROVIDER
         OPENAI_API_PROVIDER can be a full URL (https://...) or a short name
         (openai | anthropic | ollama)
      2. ANTHROPIC_AUTH_TOKEN — treated as a generic API key

    Returns:
        (api_key, model, provider_or_url)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    model   = os.environ.get("OPENAI_API_MODEL") or None
    raw     = os.environ.get("OPENAI_API_PROVIDER") or ""

    provider = raw if (raw.startswith("http://") or raw.startswith("https://")) \
               else raw.lower() or None

    if api_key:
        return api_key, model, provider

    fallback_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if fallback_key:
        return fallback_key, model, provider

    return None, model, provider


class VocolithLLMClient:
    """
    OpenAI-compatible LLM client.

    Credential resolution order:
      1. Config values passed to __init__
      2. OPENAI_API_KEY + OPENAI_API_MODEL + OPENAI_API_PROVIDER env vars
      3. ANTHROPIC_AUTH_TOKEN env var (used as a generic API key)

    OPENAI_API_PROVIDER can be:
      - A full base URL:  https://my-llm-gateway.example.com/
      - A short name:     openai | anthropic | ollama

    Proxy: set via standard HTTP_PROXY / HTTPS_PROXY env vars if needed.
    SSL:   ssl_verify=True by default; set False only for self-signed endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_local: bool = False,
        local_base_url: str = "http://localhost:11434/v1",
        local_model: str | None = None,
        max_tokens: int = 32000,
        temperature: float = 0.3,
        ssl_verify: bool = True,
    ) -> None:
        self.max_tokens  = max_tokens
        self.temperature = temperature

        if use_local:
            self.model = local_model or model
            self._client = OpenAI(
                api_key="ollama",
                base_url=local_base_url,
                http_client=httpx.Client(timeout=300.0),
            )
            log.info("LLM: local endpoint %s  model=%s", local_base_url, self.model)
            return

        env_key, env_model, env_provider = _resolve_from_env()
        if not env_key:
            raise RuntimeError(
                "No LLM API key found. Set one of:\n"
                "  OPENAI_API_KEY       (any OpenAI-compatible provider)\n"
                "  ANTHROPIC_AUTH_TOKEN (used as generic API key)"
            )

        # Model: explicit config > env override of default
        self.model = model
        if env_model and model == "gpt-4o-mini":
            self.model = env_model
            log.info("LLM model set via OPENAI_API_MODEL: %s", self.model)

        # Base URL
        provider = env_provider or "openai"
        if provider.startswith("http://") or provider.startswith("https://"):
            base_url = provider.rstrip("/")
        else:
            base_url = _PROVIDER_URLS.get(provider, _PROVIDER_URLS["openai"])

        # Proxy from environment (HTTP_PROXY / HTTPS_PROXY)
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None

        http_kwargs: dict = {"timeout": 500.0, "verify": ssl_verify}
        if proxy:
            http_kwargs["proxy"] = proxy

        self._client = OpenAI(
            api_key=env_key,
            base_url=base_url,
            http_client=httpx.Client(**http_kwargs),
        )
        log.info("LLM: base_url=%s  model=%s  ssl_verify=%s", base_url, self.model, ssl_verify)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 5,
    ) -> str:
        """Call the LLM with exponential-backoff retry. Returns response text."""
        from openai.types.chat import ChatCompletionMessageParam
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return (response.choices[0].message.content or "").strip()

            except RateLimitError:
                wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                log.warning("Rate limited — waiting %.1fs (%d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)

            except APITimeoutError:
                wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                log.warning("Timeout — retrying in %.1fs (%d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)

            except APIError as exc:
                if attempt < max_retries - 1 and getattr(exc, "status_code", 0) >= 500:
                    wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                    log.warning("API error %s — retrying in %.1fs", exc, wait)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"LLM call failed after {max_retries} attempts")

    def call_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        """Call LLM and parse the JSON response. Strips markdown fences if present."""
        if "json" not in system_prompt.lower():
            system_prompt += (
                "\n\nYou MUST respond with valid JSON only. "
                "No markdown, no explanation outside the JSON object."
            )

        raw = self.call(system_prompt, user_prompt, **kwargs).strip()

        # Strip markdown code fences.
        # The LLM sometimes adds a preamble before the block ("Here is the JSON:
        # ```json ...```"), so check for a fence anywhere in the response, not
        # just at the start.  Use findall and prefer the LAST match — if the LLM
        # emits multiple fences, the final one is most likely the clean JSON output.
        fence_matches = re.findall(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
        if fence_matches:
            raw = fence_matches[-1].strip()

        # Primary parse
        try:
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError(f"Expected JSON object, got {type(result).__name__}")
            return result
        except (json.JSONDecodeError, ValueError) as primary_exc:
            log.warning("Primary JSON parse failed: %s", primary_exc)

        # Recovery: bracket-depth scan for first syntactically complete JSON object.
        # We only accept dicts with at least one key — partial/empty objects are rejected.
        best: dict | None = None
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = raw.find(start_char)
            if start == -1:
                continue
            depth, in_string, escape = 0, False, False
            for i, ch in enumerate(raw[start:], start):
                if escape:                   escape = False; continue
                if ch == "\\" and in_string: escape = True;  continue
                if ch == '"':                in_string = not in_string; continue
                if not in_string:
                    if ch == start_char:  depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0:
                            try:
                                candidate = json.loads(raw[start:i+1])
                                if isinstance(candidate, dict) and len(candidate) > 0:
                                    # Prefer the largest (most complete) candidate
                                    if best is None or len(candidate) > len(best):
                                        best = candidate
                            except json.JSONDecodeError:
                                pass
                            break

        if best is not None:
            log.info("JSON recovered via bracket matching (%d keys).", len(best))
            return best

        raise RuntimeError(
            f"LLM returned unparseable JSON. First 300 chars: {raw[:300]!r}"
        )


def build_client_from_config(cfg) -> VocolithLLMClient:
    """Create a VocolithLLMClient from AppConfig.llm settings."""
    return VocolithLLMClient(
        model=cfg.model,
        use_local=cfg.use_local,
        local_base_url=cfg.local_base_url,
        local_model=cfg.local_model if cfg.use_local else None,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        ssl_verify=cfg.ssl_verify,
    )
