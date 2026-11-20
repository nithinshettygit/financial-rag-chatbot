"""Shared Groq call wrapper: honours Groq's "try again in Xs" hint on 429s.

Used by the router, the answer generator and the evaluation judge so a tokens-per-minute
limit causes a short wait instead of a failed request.

Env:
  LLM_MAX_ATTEMPTS        attempts per call
  LLM_MAX_WAIT_S          total seconds we may sleep per call
  LLM_REASONING_EFFORT    low|medium|high, applied to gpt-oss models only
"""
from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from typing import Any, Optional, Tuple

from groq import APIConnectionError, AuthenticationError, Groq, InternalServerError, RateLimitError

from app.config import settings

logger = logging.getLogger(__name__)

_DURATION_RE = re.compile(r"try again in\s*((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.I)
_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.I)
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_CLIENTS_LOCK = threading.Lock()
_CLIENTS_BY_TIMEOUT: dict[float, Tuple[Groq, ...]] = {}


def extract_completion_text(response: Any) -> str:
    """Return non-empty assistant text or fail explicitly on malformed completions."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("Groq returned a malformed completion") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned an empty completion")
    return content.strip()


def get_groq_clients(timeout: float = 30.0) -> Tuple[Groq, ...]:
    """Return cached Groq clients for the configured primary and fallback keys."""
    with _CLIENTS_LOCK:
        if timeout not in _CLIENTS_BY_TIMEOUT:
            keys = tuple(dict.fromkeys(
                key.strip()
                for key in (settings.GROQ_API_KEY, settings.FALLBACK_GROQ_API_KEY)
                if key and key.strip()
            ))
            _CLIENTS_BY_TIMEOUT[timeout] = tuple(
                Groq(api_key=key, timeout=timeout, max_retries=0) for key in keys
            )
        return _CLIENTS_BY_TIMEOUT[timeout]


def parse_retry_after(exc: Exception) -> Optional[float]:
    """Seconds Groq asks us to wait (message text first, then the Retry-After header)."""
    m = _DURATION_RE.search(str(exc))
    if m:
        return sum(float(v) * _UNIT_S[u.lower()] for v, u in _PART_RE.findall(m.group(1)))
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    return None


def chat_with_retry(client: Any, *, max_total_wait_s: Optional[float] = None, **kwargs: Any) -> Any:
    """Blocking chat.completions.create with rate-limit aware retries (run via asyncio.to_thread)."""
    clients = client if isinstance(client, (tuple, list)) else (client,)
    if not clients:
        raise ValueError("At least one Groq client is required")

    max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", str(settings.LLM_MAX_ATTEMPTS)))
    budget = (
        max_total_wait_s
        if max_total_wait_s is not None
        else float(os.getenv("LLM_MAX_WAIT_S", str(settings.LLM_MAX_WAIT_S)))
    )
    effort = os.getenv("LLM_REASONING_EFFORT", settings.LLM_REASONING_EFFORT)
    if effort and "gpt-oss" in str(kwargs.get("model", "")) and "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = effort

    waited = 0.0
    client_index = 0
    rate_limited_clients = set()
    invalid_response_clients = set()
    for attempt in range(1, max_attempts + 1):
        try:
            response = clients[client_index].chat.completions.create(**kwargs)
            try:
                extract_completion_text(response)
            except ValueError:
                logger.warning("Groq returned an empty or malformed completion")
                invalid_response_clients.add(client_index)
                next_index = next(
                    (i for i in range(len(clients)) if i not in invalid_response_clients),
                    None,
                )
                if attempt == max_attempts:
                    raise
                if next_index is not None:
                    client_index = next_index
                else:
                    invalid_response_clients.clear()
                continue
            return response
        except (RateLimitError, AuthenticationError, APIConnectionError, InternalServerError) as exc:
            if isinstance(exc, RateLimitError):
                rate_limited_clients.add(client_index)
                next_index = next(
                    (i for i in range(len(clients)) if i not in rate_limited_clients), None
                )
                if next_index is not None and attempt < max_attempts:
                    logger.warning("Groq rate limit reached; switching to another configured API key")
                    client_index = next_index
                    continue
                rate_limited_clients.clear()
            elif isinstance(exc, AuthenticationError):
                next_index = next((i for i in range(len(clients)) if i != client_index), None)
                if next_index is not None and attempt < max_attempts:
                    logger.warning("Groq request failed; switching to another configured API key")
                    client_index = next_index
                    continue
                raise
            elif isinstance(exc, APIConnectionError):
                next_index = next((i for i in range(len(clients)) if i != client_index), None)
                if next_index is not None and attempt < max_attempts:
                    logger.warning("Groq connection failed; switching to another configured API key")
                    client_index = next_index
                    continue

            hint = parse_retry_after(exc) if isinstance(exc, RateLimitError) else None
            delay = (hint if hint is not None else min(2 ** attempt, 15)) + 0.5 + random.random() * 0.5
            if attempt == max_attempts or waited + delay > budget:
                raise
            logger.warning("%s (attempt %d/%d); sleeping %.1fs", type(exc).__name__, attempt, max_attempts, delay)
            time.sleep(delay)
            waited += delay