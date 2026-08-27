"""LLM client via Google GenAI SDK + instructor (development and production)."""

from __future__ import annotations

import time
from typing import TypeVar

import instructor
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from crosscheck.config import get_google_api_key, resolve_llm_models

T = TypeVar("T", bound=BaseModel)

# Avoid indefinite hangs on stalled TLS reads (seen as "waiting for response …").
_HTTP_TIMEOUT_MS = 60_000
_HTTP_RETRY_ATTEMPTS = 2

# Sleep when the API rate-limits us (typically HTTP 429).
_RATE_LIMIT_SLEEP_SECONDS = 62
_RATE_LIMIT_MAX_SLEEPS = 8

# Transient network / server drops (often under load; not always a 429 body).
_TRANSIENT_SLEEP_SECONDS = 15
_TRANSIENT_MAX_RETRIES = 5

_RETRYABLE_KEYWORDS = (
    "rate limit",
    "429",
    "404",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "overloaded",
    "capacity",
    "unavailable",
    "resource_exhausted",
    "deadline",
)

_TRANSIENT_KEYWORDS = (
    "disconnect",
    "disconnected",
    "connection reset",
    "connection aborted",
    "connection error",
    "remoteprotocol",
    "remote protocol",
    "broken pipe",
    "eof occurred",
    "ssl",
    "network",
)

_RATE_LIMIT_KEYWORDS = (
    "429",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "resource exhausted",
    "quota exceeded",
    "quota_exceeded",
    "too many requests",
)

_MODE_KEYWORDS = (
    "json_object",
    "json_schema",
    "response format",
    "structured output",
    "tool",
    "does not support",
)

GENAI_MODES = (
    instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    instructor.Mode.GENAI_JSON,
    instructor.Mode.GENAI_TOOLS,
)


def _mode_label(mode: instructor.Mode) -> str:
    """Short printable name for an instructor mode."""
    name = getattr(mode, "name", None) or str(mode)
    return name.replace("GENAI_", "").lower()


def _exc_preview(exc: Exception, limit: int = 160) -> str:
    """One-line exception preview for progress logs."""
    msg = " ".join(str(exc).split())
    if len(msg) > limit:
        return msg[: limit - 1] + "…"
    return msg


def _exc_chain(exc: BaseException) -> list[BaseException]:
    """Walk exception chain for nested SDK / instructor wrappers."""
    seen: set[int] = set()
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(cur)
        nxt = cur.__cause__ or cur.__context__
        cur = nxt if isinstance(nxt, BaseException) else None
    return out


def _exc_messages(exc: BaseException) -> str:
    return " ".join(str(e).lower() for e in _exc_chain(exc))


def _http_status(exc: Exception) -> int | None:
    """Best-effort HTTP status from SDK / HTTPX-style exceptions."""
    for node in _exc_chain(exc):
        for attr in ("status_code", "code"):
            val = getattr(node, attr, None)
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
        response = getattr(node, "response", None)
        if response is not None:
            val = getattr(response, "status_code", None)
            if isinstance(val, int):
                return val
        details = getattr(node, "details", None)
        if isinstance(details, dict):
            val = details.get("code") or details.get("status_code")
            if isinstance(val, int):
                return val
    return None


def _is_rate_limit(exc: Exception) -> bool:
    """True for HTTP 429 / quota / resource_exhausted style errors."""
    if _http_status(exc) == 429:
        return True
    msg = _exc_messages(exc)
    return any(k in msg for k in _RATE_LIMIT_KEYWORDS)


def _is_transient_network(exc: Exception) -> bool:
    """True for dropped connections / protocol errors (may accompany throttling)."""
    if _is_rate_limit(exc):
        return False
    msg = _exc_messages(exc)
    if any(k in msg for k in _TRANSIENT_KEYWORDS):
        return True
    # httpx.RemoteProtocolError, ConnectionError, etc.
    for node in _exc_chain(exc):
        name = type(node).__name__.lower()
        if "protocol" in name or "connection" in name:
            return True
    return False


def _is_retryable(exc: Exception) -> bool:
    if _is_rate_limit(exc) or _is_transient_network(exc):
        return True
    msg = _exc_messages(exc)
    return any(k in msg for k in _RETRYABLE_KEYWORDS)


def _is_mode_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _MODE_KEYWORDS)


def _is_model_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "unavailable" in msg or "not found" in msg


def _make_genai_client(mode: instructor.Mode) -> instructor.Instructor:
    raw = genai.Client(
        api_key=get_google_api_key(),
        http_options=types.HttpOptions(
            timeout=_HTTP_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=_HTTP_RETRY_ATTEMPTS),
        ),
    )
    return instructor.from_genai(raw, mode=mode)


def complete_structured(
    *,
    response_model: type[T],
    messages: list[dict[str, str]],
    max_retries: int = 1,
) -> tuple[T, str]:
    """Call Gemini with structured output; try ranked models and modes as needed.

    On HTTP 429 / rate-limit errors, sleeps ``_RATE_LIMIT_SLEEP_SECONDS`` and
    retries the same model/mode. Transient disconnects / protocol errors sleep
    ``_TRANSIENT_SLEEP_SECONDS`` and retry (Google sometimes drops the socket
    under load without a 429 body).

    Returns (parsed_model, model_id_used).
    """
    models = resolve_llm_models()
    if not models:
        raise RuntimeError("No LLM models configured")
    primary = models[0]
    last_exc: Exception | None = None
    schema = response_model.__name__
    approx_chars = sum(len(m.get("content", "")) for m in messages)
    rate_limit_sleeps = 0
    transient_retries = 0

    print(
        f"  [llm] start schema={schema} prompt≈{approx_chars:,} chars "
        f"rank={len(models)} models timeout={_HTTP_TIMEOUT_MS // 1000}s",
        flush=True,
    )

    for model in models:
        model_unavailable = False
        for mode in GENAI_MODES:
            mode_name = _mode_label(mode)
            client = _make_genai_client(mode)
            attempt = 0
            while attempt < max_retries:
                try:
                    result = client.chat.completions.create(
                        model=model,
                        response_model=response_model,
                        messages=messages,
                        max_retries=0,
                    )
                    if model != primary or mode_name != _mode_label(GENAI_MODES[0]):
                        print(
                            f"  [llm] ok model={model} mode={mode_name}",
                            flush=True,
                        )
                    else:
                        print(f"  [llm] ok model={model}", flush=True)
                    return result, model
                except (ValidationError, ValueError) as exc:
                    last_exc = exc
                    print(
                        f"  [llm] parse error model={model} mode={mode_name}: "
                        f"{_exc_preview(exc)}",
                        flush=True,
                    )
                    attempt += 1
                    if attempt < max_retries:
                        continue
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_rate_limit(exc):
                        rate_limit_sleeps += 1
                        if rate_limit_sleeps > _RATE_LIMIT_MAX_SLEEPS:
                            print(
                                f"  [llm] rate limit exceeded "
                                f"{_RATE_LIMIT_MAX_SLEEPS} sleeps; giving up "
                                f"model={model} mode={mode_name}: "
                                f"{_exc_preview(exc)}",
                                flush=True,
                            )
                            break
                        print(
                            f"  [llm] rate limit (429); "
                            f"sleep {_RATE_LIMIT_SLEEP_SECONDS}s "
                            f"({rate_limit_sleeps}/{_RATE_LIMIT_MAX_SLEEPS}) "
                            f"then retry model={model} …",
                            flush=True,
                        )
                        time.sleep(_RATE_LIMIT_SLEEP_SECONDS)
                        # Do not consume a parse/retry attempt — retry same call.
                        continue
                    if _is_transient_network(exc):
                        transient_retries += 1
                        if transient_retries > _TRANSIENT_MAX_RETRIES:
                            print(
                                f"  [llm] transient network errors exceeded "
                                f"{_TRANSIENT_MAX_RETRIES} retries; giving up "
                                f"model={model} mode={mode_name}: "
                                f"{_exc_preview(exc)}",
                                flush=True,
                            )
                            break
                        print(
                            f"  [llm] transient network error; "
                            f"sleep {_TRANSIENT_SLEEP_SECONDS}s "
                            f"({transient_retries}/{_TRANSIENT_MAX_RETRIES}) "
                            f"then retry model={model} …",
                            flush=True,
                        )
                        time.sleep(_TRANSIENT_SLEEP_SECONDS)
                        continue
                    if _is_model_unavailable(exc):
                        print(
                            f"  [llm] unavailable model={model}; trying next",
                            flush=True,
                        )
                        model_unavailable = True
                        break
                    if _is_mode_error(exc):
                        break
                    if _is_retryable(exc) and attempt + 1 < max_retries:
                        attempt += 1
                        continue
                    if _is_retryable(exc) or _is_mode_error(exc):
                        break
                    print(f"  [llm] error: {_exc_preview(exc)}", flush=True)
                    raise
            if model_unavailable:
                break

    assert last_exc is not None
    print(f"  [llm] all models failed: {_exc_preview(last_exc)}", flush=True)
    raise last_exc
