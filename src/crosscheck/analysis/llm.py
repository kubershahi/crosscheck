"""LLM client via Google GenAI SDK + instructor (development and production)."""

from __future__ import annotations

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


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
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

    Returns (parsed_model, model_id_used).
    """
    models = resolve_llm_models()
    if not models:
        raise RuntimeError("No LLM models configured")
    primary = models[0]
    last_exc: Exception | None = None
    schema = response_model.__name__
    approx_chars = sum(len(m.get("content", "")) for m in messages)

    print(
        f"  [llm] start schema={schema} prompt≈{approx_chars:,} chars "
        f"rank={len(models)} models timeout={_HTTP_TIMEOUT_MS // 1000}s",
        flush=True,
    )

    for model_idx, model in enumerate(models, start=1):
        model_unavailable = False
        for mode in GENAI_MODES:
            mode_name = _mode_label(mode)
            client = _make_genai_client(mode)
            for attempt in range(max_retries):
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
                    if attempt + 1 < max_retries:
                        continue
                    break
                except Exception as exc:
                    last_exc = exc
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
