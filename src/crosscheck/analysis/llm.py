"""LLM client via Google GenAI SDK + instructor (development and production)."""

from __future__ import annotations

from typing import TypeVar

import instructor
from google import genai
from pydantic import BaseModel, ValidationError

from crosscheck.config import get_google_api_key, resolve_llm_models

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_KEYWORDS = (
    "rate limit",
    "429",
    "404",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "overloaded",
    "capacity",
    "unavailable",
    "resource_exhausted",
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
    raw = genai.Client(api_key=get_google_api_key())
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
        f"rank={len(models)} models",
        flush=True,
    )

    for model_idx, model in enumerate(models, start=1):
        model_unavailable = False
        print(
            f"  [llm] ({model_idx}/{len(models)}) trying model={model}",
            flush=True,
        )
        for mode in GENAI_MODES:
            mode_name = _mode_label(mode)
            client = _make_genai_client(mode)
            for attempt in range(max_retries):
                print(
                    f"  [llm]   → mode={mode_name} attempt={attempt + 1}/{max_retries} "
                    f"waiting for response …",
                    flush=True,
                )
                try:
                    result = client.chat.completions.create(
                        model=model,
                        response_model=response_model,
                        messages=messages,
                        max_retries=0,
                    )
                    if model != primary:
                        print(
                            f"  [llm] ok via fallback model={model} mode={mode_name}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [llm] ok model={model} mode={mode_name}",
                            flush=True,
                        )
                    return result, model
                except (ValidationError, ValueError) as exc:
                    last_exc = exc
                    print(
                        f"  [llm]   ✗ parse/validation: {_exc_preview(exc)}",
                        flush=True,
                    )
                    if attempt + 1 < max_retries:
                        print("  [llm]   retrying same mode …", flush=True)
                        continue
                    break
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"  [llm]   ✗ {_exc_preview(exc)}",
                        flush=True,
                    )
                    if _is_model_unavailable(exc):
                        print(
                            f"  [llm]   model unavailable, skipping remaining modes",
                            flush=True,
                        )
                        model_unavailable = True
                        break
                    if _is_mode_error(exc):
                        print(
                            f"  [llm]   mode unsupported, trying next mode …",
                            flush=True,
                        )
                        break
                    if _is_retryable(exc) and attempt + 1 < max_retries:
                        print("  [llm]   retryable error, retrying …", flush=True)
                        continue
                    if _is_retryable(exc) or _is_mode_error(exc):
                        break
                    raise
            if model_unavailable:
                break

        if model != models[-1]:
            print(
                f"  [llm] {model} failed, trying next ranked model …",
                flush=True,
            )

    assert last_exc is not None
    print(f"  [llm] all models failed; last error: {_exc_preview(last_exc)}", flush=True)
    raise last_exc
