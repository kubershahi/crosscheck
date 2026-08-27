"""Read/write JSONL files, including pretty-printed multi-line objects."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def iter_json_objects_text(text: str) -> Iterator[JsonObject]:
    """Yield JSON objects from NDJSON or pretty-printed concatenated objects."""
    text = text.strip()
    if not text:
        return

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and all(
        ln.lstrip().startswith("{") and ln.rstrip().endswith("}") for ln in lines
    ):
        for line in lines:
            yield json.loads(line)
        return

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(
                f"expected JSON object at offset {idx}, got {type(obj).__name__}"
            )
        yield obj
        idx = next_idx


def iter_json_objects(path: Path | str) -> Iterator[JsonObject]:
    """Yield JSON objects from a JSONL file on disk."""
    path = Path(path)
    if not path.exists():
        return
    yield from iter_json_objects_text(path.read_text(encoding="utf-8"))


def is_pretty_json_text(text: str) -> bool:
    """True when objects span multiple lines (pretty-printed)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return not all(
        ln.lstrip().startswith("{") and ln.rstrip().endswith("}") for ln in lines
    )


def resolve_pretty(path: Path, *, pretty: bool | None) -> bool:
    """Resolve output formatting; ``None`` preserves an existing file's style."""
    if pretty is not None:
        return pretty
    if not path.exists() or path.stat().st_size == 0:
        return False
    return is_pretty_json_text(path.read_text(encoding="utf-8"))


def format_json_object(obj: JsonObject, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent="\t") + "\n"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_json_objects(
    path: Path,
    objects: list[JsonObject],
    *,
    pretty: bool | None = None,
) -> None:
    """Rewrite a JSONL file from a list of objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    use_pretty = resolve_pretty(path, pretty=pretty)
    body = "".join(format_json_object(obj, pretty=use_pretty) for obj in objects)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(body)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def append_json_object(
    path: Path,
    obj: JsonObject,
    *,
    pretty: bool | None = None,
) -> None:
    """Append one JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    use_pretty = resolve_pretty(path, pretty=pretty)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(format_json_object(obj, pretty=use_pretty))
        fh.flush()
        os.fsync(fh.fileno())


def drop_json_object_ids(
    path: Path,
    drop: set[str],
    *,
    id_key: str = "claim_id",
    pretty: bool | None = None,
) -> int:
    """Rewrite JSONL excluding rows whose ``id_key`` is in ``drop``."""
    if not path.exists() or not drop:
        return 0
    kept: list[JsonObject] = []
    removed = 0
    for row in iter_json_objects(path):
        cid = row.get(id_key)
        if isinstance(cid, str) and cid in drop:
            removed += 1
            continue
        kept.append(row)
    write_json_objects(path, kept, pretty=pretty)
    return removed