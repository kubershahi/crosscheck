"""Tests for JSONL helpers (compact and pretty-printed objects)."""

from __future__ import annotations

from pathlib import Path

import pytest

from crosscheck.io.jsonl import (
    append_json_object,
    drop_json_object_ids,
    is_pretty_json_text,
    iter_json_objects,
    iter_json_objects_text,
    write_json_objects,
)


def test_iter_compact_jsonl() -> None:
    text = '{"claim_id":"A_01","n":1}\n{"claim_id":"A_02","n":2}\n'
    rows = list(iter_json_objects_text(text))
    assert [r["claim_id"] for r in rows] == ["A_01", "A_02"]


def test_iter_pretty_jsonl(tmp_path: Path) -> None:
    pretty = (
        '{\n\t"claim_id": "A_01",\n\t"claim": "hello {world}"\n}\n'
        '{\n\t"claim_id": "A_02",\n\t"claim": "second"\n}\n'
    )
    path = tmp_path / "claims.jsonl"
    path.write_text(pretty, encoding="utf-8")
    rows = list(iter_json_objects(path))
    assert [r["claim_id"] for r in rows] == ["A_01", "A_02"]
    assert rows[0]["claim"] == "hello {world}"
    assert is_pretty_json_text(pretty)


def test_write_preserves_pretty_format(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    path.write_text(
        '{\n\t"claim_id": "A_01",\n\t"flag": false\n}\n',
        encoding="utf-8",
    )
    rows = list(iter_json_objects(path))
    rows[0]["flag"] = True
    write_json_objects(path, rows)
    text = path.read_text(encoding="utf-8")
    assert '\t"claim_id"' in text
    assert list(iter_json_objects(path))[0]["flag"] is True


def test_drop_claim_ids_pretty(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        '{\n\t"claim_id": "A_01"\n}\n{\n\t"claim_id": "A_02"\n}\n',
        encoding="utf-8",
    )
    removed = drop_json_object_ids(path, {"A_01"})
    assert removed == 1
    assert [r["claim_id"] for r in iter_json_objects(path)] == ["A_02"]


def test_append_matches_existing_pretty_style(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_json_object(path, {"claim_id": "A_01"}, pretty=True)
    append_json_object(path, {"claim_id": "A_02"})
    text = path.read_text(encoding="utf-8")
    assert '\t"claim_id"' in text
    assert [r["claim_id"] for r in iter_json_objects(path)] == ["A_01", "A_02"]


def test_iter_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert list(iter_json_objects(path)) == []


def test_iter_invalid_raises() -> None:
    with pytest.raises(ValueError, match="expected JSON object"):
        list(iter_json_objects_text('["not", "a", "dict"]'))
