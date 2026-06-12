"""Tests for the Qdrant payload-index migration script."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_payload_indexes.py"
SPEC = importlib.util.spec_from_file_location("migrate_payload_indexes", SCRIPT_PATH)
migrate_payload_indexes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(migrate_payload_indexes)


def test_create_keyword_indexes_creates_expected_fields():
    client = MagicMock()
    migrate_payload_indexes.create_keyword_indexes(client, "test_chunks")
    field_names = [
        call.kwargs["field_name"]
        for call in client.create_payload_index.call_args_list
    ]
    assert field_names == [
        "source_host",
        "subdomain",
        "page_type",
        "subject",
        "course_level",
    ]


def test_create_keyword_indexes_tolerates_already_exists_errors():
    client = MagicMock()
    client.create_payload_index.side_effect = RuntimeError("payload index already exists")
    migrate_payload_indexes.create_keyword_indexes(client, "test_chunks")
    assert client.create_payload_index.call_count == 5
