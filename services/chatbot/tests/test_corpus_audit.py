"""Offline checks for the read-only corpus diagnostic (not answer-quality acceptance)."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "corpus_audit", Path(__file__).parents[1] / "scripts" / "audit_corpus.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_url_variants_keep_program_path():
    urls = audit.variants("https://catalog.vt.edu/undergraduate/cs/")
    assert len(urls) == 8
    assert "https://catalog.vt.edu/undergraduate/cs/" in urls
    assert "http://www.catalog.vt.edu/undergraduate/cs" in urls
    assert all("undergraduate/cs" in u for u in urls)


def test_metadata_is_not_counted_as_indexed_body():
    text = "---\nrequirements: MATH 2114\n---\nComplete 123 credit hours"
    assert not audit.markers(audit.body(text))["math_2114"]
    assert audit.markers(audit.body(text))["credit_total_123"]
    assert audit.body("No frontmatter") == "No frontmatter"


def test_fact_patterns_do_not_match_longer_numbers():
    assert not audit.markers("11234 MATH 21140 CS 11140")["credit_total_123"]
    assert not audit.markers("MATH 21140")["math_2114"]
    assert audit.markers("MATH\n2114")["math_2114"]


def test_scroll_reads_all_pages_without_vectors_or_writes(monkeypatch):
    requests = []
    responses = [
        {"points": [{"payload": {"text": "first"}}], "next_page_offset": "next"},
        {"points": [{"payload": {"text": "second"}}], "next_page_offset": None},
    ]

    def fake_open(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"result": responses.pop(0)}).encode())

    monkeypatch.setattr(audit, "urlopen", fake_open)
    chunks, complete = audit.indexed_chunks("http://qdrant:6333", "collection", ["https://vt.edu/"])
    assert complete and len(chunks) == 2
    assert all(r.full_url.endswith("/points/scroll") for r in requests)
    assert all(json.loads(r.data)["with_vector"] is False for r in requests)
    assert json.loads(requests[1].data)["offset"] == "next"


def test_scan_cap_does_not_claim_complete_absence(monkeypatch):
    response = {"result": {"points": [{"payload": {"text": "x"}}] * 100,
                           "next_page_offset": "next"}}
    monkeypatch.setattr(audit, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    chunks, complete = audit.indexed_chunks("http://qdrant:6333", "collection", ["https://vt.edu/"])
    assert len(chunks) == 1000
    assert not complete


def test_dependency_error_is_not_reported_as_missing_evidence(monkeypatch):
    def fail(*args):
        raise TimeoutError("sensitive provider details")

    monkeypatch.setattr(audit, "indexed_chunks", fail)
    result = audit.audit(SimpleNamespace(minio_bucket=None, collection="c", qdrant_url="http://qdrant"))
    for source in result["sources"]:
        assert source["index"] == {"complete": False, "error_type": "TimeoutError"}
    assert "sensitive" not in json.dumps(result)


@pytest.mark.parametrize("code, expected", [("NoSuchKey", False), ("AccessDenied", None)])
def test_storage_error_classification(monkeypatch, code, expected):
    class StorageError(Exception):
        pass

    def get(*args):
        error = StorageError("not logged")
        error.code = code
        raise error

    monkeypatch.setattr(audit, "minio_client", lambda: SimpleNamespace(get_object=get))
    monkeypatch.setattr(audit, "indexed_chunks", lambda *a: ([], True))
    result = audit.audit(SimpleNamespace(minio_bucket="b", collection="c", qdrant_url="http://qdrant"))
    assert all(s["stored"]["present"] is expected for s in result["sources"])
