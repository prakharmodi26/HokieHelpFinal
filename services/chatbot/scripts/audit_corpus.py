#!/usr/bin/env python3
"""Read-only source-to-index regression audit; no models, crawling, or index writes.

Run inside an existing admin pod (MinIO credentials stay inside that pod):
  kubectl --context endeavour -n test exec -i ADMIN_POD -- \
    python - --minio-bucket test-crawled-pages-0705 \
    < services/chatbot/scripts/audit_corpus.py > docs/evaluations/corpus-audit.json

Without --minio-bucket, uses only stdlib and Qdrant HTTP. URL lookups cover explicit
http/https, www/non-www and trailing-slash variants, not arbitrary redirects.
Fact presence is diagnostic, not proof that a passage supports an answer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

TARGETS = [
    {
        "id": "cs-catalog-major",
        "url": "https://catalog.vt.edu/undergraduate/college-engineering/computer-science/computer-science-bs/",
        "object_key": "catalog.vt.edu/programs/undergraduate-college-engineering-computer-science-computer-science-bs.md",
    },
    {
        "id": "cs-checksheet-2023-24",
        "url": "https://students.cs.vt.edu/content/dam/website_cs_vt_edu/checksheets/CS-Major-2023-24.pdf",
        "object_key": "students.cs.vt.edu/content/dam/website_cs_vt_edu/checksheets/CS-Major-2023-24.pdf.md",
    },
]
FACTS = {
    "credit_total_123": r"\b123\b",
    "math_1225": r"\bMATH\s*1225\b",
    "math_1226": r"\bMATH\s*1226\b",
    "math_2114": r"\bMATH\s*2114\b",
    "cs_1114": r"\bCS\s*1114\b",
}


def variants(url):
    p = urlsplit(url)
    host = p.netloc.removeprefix("www.")
    return sorted({urlunsplit((scheme, prefix + host, path, p.query, ""))
                   for scheme in ("http", "https") for prefix in ("", "www.")
                   for path in (p.path.rstrip("/"), p.path.rstrip("/") + "/")})


def body(text):
    if text.startswith("---\n"):
        sections = text.split("\n---", 1)
        if len(sections) == 2:
            return sections[1].lstrip()
    return text


def markers(text):
    return {name: bool(re.search(pattern, text, re.I)) for name, pattern in FACTS.items()}


def indexed_chunks(base, collection, urls):
    points, offset = [], None
    while True:
        data = {"filter": {"must": [{"key": "url", "match": {"any": urls}}]},
                "limit": 100, "with_vector": False, "with_payload": True}
        if offset is not None:
            data["offset"] = offset
        request = Request(f"{base.rstrip('/')}/collections/{collection}/points/scroll",
                          data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=20) as response:
            result = json.load(response)["result"]
        points.extend(p["payload"] for p in result["points"])
        offset = result.get("next_page_offset")
        if offset is None or len(points) >= 1000:
            return points, offset is None


def minio_client():
    from minio import Minio
    import urllib3
    endpoint = os.environ["MINIO_ENDPOINT"]
    secure = endpoint.startswith("https://")
    return Minio(endpoint.removeprefix("http://").removeprefix("https://").rstrip("/"),
                 access_key=os.environ["MINIO_ACCESS_KEY"], secret_key=os.environ["MINIO_SECRET_KEY"],
                 secure=secure, http_client=urllib3.PoolManager(
                     timeout=urllib3.Timeout(connect=3, read=15), retries=False))


def audit(args):
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "collection": args.collection, "bucket": args.minio_bucket,
              "limitations": ["Exact URL variants only; not all redirects or corpus copies.",
                              "Presence checks do not verify semantic support or currentness.",
                              "No public-web comparison or retrieval ranking is performed here."],
              "sources": []}
    storage = minio_client() if args.minio_bucket else None
    for target in TARGETS:
        row = dict(target)
        try:
            chunks, complete = indexed_chunks(args.qdrant_url, args.collection, variants(target["url"]))
            row["index"] = {"complete": complete, "chunk_count": len(chunks),
                            "body_markers": markers("\n".join(c.get("text", "") for c in chunks)),
                            "chunks": []}
            for chunk in sorted(chunks, key=lambda c: c.get("chunk_id", "")):
                text = chunk.get("text", "")
                item = {k: chunk[k] for k in ("chunk_id", "url", "title", "program_name", "program_level",
                                              "catalog_edition", "headings_path", "embed_version") if k in chunk}
                item.update(chars=len(text), body_markers=markers(text),
                            credit_excerpts=[text[max(0, m.start()-65):m.end()+65]
                                             for m in list(re.finditer(FACTS["credit_total_123"], text))[:2]])
                row["index"]["chunks"].append(item)
        except Exception as exc:
            row["index"] = {"complete": False, "error_type": type(exc).__name__}
        if storage:
            try:
                response = storage.get_object(args.minio_bucket, target["object_key"])
                try:
                    raw = response.read()
                finally:
                    response.close()
                    response.release_conn()
                text = raw.decode("utf-8")
                row["stored"] = {"present": True, "bytes": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest(),
                                 "body_chars": len(body(text)), "body_markers": markers(body(text))}
            except Exception as exc:
                code = getattr(exc, "code", None)
                row["stored"] = {"present": False if code == "NoSuchKey" else None,
                                 "error_type": type(exc).__name__, "code": code}
        report["sources"].append(row)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument("--collection", default="hokiehelp-test")
    parser.add_argument("--minio-bucket")
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if any("error_type" in s["index"] or not s["index"]["complete"] or
           "error_type" in s.get("stored", {}) for s in result["sources"]):
        raise SystemExit(1)
