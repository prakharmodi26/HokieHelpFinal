#!/usr/bin/env python3
"""Snapshot a BGE-M3 collection, archive to MinIO, restore to a NEW test collection.

Run inside the cluster with existing admin MinIO credentials in the environment.
Never replaces an existing target; original points and objects remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone

import httpx
from minio import Minio


def run(source: str, target: str, bucket: str, qdrant_url: str) -> dict:
    if source == target or not target.startswith("hokiehelp-quality-"):
        raise ValueError("Target must be a separate hokiehelp-quality-* collection")
    if not bucket.startswith("hokiehelp-quality-"):
        raise ValueError("Use an isolated hokiehelp-quality-* backup bucket")
    with httpx.Client(base_url=qdrant_url, timeout=900) as client:
        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()["result"]

        existing = request("GET", "/collections")
        names = {row["name"] for row in existing["collections"]}
        if target in names:
            raise ValueError("Target already exists; refusing to overwrite")
        info = request("GET", f"/collections/{source}")
        params = info["config"]["params"]
        assert params["vectors"]["dense"]["size"] == 1024
        assert "sparse" in params["sparse_vectors"]
        source_count = request("POST", f"/collections/{source}/points/count", json={"exact": True})["count"]
        endpoint = os.environ["MINIO_ENDPOINT"]
        minio = Minio(endpoint.removeprefix("http://").removeprefix("https://").rstrip("/"),
                      access_key=os.environ["MINIO_ACCESS_KEY"], secret_key=os.environ["MINIO_SECRET_KEY"],
                      secure=endpoint.startswith("https://"))
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)
        snapshot = request("POST", f"/collections/{source}/snapshots")
        name = snapshot["name"]
        download = f"{qdrant_url}/collections/{source}/snapshots/{name}"
        key = f"backups/{source}/{name}"
        with urllib.request.urlopen(download, timeout=900) as stream:
            minio.put_object(bucket, key, stream, length=snapshot["size"],
                             content_type="application/octet-stream", part_size=16 * 1024 * 1024)
        archived = minio.stat_object(bucket, key)
        assert archived.size == snapshot["size"]
        # Loopback is on the Qdrant node: no self-Service ingress exception needed.
        request("PUT", f"/collections/{target}/snapshots/recover", json={
            "location": f"http://127.0.0.1:6333/collections/{source}/snapshots/{name}",
            "priority": "snapshot", "checksum": snapshot.get("checksum"),
        })
        target_count = request("POST", f"/collections/{target}/points/count", json={"exact": True})["count"]
        assert source_count == target_count, "Clone point count differs"
        target_info = request("GET", f"/collections/{target}")
        assert target_info["config"]["params"]["vectors"] == params["vectors"]
        assert target_info["config"]["params"]["sparse_vectors"] == params["sparse_vectors"]
        sample = request("POST", f"/collections/{source}/points/scroll", json={
            "limit": 5, "with_payload": True, "with_vector": True,
        })["points"]
        cloned = request("POST", f"/collections/{target}/points", json={
            "ids": [p["id"] for p in sample], "with_payload": True, "with_vector": True,
        })
        assert sorted(sample, key=lambda p: p["id"]) == sorted(cloned, key=lambda p: p["id"])
        return {"created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
                "target": target, "source_count": source_count, "target_count": target_count,
                "snapshot": snapshot, "backup_bucket": bucket, "backup_key": key,
                "backup_bytes": archived.size, "sample_points_verified": len(sample)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.target, args.bucket, args.qdrant_url), indent=2))
