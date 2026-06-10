"""Build the static production host-registry.yaml + host-blocklist.yaml.

One-time build from the offline snapshot + tiering decisions (2026-06-10).
Inputs:
  discovery/hosts/tiering/agent*-*.md     -> per-host mode (full/catalog/drop/special)
  discovery/hosts/tiering/_overrides.json -> owner overrides
  discovery/hosts/vtedu-blocklist.yaml    -> base patterns + dead/infra hosts
Outputs:
  config/host-registry.yaml
  config/host-blocklist.yaml
"""
from __future__ import annotations
import glob, json, os, re
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # services/crawler
HOSTS = os.path.join(HERE, "discovery", "hosts")
TIER = os.path.join(HOSTS, "tiering")
CONFIG = os.path.join(HERE, "config")
os.makedirs(CONFIG, exist_ok=True)

# Owner overrides (final, from review)
OVERRIDES = {
    "engapplets.vt.edu": "drop", "engapplets.aoe.vt.edu": "drop", "www.engapplets.vt.edu": "drop",
    "web-cat.cs.vt.edu": "drop", "web-cat5.cs.vt.edu": "drop",
    "opendsa.cs.vt.edu": "catalog", "opendsa-server.cs.vt.edu": "catalog",
    "genomerxiv.cs.vt.edu": "catalog",
    "codeworkout.cs.vt.edu": "catalog", "geovine.cgit.vt.edu": "catalog",
}

# special handlers (runtime/structured, not normal crawl)
SPECIAL = {
    "dining.vt.edu": "dining_tool",       # runtime agent tool; NOT crawled into corpus
    "www.dining.vt.edu": "dining_tool",
    "catalog.vt.edu": "courseleaf_catalog",  # browser-render + structured parse; current-year; regular refresh
}

# known catalog API hints (from tiering research)
CATALOG_API = {
    "vtechworks.lib.vt.edu": {"type": "dspace", "rest": "/server/api", "oai": "/server/oai/request"},
    "eprints.cs.vt.edu": {"type": "eprints", "oai": "/cgi/oai2"},
    "dataverse.vtti.vt.edu": {"type": "dataverse", "api": "/api"},
    "data.lib.vt.edu": {"type": "dataverse?", "note": "403 on probe; verify manually"},
}


def load_modes() -> dict[str, str]:
    modes: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(TIER, "agent*-*.md"))):
        for line in open(f):
            line = line.strip()
            if not line.startswith("|"):
                continue
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) < 2:
                continue
            host = c[0].strip("`").strip()
            mode = c[1].lower()
            if host.lower() in ("host", "") or set(host) <= set("-: "):
                continue
            if mode not in ("full", "catalog", "drop", "special"):
                continue
            modes.setdefault(host, mode)
    modes.update(OVERRIDES)
    # ensure dining/catalog handlers are special
    for h in SPECIAL:
        modes[h] = "special"
    return modes


def main():
    modes = load_modes()
    full = sorted(h for h, m in modes.items() if m == "full")
    catalog = sorted(h for h, m in modes.items() if m == "catalog")
    special = sorted(h for h, m in modes.items() if m == "special")
    drop = sorted(h for h, m in modes.items() if m == "drop")

    # ---- registry ----
    registry = {
        "_meta": {
            "built": "2026-06-10",
            "source": "offline snapshot (3508 hosts -> 780 keep) + 4-lane tiering + owner review",
            "note": "STATIC. Registry is a SEED SET, not a boundary: the crawler also follows "
                    "any live *.vt.edu host not blocklisted. Rebuild by hand from a fresh snapshot.",
            "modes": {
                "full": "crawl all pages, ingest text into RAG corpus",
                "catalog": "metadata-only cards (no full-text ingest); repositories/archives/project sites",
                "special": "non-standard handler (see 'handler'): dining_tool=runtime agent tool (not crawled); "
                           "courseleaf_catalog=browser-render + structured course/program ingest",
            },
        },
        "full": full,
        "catalog": [
            ({"host": h, **CATALOG_API[h]} if h in CATALOG_API else {"host": h})
            for h in catalog
        ],
        "special": [{"host": h, "handler": SPECIAL.get(h, "review")} for h in special],
    }
    with open(os.path.join(CONFIG, "host-registry.yaml"), "w") as fh:
        yaml.safe_dump(registry, fh, default_flow_style=False, sort_keys=False, width=120)

    # ---- blocklist ----
    base = yaml.safe_load(open(os.path.join(HOSTS, "vtedu-blocklist.yaml"))) or {}
    patterns = base.get("patterns", [])
    block_hosts = set(base.get("hosts", [])) | set(drop)
    # never blocklist a registry host
    keep = set(full) | set(catalog) | set(special)
    block_hosts -= keep
    blocklist = {
        "_meta": {
            "built": "2026-06-10",
            "note": "patterns generalize to NEW junk hosts (kept live in production); "
                    "hosts = explicit dead/infra/auth/transactional + tiering drops.",
        },
        "patterns": patterns,
        "hosts": sorted(block_hosts),
    }
    with open(os.path.join(CONFIG, "host-blocklist.yaml"), "w") as fh:
        yaml.safe_dump(blocklist, fh, default_flow_style=False, sort_keys=False, width=120)

    print(f"registry: full={len(full)} catalog={len(catalog)} special={len(special)}")
    print(f"  special hosts: {[(s['host'], s['handler']) for s in registry['special']]}")
    print(f"  catalog hosts: {catalog}")
    print(f"blocklist: patterns={len(patterns)} hosts={len(block_hosts)} (incl. {len(drop)} tiering drops)")
    print(f"total accounted: {len(full)+len(catalog)+len(special)+len(drop)}")


if __name__ == "__main__":
    main()
