"""Load the mode-structured production host registry and derive seed root URLs.

Schema (config/host-registry.yaml), built by scripts/build_registry.py:
  full:    [host, ...]                         # crawl + ingest text
  catalog: [{host, type?, api?/oai?/rest?}]    # metadata-only cards (Phase 8)
  special: [{host, handler}]                   # non-standard handler (Phase 8)

`handler` values: `dining_tool` (runtime agent tool — NOT crawled),
`courseleaf_catalog` (browser-render + structured course/program ingest).
"""
from __future__ import annotations

import yaml

# handlers that are NOT part of the crawl (handled at query time / out-of-band)
_NON_CRAWL_HANDLERS = {"dining_tool"}


def _load(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _entry_host(e) -> str:
    return e if isinstance(e, str) else e["host"]


def load_modes(path: str) -> dict[str, str]:
    data = _load(path)
    modes: dict[str, str] = {}
    for mode in ("full", "catalog", "special"):
        for e in data.get(mode, []) or []:
            modes[_entry_host(e)] = mode
    return modes


def host_handler(path: str, host: str) -> str | None:
    for e in _load(path).get("special", []) or []:
        if _entry_host(e) == host:
            return e.get("handler") if isinstance(e, dict) else None
    return None


def load_seed_hosts(path: str) -> list[str]:
    """Hosts the crawler seeds: full + catalog + browser-crawled special.
    Excludes runtime-tool handlers (e.g. dining_tool) which are not crawled."""
    data = _load(path)
    hosts: list[str] = list(data.get("full", []) or [])
    hosts += [_entry_host(e) for e in (data.get("catalog", []) or [])]
    for e in data.get("special", []) or []:
        if isinstance(e, dict) and e.get("handler") in _NON_CRAWL_HANDLERS:
            continue
        hosts.append(_entry_host(e))
    return sorted(set(hosts))


def host_root_urls(hosts: list[str]) -> list[str]:
    return [f"https://{h}/" for h in hosts]
