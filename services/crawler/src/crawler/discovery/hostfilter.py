"""Host blocklist filter — consumed by the production crawl (patterns + explicit hosts)."""
from __future__ import annotations

import re

import yaml


class HostFilter:
    def __init__(self, patterns: list[str], hosts: list[str]) -> None:
        self._patterns = [re.compile(p) for p in patterns]
        self._hosts = set(h.lower() for h in hosts)

    def is_blocked(self, host: str) -> bool:
        host = (host or "").lower()
        if host in self._hosts:
            return True
        return any(p.search(host) for p in self._patterns)

    @classmethod
    def from_yaml(cls, path: str) -> "HostFilter":
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls(patterns=data.get("patterns", []), hosts=data.get("hosts", []))
