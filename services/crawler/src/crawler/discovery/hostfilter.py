"""Host blocklist filter — consumed by the production crawl (patterns + explicit hosts).

Filters on two axes:
  * host — explicit ``hosts:`` list + regex ``patterns:`` (matched against the hostname).
  * path — ``path_patterns`` regexes matched against the URL path+query. These drop
    archive/pagination traps (e.g. ``/playlists/track/``) WITHOUT blocking the host's
    front page, which a host-level block would also kill.

Path patterns come from two merged sources: a ``path_patterns:`` key in the blocklist
YAML and the ``CRAWL_BLOCKED_PATHS`` env list (injected by the caller via from_yaml).
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import yaml


class HostFilter:
    def __init__(
        self,
        patterns: list[str],
        hosts: list[str],
        path_patterns: list[str] | None = None,
    ) -> None:
        self._patterns = [re.compile(p) for p in patterns]
        self._hosts = set(h.lower() for h in hosts)
        self._path_patterns = [re.compile(p) for p in (path_patterns or [])]

    def is_blocked(self, host: str) -> bool:
        host = (host or "").lower()
        if host in self._hosts:
            return True
        return any(p.search(host) for p in self._patterns)

    def is_path_blocked(self, url: str) -> bool:
        parsed = urlparse(url)
        target = parsed.path + (("?" + parsed.query) if parsed.query else "")
        return any(p.search(target) for p in self._path_patterns)

    def is_blocked_url(self, url: str) -> bool:
        """True if either the host OR the URL path is blocked."""
        host = (urlparse(url).hostname or "").lower()
        return self.is_blocked(host) or self.is_path_blocked(url)

    @classmethod
    def from_yaml(cls, path: str, extra_path_patterns: list[str] | None = None) -> "HostFilter":
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        path_patterns = list(data.get("path_patterns", []))
        if extra_path_patterns:
            path_patterns.extend(extra_path_patterns)
        return cls(
            patterns=data.get("patterns", []),
            hosts=data.get("hosts", []),
            path_patterns=path_patterns,
        )
