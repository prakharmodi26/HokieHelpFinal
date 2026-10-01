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

import logging
import re
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)


def _compile_patterns(patterns: list[str], *, kind: str) -> list[re.Pattern]:
    """Compile regex patterns, skipping (and warning on) any that are malformed.

    A single bad pattern must never crash the crawler. The blocklist is a
    noise-reduction layer, not a security boundary (scope/affiliation gates are
    the real boundary), so dropping one unparseable pattern — loudly — is the
    correct degrade: the crawler keeps running and that one rule simply does not
    apply. Sources include CRAWL_BLOCKED_PATHS (operator-supplied env), so a typo
    there can no longer take the whole run down with a traceback.
    """
    compiled: list[re.Pattern] = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error as exc:
            logger.warning("Skipping invalid %s pattern %r: %s", kind, p, exc)
    return compiled


class HostFilter:
    def __init__(
        self,
        patterns: list[str],
        hosts: list[str],
        path_patterns: list[str] | None = None,
    ) -> None:
        self._patterns = _compile_patterns(patterns, kind="host")
        self._hosts = set(h.lower() for h in hosts)
        self._path_patterns = _compile_patterns(path_patterns or [], kind="path")

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
