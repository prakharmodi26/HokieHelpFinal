"""Loader and matcher for the VT-affiliated hosts allowlist.

Reads ``config/vt-affiliated-hosts.yaml`` (or a custom path via
``VT_AFFILIATED_HOSTS_PATH``) and exposes an ``AffiliatedHostsMatcher``
that ``classify_url`` can use to decide whether a non-vt.edu host is
still VT-affiliated content.

Schema (per-entry):
  - ``domain: <str>`` with ``wildcard_subdomains: true`` — matches the
    apex domain OR any subdomain (host.endswith("." + domain)).
  - ``host: <str>`` — exact hostname match.
  - ``max_urls: int`` — per-host URL cap (mirroring host-registry.yaml).
  - ``path_exclude: [<prefix>, ...]`` — path-prefix blocklist (not regex).
"""
from __future__ import annotations

from urllib.parse import urlparse

import yaml


class AffiliatedHostsMatcher:
    """Immutable matcher built from a list of allowlist entry dicts.

    Accepts both the YAML-parsed list and inline dicts for testing.
    """

    def __init__(self, entries: list[dict]) -> None:
        # exact-match set (host: entries)
        self._exact: set[str] = set()
        # wildcard domain suffixes (domain: entries with wildcard_subdomains)
        self._wildcard_domains: list[str] = []
        # per-host caps and path exclusions stored for wiring into engine
        self._caps: dict[str, int] = {}
        self._path_excludes: dict[str, list[str]] = {}

        for e in entries:
            max_urls = int(e.get("max_urls", -1))
            path_exclude = list(e.get("path_exclude") or [])

            if "domain" in e and e.get("wildcard_subdomains"):
                domain = e["domain"].lower()
                self._wildcard_domains.append(domain)
                # Cap and path_excludes are per-entry; for wildcard domains the
                # cap applies to each matched subdomain individually (engine
                # tracks by exact host, so the wildcard cap is advisory here).
                self._caps[domain] = max_urls
                if path_exclude:
                    self._path_excludes[domain] = path_exclude
            elif "host" in e:
                host = e["host"].lower()
                self._exact.add(host)
                self._caps[host] = max_urls
                if path_exclude:
                    self._path_excludes[host] = path_exclude

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_affiliated(self, url: str) -> bool:
        """Return True if *url*'s hostname is an affiliated non-vt.edu host."""
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        if host in self._exact:
            return True
        for domain in self._wildcard_domains:
            if host == domain or host.endswith("." + domain):
                return True
        return False

    def host_cap(self, host: str) -> int | None:
        """Return the per-host cap for *host*, or None if not in allowlist."""
        host = host.lower()
        if host in self._caps:
            return self._caps[host]
        for domain in self._wildcard_domains:
            if host == domain or host.endswith("." + domain):
                return self._caps.get(domain)
        return None

    def path_excludes_for(self, host: str) -> list[str]:
        """Return the path-exclude prefixes for *host* (empty list if none)."""
        host = host.lower()
        if host in self._path_excludes:
            return self._path_excludes[host]
        for domain in self._wildcard_domains:
            if host == domain or host.endswith("." + domain):
                return self._path_excludes.get(domain, [])
        return []

    def host_caps_dict(self) -> dict[str, int]:
        """Return all explicit caps as a flat dict (for merging into engine caps)."""
        return dict(self._caps)


def load_affiliated_matcher(path: str) -> AffiliatedHostsMatcher:
    """Load the YAML allowlist file and return a ready matcher.

    Silently returns an empty matcher if the file does not exist or is empty,
    so the crawler degrades gracefully in minimal environments.
    """
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return AffiliatedHostsMatcher([])
    return AffiliatedHostsMatcher(data.get("hosts", []) or [])
