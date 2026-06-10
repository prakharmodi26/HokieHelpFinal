#!/usr/bin/env python3
"""Dev/ops host health check + keep/drop classification for vt.edu subdomains.

Reads an enumerated host list, probes each politely (https then http), records
health + classification, and writes results INCREMENTALLY so a crash keeps
partial progress. Outputs: a health CSV, a proposed blocklist YAML, and a
summary markdown.

    uv run python scripts/healthcheck_hosts.py \
        --in discovery/hosts/vtedu-subdomains.txt \
        --outdir discovery/hosts \
        --concurrency 40
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
from collections import Counter

import httpx
import yaml

# --- classification patterns (host-name based) ---
_DROP_PATTERNS: list[tuple[str, str]] = [
    (r"^(login|sso|cas|idp|shib|2fa|auth)\b", "auth"),
    (r"\.middleware\.vt\.edu$", "auth"),
    (r"^(mail|webmail|smtp|imap|pop|owa|exchange|mx\d*)\.", "mail"),
    (r"^(git|gitlab|gitea|jenkins|ci|vpn|ftp|ns\d*|dns|ldap|radius|proxy)\b", "infra"),
    (r"^api\.", "api"),
    (r"^(canvas|hokiespa|banweb|gobblerconnect|webcheckout|pay)\b", "transactional"),
    (r"\.banner\.vt\.edu$", "transactional"),
    (r"(^|\.)(stage-|staging\.|dev\.|test\.|qa\.|uat\.|sandbox\.)", "staging"),
    (r"-devl", "staging"),
]
_AUTH_HOSTS_RE = re.compile(r"^(login|sso|cas|idp|shib)\b|\.middleware\.vt\.edu$")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def pattern_drop_reason(host: str) -> str | None:
    for pat, reason in _DROP_PATTERNS:
        if re.search(pat, host):
            return reason
    return None


def word_count_from_html(html: str) -> int:
    if not html:
        return 0
    text = _TAG_RE.sub(" ", html)
    text = _WS_RE.sub(" ", text).strip()
    return len(text.split()) if text else 0


FIELDS = [
    "host", "resolves", "scheme", "http_status", "final_url",
    "redirected_to_auth", "content_type", "server", "homepage_word_count",
    "classification", "drop_reason",
]


async def probe(client: httpx.AsyncClient, host: str) -> dict:
    row = {f: "" for f in FIELDS}
    row["host"] = host

    # pattern-based drop is decided regardless of liveness
    pat_reason = pattern_drop_reason(host)

    resp = None
    used_scheme = ""
    for scheme in ("https", "http"):
        try:
            resp = await client.get(f"{scheme}://{host}/", follow_redirects=True)
            used_scheme = scheme
            break
        except Exception:
            continue

    if resp is None:
        row["resolves"] = "False"
        row["classification"] = "drop"
        row["drop_reason"] = pat_reason or "dead"
        return row

    final_host = (resp.url.host or "").lower()
    redirected_to_auth = bool(_AUTH_HOSTS_RE.search(final_host))
    ctype = resp.headers.get("content-type", "")
    is_html = "html" in ctype.lower()

    row.update({
        "resolves": "True",
        "scheme": used_scheme,
        "http_status": str(resp.status_code),
        "final_url": str(resp.url),
        "redirected_to_auth": str(redirected_to_auth),
        "content_type": ctype,
        "server": resp.headers.get("server", ""),
        "homepage_word_count": str(word_count_from_html(resp.text) if is_html else 0),
    })

    reason = None
    if resp.status_code >= 500:
        reason = "dead"
    elif pat_reason:
        reason = pat_reason
    elif redirected_to_auth:
        reason = "auth"
    elif not is_html and ctype:
        reason = "non_html"
    row["classification"] = "drop" if reason else "keep"
    row["drop_reason"] = reason or ""
    return row


async def run(hosts: list[str], outdir: str, concurrency: int) -> None:
    csv_path = f"{outdir}/vtedu-hosts-health.csv"
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    done = 0
    lock = asyncio.Lock()

    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()
    csv_file.flush()

    async with httpx.AsyncClient(
        timeout=10.0, headers={"User-Agent": "HokieHelp-Crawler/2.0 (+vt content index)"},
        limits=httpx.Limits(max_connections=concurrency + 10),
    ) as client:
        async def worker(host: str) -> None:
            nonlocal done
            async with sem:
                row = await probe(client, host)
            async with lock:
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                done += 1
                if done % 100 == 0:
                    print(f"  probed {done}/{len(hosts)}", file=sys.stderr, flush=True)

        await asyncio.gather(*(worker(h) for h in hosts))

    csv_file.close()

    # --- blocklist.yaml ---
    explicit_drop_hosts = sorted(
        r["host"] for r in rows
        if r["classification"] == "drop" and r["drop_reason"] in ("dead", "auth", "transactional", "non_html")
    )
    blocklist = {
        "patterns": [p for p, _ in _DROP_PATTERNS],
        "hosts": explicit_drop_hosts,
    }
    with open(f"{outdir}/vtedu-blocklist.yaml", "w") as fh:
        fh.write("# Generated by healthcheck_hosts.py from crt.sh enumeration.\n")
        fh.write("# patterns are regex matched against the host name.\n")
        yaml.safe_dump(blocklist, fh, default_flow_style=False, sort_keys=False)

    # --- SUMMARY.md ---
    total = len(rows)
    live = sum(1 for r in rows if r["resolves"] == "True")
    keep = [r for r in rows if r["classification"] == "keep"]
    drop_reasons = Counter(r["drop_reason"] for r in rows if r["classification"] == "drop")

    high_value_kw = ("dining", "registrar", "provost", "hr.", "housing", "lib.",
                     "admission", "undergrad", "graduate", "recsports", "parking",
                     "news", "advising", "career", "finaid", "bursar", "studentaffairs")
    high_value = sorted(
        r["host"] for r in keep
        if any(k in r["host"] for k in high_value_kw)
    )[:40]

    with open(f"{outdir}/SUMMARY.md", "w") as fh:
        fh.write("# vt.edu Subdomain Health Check — Summary\n\n")
        fh.write(f"- Total enumerated: {total}\n")
        fh.write(f"- Resolved/live: {live}\n")
        fh.write(f"- KEEP (content hosts): {len(keep)}\n")
        fh.write(f"- DROP: {total - len(keep)}\n\n")
        fh.write("## Drop reasons\n\n")
        for reason, count in drop_reasons.most_common():
            fh.write(f"- {reason}: {count}\n")
        fh.write("\n## Suggested high-value allowlist (bounded validation)\n\n")
        for h in high_value:
            fh.write(f"- {h}\n")

    print(f"DONE total={total} live={live} keep={len(keep)} drop={total - len(keep)}",
          file=sys.stderr, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--outdir", default="discovery/hosts")
    p.add_argument("--concurrency", type=int, default=40)
    args = p.parse_args()
    with open(args.infile) as fh:
        hosts = sorted({line.strip().lower() for line in fh if line.strip()})
    print(f"Probing {len(hosts)} hosts (concurrency={args.concurrency})...",
          file=sys.stderr, flush=True)
    asyncio.run(run(hosts, args.outdir, args.concurrency))


if __name__ == "__main__":
    main()
