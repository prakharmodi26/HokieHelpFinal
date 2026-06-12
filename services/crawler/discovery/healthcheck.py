"""Steps 2-4 — Concurrent polite health check, classification, and output generation."""
import asyncio
import csv
import re
import sys
from datetime import date
from html.parser import HTMLParser

import httpx

OUT_DIR = "discovery/hosts"
CONCURRENCY = 40
TIMEOUT = 10.0

# ---- auth host detection (for redirected_to_auth) ----
AUTH_HOST_RE = re.compile(
    r"^(login\.vt\.edu$|cas.*\.vt\.edu$|sso.*\.vt\.edu$|idp.*\.vt\.edu$|shib.*\.vt\.edu$)"
)


def is_auth_host(host: str) -> bool:
    if host == "login.vt.edu":
        return True
    if host.endswith(".middleware.vt.edu"):
        return True
    # cas*, sso*, idp*, shib* as leftmost label
    first = host.split(".")[0]
    if re.match(r"^(cas|sso|idp|shib)", first):
        return True
    return False


# ---- HTML visible-text word counter ----
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "head"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "head"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


def word_count(html: str) -> int:
    try:
        p = TextExtractor()
        p.feed(html)
        text = " ".join(p.parts)
        words = re.findall(r"\b\w+\b", text)
        return len(words)
    except Exception:
        return 0


def is_html(ct: str) -> bool:
    ct = (ct or "").lower()
    return "html" in ct or ct == "" or "text/plain" in ct


# ---- classification ----
NON_HTML_RE = re.compile(r"(json|xml|octet-stream|pdf|zip|csv|javascript|image/|font/)")


def classify(rec: dict) -> tuple[str, str]:
    host = rec["host"]
    first = host.split(".")[0]
    status = rec.get("http_status")
    resolves = rec.get("resolves")
    ct = (rec.get("content_type") or "").lower()

    # dead
    if not resolves or not rec.get("scheme"):
        return "drop", "dead"
    if status is not None and status >= 500:
        return "drop", "dead"

    # auth/identity
    if re.match(r"^(login|sso|cas|idp|shib|2fa|auth)\b", first) or re.match(
        r"^(login|sso|cas|idp|shib|2fa|auth)", first
    ):
        # use word boundary against label
        if re.match(r"^(login|sso|cas|idp|shib|2fa|auth)([^a-z0-9]|$)", first) or first in (
            "login", "sso", "cas", "idp", "shib", "2fa", "auth",
        ) or re.match(r"^(login|sso|cas|idp|shib|2fa|auth)\d*$", first):
            return "drop", "auth/identity"
    if host.endswith(".middleware.vt.edu"):
        return "drop", "auth/identity"
    if rec.get("redirected_to_auth"):
        return "drop", "auth/identity"

    # mail/collab
    if re.match(r"^(mail|webmail|smtp|imap|pop|owa|exchange|mx\d*)\.", host):
        return "drop", "mail/collab"

    # dev/infra/vcs
    if re.match(r"^(git|gitlab|gitea|jenkins|ci|vpn|ftp|ns\d*|dns|ldap|radius|proxy)\b", first) or (
        first in ("git", "gitlab", "gitea", "jenkins", "ci", "vpn", "ftp", "dns", "ldap", "radius", "proxy")
        or re.match(r"^ns\d*$", first)
    ):
        return "drop", "dev/infra/vcs"

    # api / non-HTML
    if host.startswith("api."):
        return "drop", "api/non-HTML"
    if ct and not is_html(ct) and NON_HTML_RE.search(ct):
        return "drop", "api/non-HTML"

    # transactional apps
    if re.match(r"^(canvas|hokiespa|banweb|gobblerconnect|webcheckout|pay)\b", first) or first in (
        "canvas", "hokiespa", "banweb", "gobblerconnect", "webcheckout", "pay",
    ):
        return "drop", "transactional"
    if host.endswith(".banner.vt.edu"):
        return "drop", "transactional"

    # staging/internal
    if "stage-" in host or "staging." in host or "-devl" in host or re.match(
        r"^(dev|test|qa|uat|sandbox)\.", host
    ):
        return "drop", "staging/internal"

    return "keep", ""


async def probe(host: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> dict:
    rec = {
        "host": host,
        "resolves": False,
        "scheme": "",
        "http_status": None,
        "final_url": "",
        "redirected_to_auth": False,
        "content_type": "",
        "server": "",
        "homepage_word_count": 0,
    }
    async with sem:
        for scheme in ("https", "http"):
            url = f"{scheme}://{host}/"
            try:
                r = await client.get(url)
                rec["resolves"] = True
                rec["scheme"] = scheme
                rec["http_status"] = r.status_code
                rec["final_url"] = str(r.url)
                rec["content_type"] = r.headers.get("content-type", "")
                rec["server"] = r.headers.get("server", "")
                final_host = r.url.host or ""
                if final_host and is_auth_host(final_host):
                    rec["redirected_to_auth"] = True
                if is_html(rec["content_type"]):
                    try:
                        rec["homepage_word_count"] = word_count(r.text)
                    except Exception:
                        rec["homepage_word_count"] = 0
                return rec
            except Exception:
                # try next scheme; mark resolves False unless connected
                continue
    return rec


async def run(hosts: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=20)
    timeout = httpx.Timeout(TIMEOUT, connect=TIMEOUT)
    headers = {"User-Agent": "HokieHelp-crawler-discovery/0.1 (polite content registry)"}
    results: list[dict] = []
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout, limits=limits, headers=headers, verify=False
    ) as client:
        tasks = [probe(h, client, sem) for h in hosts]
        done = 0
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            results.append(rec)
            done += 1
            if done % 200 == 0:
                print(f"  ...{done}/{len(hosts)} probed", file=sys.stderr)
    results.sort(key=lambda r: r["host"])
    return results


def write_outputs(results: list[dict]):
    today = date.today().isoformat()

    # classify
    for rec in results:
        cls, reason = classify(rec)
        rec["classification"] = cls
        rec["drop_reason"] = reason

    # CSV
    fields = [
        "host", "resolves", "scheme", "http_status", "final_url",
        "redirected_to_auth", "content_type", "server", "homepage_word_count",
        "classification", "drop_reason",
    ]
    with open(f"{OUT_DIR}/vtedu-hosts-health.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in results:
            w.writerow({k: rec.get(k, "") for k in fields})

    # Blocklist YAML
    patterns = [
        r"^login\\.",
        r"^sso\\.",
        r"^(cas|idp|shib|2fa|auth)\\b",
        r"\\.middleware\\.vt\\.edu$",
        r"^(mail|webmail|smtp|imap|pop|owa|exchange|mx\\d*)\\.",
        r"^(git|gitlab|gitea|jenkins|ci|vpn|ftp|ns\\d*|dns|ldap|radius|proxy)\\b",
        r"^api\\.",
        r"^(canvas|hokiespa|banweb|gobblerconnect|webcheckout|pay)\\b",
        r"\\.banner\\.vt\\.edu$",
        r"stage-",
        r"staging\\.",
        r"-devl",
        r"^(dev|test|qa|uat|sandbox)\\.",
    ]
    # explicit hosts: dropped hosts that are LIVE (resolved) and auth/transactional/api/mail
    explicit_reasons = {"auth/identity", "mail/collab", "api/non-HTML", "transactional", "dev/infra/vcs"}
    explicit_hosts = sorted(
        rec["host"] for rec in results
        if rec["classification"] == "drop"
        and rec["resolves"]
        and rec["drop_reason"] in explicit_reasons
    )
    with open(f"{OUT_DIR}/vtedu-blocklist.yaml", "w") as f:
        f.write(f"# Generated {today} from crt.sh health check. "
                "Patterns are matched against the host.\n")
        f.write("patterns:\n")
        for p in patterns:
            f.write(f'  - "{p}"\n')
        f.write("hosts:   # explicit individual hosts to block "
                "(dead/auth/transactional ones found live)\n")
        if explicit_hosts:
            for h in explicit_hosts:
                f.write(f"  - {h}\n")
        else:
            f.write("  []\n")

    # SUMMARY
    total = len(results)
    resolved = [r for r in results if r["resolves"]]
    live = [r for r in resolved if (r["http_status"] or 0) < 500 and r["http_status"]]
    dead = [r for r in results if r["classification"] == "drop" and r["drop_reason"] == "dead"]
    keep = [r for r in results if r["classification"] == "keep"]
    drop = [r for r in results if r["classification"] == "drop"]

    from collections import Counter
    reason_counts = Counter(r["drop_reason"] for r in drop)

    # high-value keep hosts
    hv_keywords = [
        "dining", "registrar", "provost", "hr.vt", "housing", "lib.vt", "library",
        "admissions", "undergrad", "graduate", "grad.vt", "recsports", "parking",
        "news", "college", "engineering", "science", "business", "agriculture",
        "vetmed", "architecture", "liberalarts", "cnre", "cals", "pamplin",
        "studentaffairs", "studentlife", "financialaid", "bursar", "career",
        "research", "global", "honors", "alumni", "calendar", "facilities",
        "transportation", "police", "health", "counseling",
    ]

    def hv_score(rec):
        h = rec["host"]
        wc = rec["homepage_word_count"] or 0
        score = 0
        for kw in hv_keywords:
            if kw in h:
                score += 100
                break
        score += min(wc, 2000) / 100
        # prefer department-like 2-3 label hosts with content
        return score

    keep_hv = sorted(
        [r for r in keep if r["homepage_word_count"] and r["homepage_word_count"] > 50],
        key=hv_score, reverse=True,
    )
    # dedupe/prioritize keyword matches first, then by word count
    kw_matches = [r for r in keep_hv if any(kw in r["host"] for kw in hv_keywords)]
    seen = set()
    top = []
    for r in kw_matches + keep_hv:
        if r["host"] in seen:
            continue
        seen.add(r["host"])
        top.append(r)
        if len(top) >= 40:
            break

    with open(f"{OUT_DIR}/SUMMARY.md", "w") as f:
        f.write(f"# vt.edu Host Discovery Summary\n\n")
        f.write(f"Generated {today} from crt.sh CT logs (+ Common Crawl host index) "
                "and a polite concurrent health check.\n\n")
        f.write("## Counts\n\n")
        f.write(f"- Total enumerated hosts: **{total}**\n")
        f.write(f"- Resolved (connected at all): **{len(resolved)}**\n")
        f.write(f"- Live (resolved + final status < 500): **{len(live)}**\n")
        f.write(f"- Dead (no DNS / all schemes failed / status >= 500): **{len(dead)}**\n")
        f.write(f"- **KEEP: {len(keep)}**\n")
        f.write(f"- **DROP: {len(drop)}**\n\n")
        f.write("### Drop breakdown by reason\n\n")
        for reason, cnt in reason_counts.most_common():
            f.write(f"- {reason}: {cnt}\n")
        f.write("\n## Top high-value KEEP hosts (seed allowlist)\n\n")
        f.write("These look like crawlable public content sites (dining, registrar, "
                "colleges/departments, library, housing, admissions, news, etc.):\n\n")
        for r in top:
            f.write(f"- `{r['host']}` — {r['homepage_word_count']} words, "
                    f"status {r['http_status']}\n")
        f.write("\n## Notes & surprising findings\n\n")
        # host family analysis
        from collections import Counter as C
        suffixes = C()
        for r in results:
            parts = r["host"].split(".")
            if len(parts) >= 3:
                suffixes[".".join(parts[-3:])] += 1
        big_fams = [(s, c) for s, c in suffixes.most_common(15) if c >= 10]
        f.write("Largest host families (by shared 3-label suffix):\n\n")
        for s, c in big_fams:
            f.write(f"- `*.{s}`: {c} hosts\n")
        f.write("\n")

    return {
        "total": total, "resolved": len(resolved), "live": len(live),
        "dead": len(dead), "keep": len(keep), "drop": len(drop),
        "reasons": dict(reason_counts), "top": [r["host"] for r in top],
    }


def main():
    with open(f"{OUT_DIR}/vtedu-subdomains.txt") as f:
        hosts = [ln.strip() for ln in f if ln.strip()]
    print(f"[health] probing {len(hosts)} hosts (concurrency={CONCURRENCY})", file=sys.stderr)
    results = asyncio.run(run(hosts))
    stats = write_outputs(results)
    print("=== DONE ===", file=sys.stderr)
    print(f"total={stats['total']} resolved={stats['resolved']} live={stats['live']} "
          f"keep={stats['keep']} drop={stats['drop']}", file=sys.stderr)
    print(f"drop reasons: {stats['reasons']}", file=sys.stderr)


if __name__ == "__main__":
    main()
