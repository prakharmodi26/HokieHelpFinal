"""Step 1 — Enumerate vt.edu subdomains via Certificate Transparency (crt.sh) + optional Common Crawl."""
import json
import sys
import httpx

OUT_DIR = "discovery/hosts"


def normalize_host(h: str) -> str | None:
    h = h.strip().lower()
    if not h:
        return None
    if h.startswith("*."):
        h = h[2:]
    # strip any leftover whitespace / trailing dots
    h = h.strip().rstrip(".")
    if not h:
        return None
    if h == "vt.edu" or h.endswith(".vt.edu"):
        return h
    return None


def from_crtsh() -> set[str]:
    import time
    hosts: set[str] = set()
    # crt.sh is flaky (502s under load). Try a few query encodings + backoff.
    urls = [
        "https://crt.sh/?q=%25.vt.edu&output=json",
        "https://crt.sh/?q=.vt.edu&output=json",
        "https://crt.sh/json?q=%25.vt.edu",
    ]
    data = None
    headers = {"User-Agent": "HokieHelp-crawler-discovery/0.1"}
    for attempt in range(8):
        url = urls[attempt % len(urls)]
        print(f"[crt.sh] attempt {attempt+1} -> {url}", file=sys.stderr)
        try:
            r = httpx.get(url, timeout=180.0, follow_redirects=True, headers=headers)
            if r.status_code != 200:
                raise RuntimeError(f"status {r.status_code}")
            data = r.json()
            if data:
                break
        except Exception as e:
            print(f"[crt.sh] attempt {attempt+1} failed: {e}", file=sys.stderr)
            data = None
            time.sleep(min(5 + attempt * 5, 30))
    if not data:
        print("[crt.sh] FAILED to fetch after retries", file=sys.stderr)
        return hosts
    for entry in data:
        nv = entry.get("name_value", "") or ""
        for line in nv.split("\n"):
            n = normalize_host(line)
            if n:
                hosts.add(n)
        cn = entry.get("common_name", "") or ""
        n = normalize_host(cn)
        if n:
            hosts.add(n)
    print(f"[crt.sh] {len(hosts)} normalized hosts", file=sys.stderr)
    return hosts


def from_commoncrawl() -> set[str]:
    hosts: set[str] = set()
    url = ("https://index.commoncrawl.org/CC-MAIN-2026-21-index"
           "?url=vt.edu&matchType=domain&output=json&fl=url")
    print(f"[CC] trying {url}", file=sys.stderr)
    try:
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True,
                          headers={"User-Agent": "HokieHelp-crawler-discovery/0.1"}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                u = obj.get("url", "")
                # extract host
                try:
                    host = u.split("//", 1)[1].split("/", 1)[0].split(":")[0]
                except Exception:
                    continue
                n = normalize_host(host)
                if n:
                    hosts.add(n)
    except Exception as e:
        print(f"[CC] skipped (unreachable/slow): {e}", file=sys.stderr)
    print(f"[CC] {len(hosts)} normalized hosts", file=sys.stderr)
    return hosts


def main():
    hosts = from_crtsh()
    if not hosts:
        print("FATAL: crt.sh returned nothing", file=sys.stderr)
        sys.exit(1)
    try:
        cc = from_commoncrawl()
        extra = cc - hosts
        print(f"[CC] {len(extra)} extra hosts beyond crt.sh", file=sys.stderr)
        hosts |= cc
    except Exception as e:
        print(f"[CC] error, skipping: {e}", file=sys.stderr)
    sorted_hosts = sorted(hosts)
    with open(f"{OUT_DIR}/vtedu-subdomains.txt", "w") as f:
        f.write("\n".join(sorted_hosts) + "\n")
    print(f"WROTE {len(sorted_hosts)} hosts to {OUT_DIR}/vtedu-subdomains.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
