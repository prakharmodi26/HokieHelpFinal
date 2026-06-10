import httpx, re, json, sys

HOSTS = [
    # repositories / ETD
    "vtechworks.lib.vt.edu", "etd.vt.edu", "etd.lib.vt.edu", "theses.lib.vt.edu",
    "scholar.lib.vt.edu", "gromit.dlib.vt.edu",
    # library service/info
    "lib.vt.edu", "data.lib.vt.edu", "datavizstudio.lib.vt.edu", "designstudio.lib.vt.edu",
    "info-skills.lib.vt.edu", "informaticslab.lib.vt.edu", "news.lib.vt.edu",
    "spec.lib.vt.edu", "lts.lib.vt.edu", "dsp.lib.vt.edu", "opentext.lib.vt.edu",
    "cdrs.lib.vt.edu", "dice.lib.vt.edu", "kiosk.lib.vt.edu", "tour.lib.vt.edu",
    # exhibits / collections (likely drop)
    "exhibits.lib.vt.edu", "omeka.lib.vt.edu", "dhomeka.lib.vt.edu", "imagebase.lib.vt.edu",
    "civilwar.vt.edu", "flu1918.lib.vt.edu", "vpiworldwarone.lib.vt.edu",
    "roanokevalleygreatwar.lib.vt.edu", "memalpha.lib.vt.edu", "odyssey.lib.vt.edu",
    "communitychange.ipg.vt.edu", "pharos.lib.vt.edu",
]

UA = {"User-Agent": "HokieHelp-Discovery/1.0 (polite; VT content crawler research)"}

META_KEYS = [
    "citation_title", "citation_author", "citation_date", "citation_publication_date",
    "citation_pdf_url", "DC.title", "DC.creator", "DC.date", "dc.title", "dc.creator",
    "og:title", "og:type",
]

def get(client, url):
    try:
        r = client.get(url, headers=UA, follow_redirects=True, timeout=10.0)
        return r
    except Exception as e:
        return e

def extract_meta_lines(html):
    found = {}
    samples = []
    # meta tags
    for m in re.finditer(r'<meta[^>]+>', html, re.I):
        tag = m.group(0)
        nm = re.search(r'name\s*=\s*["\']([^"\']+)["\']', tag, re.I)
        pr = re.search(r'property\s*=\s*["\']([^"\']+)["\']', tag, re.I)
        key = (nm.group(1) if nm else (pr.group(1) if pr else None))
        if not key:
            continue
        for k in META_KEYS:
            if key.lower() == k.lower():
                found[k] = found.get(k, 0) + 1
                if len(samples) < 6 and key.lower() in ("citation_title","citation_author","citation_date","citation_publication_date","dc.title","dc.creator"):
                    samples.append(re.sub(r'\s+', ' ', tag.strip())[:220])
    jsonld = bool(re.search(r'<script[^>]+type\s*=\s*["\']application/ld\+json', html, re.I))
    return found, samples, jsonld

def probe(host):
    out = {"host": host, "fetches": 0}
    base = f"https://{host}/"
    with httpx.Client() as client:
        r = get(client, base)
        out["fetches"] += 1
        if isinstance(r, Exception):
            out["error"] = f"{type(r).__name__}: {r}"
            return out
        out["status"] = r.status_code
        out["final_url"] = str(r.url)
        out["ctype"] = r.headers.get("content-type", "")
        out["server"] = r.headers.get("server", "")
        html = r.text if "html" in out["ctype"].lower() or out["ctype"]=="" else ""
        out["wc"] = len(re.sub(r'<[^>]+>', ' ', html).split())
        title = re.search(r'<title[^>]*>(.*?)</title>', html, re.I|re.S)
        out["title"] = re.sub(r'\s+', ' ', title.group(1)).strip()[:160] if title else ""
        # generator / platform hints
        gen = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)', html, re.I)
        out["generator"] = gen.group(1) if gen else ""
        low = html.lower()
        out["hint_dspace"] = "dspace" in low or "vtechworks" in host
        out["hint_omeka"] = "omeka" in low or "omeka" in host
        # check DSpace API/OAI endpoints for repos
        if out["hint_dspace"] or host in ("vtechworks.lib.vt.edu","etd.vt.edu","scholar.lib.vt.edu"):
            for ep in ("/server/api", "/oai/request?verb=Identify"):
                if out["fetches"] >= 3:
                    break
                rr = get(client, f"https://{host}{ep}")
                out["fetches"] += 1
                if not isinstance(rr, Exception):
                    out.setdefault("api", {})[ep] = rr.status_code
                else:
                    out.setdefault("api", {})[ep] = f"ERR {type(rr).__name__}"
    return out

results = []
for h in HOSTS:
    try:
        results.append(probe(h))
    except Exception as e:
        results.append({"host": h, "error": f"outer {type(e).__name__}: {e}"})

print(json.dumps(results, indent=1))
