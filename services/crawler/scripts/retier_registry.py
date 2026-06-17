#!/usr/bin/env python3
"""One-shot registry re-tiering (Option A) — production-test scoping.

Tiers every `full` host with an explicit max_urls so the global default
(CRAWL_MAX_URLS_PER_HOST=50) only ever governs BFS-discovered hosts NOT in the
registry:

  -1   departments / colleges / student services (crawl everything)
   1   personal faculty / research-lab / personal sites (home page only)
  25   news / calendar / video / newsletter (dynamic, low retrieval value)
 200   cpes.vt.edu + www.cpes.vt.edu (research center, bounded)

ezproxy/ext/acos are handled by host-blocklist.yaml; the 4 *.ext.vt.edu hosts
are dropped from the registry here since the `\\.ext\\.vt\\.edu$` pattern blocks
them anyway.

Run: PYTHONPATH=src python scripts/retier_registry.py
"""
from __future__ import annotations

import sys
from collections import Counter

import yaml

REG = "config/host-registry.yaml"

# 30 evidence-based faculty/lab/personal hosts (sampled object keys confirm
# research rosters, lab blogs, personal archives — see triage analysis).
FACULTY_EVIDENCE = {
    "fox.cs.vt.edu", "eatherton.cee.vt.edu", "echolab.cs.vt.edu",
    "thirdlab.cs.vt.edu", "strom.cee.vt.edu", "splice.cs.vt.edu",
    "mascots.cs.vt.edu", "bioinformatics.cs.vt.edu", "www.mcglothlin.biol.vt.edu",
    "www.wang.bse.vt.edu", "www.scottlab.bse.vt.edu", "www.wellwater.bse.vt.edu",
    "www.mics.ece.vt.edu", "mics.ece.vt.edu", "salt.cee.vt.edu", "bsld.cee.vt.edu",
    "lddi.cee.vt.edu", "sem.cee.vt.edu", "cem.cee.vt.edu", "geot.cee.vt.edu",
    "ewr.cee.vt.edu", "kohl.aaec.vt.edu", "iatrc.aaec.vt.edu",
    "ajmorrisgroup.chem.vt.edu", "virginiaview.cnre.vt.edu", "bigtree.cnre.vt.edu",
    "forestupdate.frec.vt.edu", "herbarium.biol.vt.edu", "vtsl.vetmed.vt.edu",
    "www.magma.geos.vt.edu",
}

NEWS = {
    "vtx.vt.edu", "video.vt.edu", "glcweekly.graduateschool.vt.edu",
    "calendar.lib.vt.edu", "vtnews.vt.edu", "www.vtnews.vt.edu",
    "news.lib.vt.edu", "synergy.cs.vt.edu", "blogs.ext.vt.edu",
}
CPES = {"cpes.vt.edu", "www.cpes.vt.edu"}

# faculty-pattern hosts that are actually student/degree-facing — keep at -1.
KEEPERS = {
    "website.cs.vt.edu", "students.cs.vt.edu", "courses.cs.vt.edu",
    "csgrad.cs.vt.edu", "mph.vetmed.vt.edu", "www.mph.vetmed.vt.edu",
    "undergrad.research.cee.vt.edu",
}

# Structural faculty detector for UNSEEN seeded labs (caps them at the home page
# too). Departments below; prefixes that denote a dept/service stay at -1.
DEPTS = {
    "cs", "ece", "me", "cee", "che", "ise", "mse", "bse", "aoe", "esm", "math",
    "stat", "phys", "chem", "biol", "bio", "geos", "ento", "econ", "psyc",
    "engl", "hist", "spes", "cnre", "frec", "fishwild", "ppws", "apsc", "dasc",
    "alce", "hnfe", "ohrs", "lar", "arch", "art", "music", "tbmh", "vetmed",
    "fst", "mii", "ictas", "aaec", "hdfs", "sbio", "cmda", "nde", "grado",
}
KEEP_PREFIX = {
    "www", "research", "graduate", "grad", "undergraduate", "undergrad", "ug",
    "advising", "about", "students", "courses", "csgrad", "news", "apply",
    "admissions", "mph",
}


def is_faculty_pattern(h: str) -> bool:
    labels = h.split(".")
    if labels[-2:] != ["vt", "edu"]:
        return False
    core = labels[:-2]
    if len(core) <= 1:
        return False
    dept, name = core[-1], ".".join(core[:-1])
    return dept in DEPTS and name not in KEEP_PREFIX


def host_of(e):
    return e if isinstance(e, str) else e["host"]


def cap_for(h: str) -> int | None:
    if h in KEEPERS:
        return -1
    if h.endswith(".ext.vt.edu") or h == "ext.vt.edu":
        return None  # drop — blocklisted by pattern
    if h in CPES:
        return 200
    if h in NEWS:
        return 25
    if h in FACULTY_EVIDENCE:
        return 1
    if is_faculty_pattern(h):
        return 1
    return -1


def main() -> int:
    data = yaml.safe_load(open(REG))
    full = [host_of(e) for e in (data.get("full") or [])]

    new_full = []
    counts: Counter = Counter()
    dropped = []
    for h in full:
        cap = cap_for(h)
        if cap is None:
            dropped.append(h)
            counts["DROP"] += 1
            continue
        new_full.append({"host": h, "max_urls": cap})
        counts[cap] += 1

    # Seed key campus news/video/calendar hosts that BFS would otherwise reach
    # only under the global default (50). Adding them at 25 guarantees the cap.
    present = {e["host"] for e in new_full}
    for h in ("vtx.vt.edu", "video.vt.edu", "glcweekly.graduateschool.vt.edu",
              "calendar.lib.vt.edu"):
        if h not in present:
            new_full.append({"host": h, "max_urls": 25})
            counts[25] += 1

    data["full"] = new_full
    data.setdefault("_meta", {})["retiered"] = "2026-06-16"
    data["_meta"]["tiering"] = (
        "Option A: explicit max_urls per host. -1=dept/service (unlimited), "
        "1=faculty/lab/personal (home page only), 25=news/calendar/video, "
        "200=cpes. Global CRAWL_MAX_URLS_PER_HOST=50 governs only "
        "BFS-discovered hosts not in this registry."
    )

    with open(REG, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=120)

    print("Re-tiered registry written.")
    for k in (-1, 1, 25, 200, "DROP"):
        print(f"  cap={k}: {counts[k]}")
    print(f"  total full kept: {len(new_full)}")
    print(f"  dropped (ext, now blocklisted): {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
