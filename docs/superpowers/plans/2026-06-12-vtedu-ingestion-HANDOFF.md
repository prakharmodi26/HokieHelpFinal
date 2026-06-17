# HANDOFF — vt.edu Full-Domain Ingestion Execution (2026-06-12)

> ## ⚠️ SPECIAL-HANDLING CATEGORIES — DO NOT crawl in the normal pipeline
> Two data categories are handled **separately** and must be EXCLUDED from the standard
> full/BFS crawl (including the current vtedu-test run). They live in the registry `special:` section.
>
> 1. **Dining** — `dining.vt.edu`, `www.dining.vt.edu` (`handler: dining_tool`). Runtime chatbot
>    tool over the live hours + FoodPro menu feeds (see §3, §7). **Never crawled / never embedded.**
>    The `dining.vt.edu` HTML pages are JS shells over those feeds — do not scrape them.
> 2. **Course catalog / schedule** — `catalog.vt.edu` (`handler: courseleaf_catalog`). CourseLeaf,
>    browser-render + structured ingest as a dedicated pass (Task 8.4, §4a). The engine currently
>    SKIPS these rows pre-render. NOTE: `classes.vt.edu` (the timetable/class-schedule app) is
>    still in `full:` at `-1` — confirm with owner whether it should also move to special handling.
>
> Status: handled in a separate manner per owner (2026-06-17). Do not let registry/cap changes
> silently fold these into the BFS corpus.

> **Audience:** the next LLM/agent continuing this work. You have full repo access.
> **Plan being executed:** `docs/superpowers/plans/2026-06-10-vtedu-full-domain-ingestion.md` (2,867 lines — READ the phases you're implementing; full task text is there).
> **Process:** subagent-driven development (fresh implementer per task batch, full task text pasted into the prompt — subagents don't read the plan file; spec review then quality review per batch). The owner wants complex logic done by the strongest model, mechanical work by cheaper models.

## 1. Current state (verified 2026-06-12)

- **Branch:** `feat/vtedu-full-domain-ingestion`, 26 commits on top of base `b873e99`. **NEVER pushed. NEVER push it** — the owner pushes.
- **Working tree:** clean (only pre-existing untracked files, see §8).
- **HEAD:** `9407f15`.
- **Test status:**
  - crawler: `cd services/crawler && PYTHONPATH=src uv run pytest tests/ -q` → **111 passed**.
  - chunker / embedder: green as of their Phase 5 commits (run the same pattern: `PYTHONPATH=src uv run pytest tests/ -q` in each service dir).
  - chatbot: `cd services/chatbot && PYTHONPATH=src uv run pytest tests/ -q` → **101 passed, 1 failed, 1 deselected**. The failure (`test_llm.py::test_system_prompt_contains_key_instructions`) is **PRE-EXISTING** (verified failing at `b35d6df` before any chatbot changes from this effort): the system prompt was rewritten to a "CAPABILITY BOUNDARY" version and the old assertion (`"conversation history" in SYSTEM_PROMPT`) was never updated. Not caused by this work; fix or skip it separately if desired.
  - `PYTHONPATH=src` is REQUIRED for every service's pytest.

### Commit map (base → HEAD)

```
320d11b..b921f09  Phase 0+1: scope.py (classify_url + hop history), PageMetadata enrichment,
                  frontmatter, SQLite frontier
804e77f, 21cfeef  Phase 2: registry.py + discovery/hostfilter.py loaders, blocklist in classify_url
cfcf585..0b4c46b  Phase 3: extract.py (fit_markdown), render.py (fast/JS tiers), engine.py
89d0d82, e56b34e  Phase 4: config.py vt.edu-wide, main.py registry-seeded orchestration
25c4eca, 1d62da3  cleaner.py and crawl.py retired (with their tests)
9013075           CRAWL_MAX_URLS_PER_HOST enforced in BFS expansion
43c912e..68fca37  Phase 5: chunker + embedder carry source_host/subdomain/page_type → Qdrant
                  payload + KEYWORD indexes
928e970, b35d6df  Phase 6: verify.py + scripts/verify_ingestion.py
bb56586, 27ce454, 3921e40  Phase 8.1+8.2: catalog_card.py, engine mode routing, OAI/Dataverse harvest
27621f4, 9407f15  Phase 8.3: chatbot dining tool + chat-flow grounding  ← NOT yet spec/quality reviewed (§3)
```

## 2. What is DONE and verified

- **Phases 0–6 complete** (see plan + commit map). Architecture: registry (`services/crawler/config/host-registry.yaml`: 585 `full` + 16 `catalog` + 3 `special` hosts; `config/host-blocklist.yaml`: 9 patterns + 2,474 hosts) seeds a durable SQLite frontier (pending→leased→done/failed/skipped, crash-safe resume via `requeue_leased`) → `engine.run_engine` leases batches, renders with crawl4ai 0.8.6 (`arun_many`, MemoryAdaptiveDispatcher, fast tier + JS-tier retry for empty pages via the `NEEDS_JS` sentinel), checkpoints each result immediately → generic `fit_markdown` extraction → VT pages to `crawled-pages`, one-hop externals to `crawled-pages-external` (hop history kept), mirror to `crawled-pages-cleaned`, PDFs/DOCX documents pass straight to cleaned → chunker/embedder propagate source_host/subdomain/page_type into Qdrant payload with KEYWORD indexes → `scripts/verify_ingestion.py` writes a quality report to the `logs` bucket.
- **Task 4.3 bounded validation run PASSED** (local Docker MinIO, since deleted): 10 hosts, depth 2 → 736 done / 708 failed / 29 skipped; `verify_ingestion`: **1,182 docs, 99.2% clean** (9 flagged: 7 boilerplate on policies/governance pages, 2 image-heavy PDFs). External bucket held 112 pages, zero VT hosts (clean separation). Frontmatter fields all present. Failures: ~550 `net::ERR_CONNECTION_RESET` concentrated on 2 hosts (transient rate-limiting — both responded fine to a later probe; **owner decision: retry-at-scale deferred to k8s runs, which crawl from the campus network and should fare better**); 105 robots.txt (correct behavior); ~24 anti-bot on twitter/facebook externals (expected, low value).
- **Phase 8.1+8.2 complete and controller-reviewed** (commits `bb56586`, `27ce454`, `3921e40`):
  - `src/crawler/catalog_card.py`: `extract_card` (precedence: citation_* → DC/DCTERMS → JSON-LD → OpenGraph → title/meta-description), `card_markdown`, `harvest_oai` (oai_dc + resumptionToken paging), `harvest_dataverse` (search-API paging), `harvest_catalog_records` (one card per record → cleaned bucket, key `<host>/_catalog/<sanitized-id>.md`, per-host failure tolerance).
  - `engine.run_engine(modes=...)` / `process_result(mode=...)`: `catalog` hosts → `_store_catalog_card` (page_type=`catalog_card`, no body ingest, **no BFS expansion**, crawl-time cards go to MAIN bucket and reach cleaned via the mirror); `special` hosts (catalog.vt.edu) → **marked skipped pre-render** with reason "special handler pass pending (Task 8.4: courseleaf)"; hosts not in the registry default to `full`. `modes=None` (default) = historical behavior, so old tests/callers are unaffected.
  - `registry.load_catalog_entries` added; `main.cli` threads `load_modes(...)` into the engine and runs the harvest after the documents pass (guarded; stats in the run summary).
- **Housekeeping done:** validation MinIO container (`minio-validation`) and `/tmp/vt_validation.db*` removed.

## 3. Phase 8.3 (dining tool) — COMMITTED but NEEDS REVIEW ⚠️

Commits `27621f4` + `9407f15` exist on the branch (an agent dispatch raced a user rejection; the work landed and tests pass — 101 green, only the pre-existing llm-prompt failure). **No spec review or quality review has happened. Do that first** before building on it:

- `services/chatbot/src/chatbot/tools/dining.py` (366 lines): `get_dining(date=None, location=None, meal=None)` joins the hours API to FoodPro menus via `locationNum` carried in each unit's `urls[]`; Eastern-date defaulting; in-process TTL cache (1h, injectable clock); graceful degradation (never raises into the request path); `DINING_TOOL_SCHEMA` (Ollama function-tool dict); `maybe_answer_dining(query)` deterministic keyword pre-router.
- `9407f15` wires `maybe_answer_dining` into `/ask`, `/chat`, `/chat/stream` in `app.py`: on dining intent the structured result is rendered as a labeled, timestamped context chunk prepended to the RAG context.
- `services/chatbot/tests/test_dining.py` (322 lines) + recorded fixtures (`hours_unitsopenonday.json`, `foodpro_menu.html`); live smoke test behind `-m live`, deselected by default via pytest addopts.
- Review against plan Task 8.3 (plan lines ~2810–2825) + the verified feed notes in §7. Check especially: meal filtering semantics, cache keying `(date, locationNum)`, the intent keyword list (false-positive risk on RAG queries like "dining services employment"), and that `/chat/stream` injection doesn't break streaming.

## 4. REMAINING WORK (in order)

### 4a. Task 8.4 — CourseLeaf catalog structured ingest (plan lines ~2827–2835) — strongest model

**Files:** `src/crawler/courseleaf.py`, `tests/test_courseleaf.py` (crawler service).

- Browser-render tier (JS) required — plain HTTP to catalog.vt.edu returns 202 (bot gate); crawl4ai/Playwright gets 200. Crawl `/course-descriptions/<subject>/` (~150 subjects), `/program-explorer/`, `/undergraduate/`, `/graduate/`. Subject list discoverable from `/course-descriptions/` index or `/azindex/`.
- Parse each `courseblock` → `{subject, number, level (ug<5000/grad), title, credits, prerequisites, corequisites, when_offered, cross_listings, description}`. Map subject→college for a department facet.
- Dual representation: structured fields as Qdrant payload (exact lookups) + an embedded per-course markdown card (semantic search). Programs similarly, `page_type=program`. Course cards `page_type=course`.
- Current academic year only; designed for regular re-crawl (content-hash dedup → update changed, delete removed). No `/previous-publications/`, NO professors/sections (Timetable explicitly out of scope, deferred).
- Wire as a dedicated pass (engine currently SKIPS catalog.vt.edu rows with reason "special handler pass pending (Task 8.4: courseleaf)" — replace that seam: either route those rows to the courseleaf handler or run a standalone pass in `main.cli` keyed off `registry.host_handler(path,"catalog.vt.edu") == "courseleaf_catalog"`). Tests fixture-based (recorded courseblock HTML); no live calls.

### 4b. Task 8.5 — Downstream page_type + structured payload (plan line ~2837) — mechanical, cheap model OK

Extend the Phase 5 pattern (commits `43c912e`, `27b4af6`, `68fca37` show exactly how):
- chunker (`services/chunker/src/chunker/parser.py`, `models.py`, `splitter.py`): already carries `page_type`; add course/program structured fields passthrough if frontmatter carries them.
- embedder (`services/embedder/src/embedder/indexer.py`): payload += course fields; add KEYWORD payload indexes for `page_type` (may already exist — CHECK; Phase 5 added source_host/subdomain/page_type indexes), `subject`, `course_level`.
- ⚠️ **Migration note:** payload indexes are only created in `_ensure_collection` at collection-creation time. Existing Qdrant collections need a one-off `create_payload_index` migration for any new fields (this was already flagged for the Phase 5 fields — write a small idempotent script, e.g. `services/embedder/scripts/migrate_payload_indexes.py`).

### 4c. Phase 8 final review

After 8.4+8.5: dispatch a final code reviewer across `bb56586..HEAD`, then mark Phase 8 done.

### 4d. Phase 7 — k8s (Tasks 7.1–7.3) — ops/runbook, was ON HOLD, now unblocked by owner (see §6)

Per plan: `k8s/crawler-pvc.yaml` (frontier DB), `k8s/crawler-configmap.yaml` (vt.edu values incl. `CRAWL_MAX_DEPTH=10`), `k8s/pipeline-configmap.yaml`, pipeline `jobs.py` PVC mount + memory bump (6Gi request/12Gi limit), `k8s/crawler-test-job.yaml` (SCRATCH buckets), `k8s/crawl-cronjob.yaml` (commit it, do NOT apply against main buckets). Then the full ingestion test Job on scratch buckets + verification gates.
- **HARD STOP:** the executor stops after Task 7.2 verification. Promotion to main buckets/collection is owner-only.

## 5. Operational constraints (from owner + project CLAUDE.md — binding)

- **NEVER `git push`.** Owner pushes.
- k8s: always show `kubectl config current-context` before applying anything; default namespace `test`; **never deploy to prod** unless the owner literally says "deploy to prod"; prefer `kubectl apply -f`; manifests live in `k8s/`; after deploy run `kubectl get pods -n test`, `kubectl get svc -n test`, `kubectl rollout status deployment/<name> -n test`.
- Library docs: use the `ctx7` CLI (`ctx7 library <name> "<q>"` → `ctx7 docs <id> "<q>"`), not training knowledge.
- SDD red flags: never dispatch parallel implementers; full task text in prompts; spec review before quality review; reviewer-found issues → fix → re-review.

## 6. k8s situation — owner's words (2026-06-11/12)

1. Cluster: **endeavour** (kubectl config on this machine has access). Namespace `test`.
2. "Currently you will only test the whole ingestion as it would work in production." — i.e. the k8s work is a full production-shaped test run, still on scratch/test resources until owner promotes.
3. **No RBAC currently** — don't design around service accounts yet; can come later.
4. Owner can set anything via the **Rancher web UI at `launch.cs.vt.edu`** and asked to be told what to do there. **Known cluster problem to raise with the owner:** the in-cluster MinIO pod (`minio-65bf8ccbbf-*`) was in CrashLoopBackOff with a node-level containerd/runc cgroup error ("expected cgroupsPath to be of format slice:prefix:name for systemd cgroups", thousands of restarts) — NOT fixable from kubectl; needs a node fix/redeploy, likely via Rancher. Validation was done against local Docker MinIO instead. Before Phase 7's test Job, MinIO in the cluster must be healthy — tell the owner to check/redeploy MinIO (and confirm storage class for the crawler PVC) in Rancher.
5. Owner theory (agreed): connection-reset failures will largely vanish when crawling from the campus-network cluster; transient-failure retries (`Frontier.requeue_failed`) are deferred to cluster runs.

## 7. Verified external data feeds (hard-won — do NOT re-reverse-engineer)

**VT Dining** (runtime tool, never crawled/embedded):
- Hours JSON: `https://apps.students.vt.edu/hours/Api/NonRestricted/UnitsOpenOnDay/Date/<YYYY-MM-DD>` → units OPEN that day: `name`, `hours[]` (`label` Breakfast/Lunch/Dinner, `open_time`, `close_time`, `title`), `urls[]` (FoodPro menu URL containing `locationNum` — the join key, plus About page). Also `…/UnitsOpen/readByDateRange/<startISO>/<endISO>`, `…/Announcements/readByDateRange/<d>/<d>/false`. Absent unit = closed. Roster is date-scoped.
- Menus HTML (server-rendered, httpx is enough): `https://foodpro.students.vt.edu/menus/MenuAtLocation.aspx?locationNum=<N>&naFlag=1` → parse CSS classes `recipe_title`, `recipe_description`, `portion_size`, `legend_icon vegan|vegetarian`.
- `dining.vt.edu/menus.html`/`hours.html` are JS shells over these — never scrape them.

**Course catalog:** `catalog.vt.edu` is CourseLeaf; curl gets HTTP 202 empty (bot challenge), real browser engine gets 200 → browser-render tier, then BFS over `/course-descriptions/`, `/program-explorer/`, `/undergraduate/`, `/graduate/`, `/azindex/`.

**Catalog-mode repos (already implemented in 8.2):** vtechworks.lib.vt.edu (DSpace, OAI `/server/oai/request`, REST `/server/api`), eprints.cs.vt.edu (OAI `/cgi/oai2`), dataverse.vtti.vt.edu (API `/api`). `data.lib.vt.edu` is `type: dataverse?` with a 403-on-probe note — verify manually before harvesting it.

## 8. Gotchas & environment notes

- `PYTHONPATH=src` required for pytest in every Python service (src layout without editable install).
- **Heredoc corruption:** a PreToolUse shell-rewrite hook (cat→bat) injects ANSI/control chars into `cat <<EOF` heredocs appending to files. NEVER append code via heredoc — use the Edit/Write tools. (Bit us twice.)
- **Rejected-but-committed agents:** agent dispatches that appear "rejected"/interrupted may still have landed commits. ALWAYS `git log --oneline -5` before re-dispatching or assuming work is lost (happened with Batch F `e56b34e` and again with 8.3 `27621f4`/`9407f15`).
- Python is approved for these services (pre-existing Python codebase; owner's global TypeScript preference doesn't apply here).
- Untracked files, leave alone: `services/crawler/discovery/{enumerate,healthcheck}.py` (offline snapshot artifacts used to build the registry), `docs/superpowers/plans/2026-05-27-*.md` (older plan), `.agents/`, `CLAUDE_TODO.md`, `PROJECT_OVERVIEW.md`, `.claude/*`.
- crawl4ai 0.8.6 facts (verified against installed source): `CrawlResult.url` is the ORIGINAL requested URL; final URL lives in `redirected_url` (engine `_match_item` relies on this). `result.html` is raw page HTML. `arun_many(urls=, config=, dispatcher=)` streams results.
- The registry is a **seed set, not a boundary**: any live `*.vt.edu` host found via BFS is in scope unless blocklisted. Don't hand-edit `host-registry.yaml`; it's built by `scripts/build_registry.py`.
- Session memory files (if you're an LLM with this user's memory dir): `vtedu-ingestion-plan-execution-progress.md`, `vt-dining-catalog-data-sources.md` mirror parts of this handoff.

## 9. Suggested immediate next actions

1. Spec + quality review of `27621f4`/`9407f15` (Task 8.3) against plan lines ~2810–2825; fix anything found.
2. Dispatch Task 8.4 (CourseLeaf) to the strongest model with the full task text + §7 catalog notes + the engine skip-seam description (§4a).
3. Dispatch Task 8.5 (cheap model) with the Phase 5 commits as the pattern; include the Qdrant index-migration script.
4. Final Phase 8 review; update the progress memory.
5. Phase 7 k8s prep; before the test Job, give the owner the Rancher checklist (MinIO health, storage class for PVC) per §6.
