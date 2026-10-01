# HokieHelp answer-quality handoff

**Start here in the next session. This is an unfinished checkpoint, not a release.**

This document supersedes older evaluation notes where they conflict. Cluster observations below were last verified during the **2026-09-24 UTC** experiments; recheck live state before making changes. This commit/push handoff does not deploy anything further.

## 1. Repositories / branches

Remote: `git@github.com:prakharmodi26/HokieHelpFinal.git`.

| Work | Branch | Existing host worktree |
|---|---|---|
| Chatbot, frontend, evaluations, this handoff | `feat/chatbot-answer-quality` | `/home/prakharmodi/projects/HokieHelpFinal` |
| Catalog ingestion repair | `fix/catalog-curriculum-ingestion` | `/home/prakharmodi/projects/HokieHelpIngestionRepair` |

Keep these changes separate, as the user requested. The chatbot branch started from older main (`1ef015f`), **not** the newer unmerged application running in Kubernetes. Do not assume feature parity with that live stack. The ingestion branch started at `origin/feat/vtedu-full-domain-ingestion` (`e0515fd`). Its broad crawler diff includes a restoration of the **actual deployed crawler source/config/dependencies** before the table repair; it is not all newly authored repair logic.

Paired ingestion checkpoint: **`9386e7d30b26d63db806264e7085efa5bf515cbb`**. Runbook/artifacts: `services/crawler/k8s/quality-20260922/` on the ingestion branch. Its `HANDOFF.md` explains source provenance and scope.

Local `.memory/`, credentials, Docker login config, rollback manifests and virtual environments are not published. This handoff and the committed public/synthetic evaluations are the portable record.

## 2. User decisions / constraints

- Improve answer quality before production hardening; the current website is a test deployment, not production.
- Scope is Virginia Tech-wide, not only CS, subject to corpus coverage.
- Use VT-ARC for generation/planning, **existing BAAI/bge-m3 dense+sparse embeddings and existing reranker** for retrieval. Do not accidentally use the old checkout's BGE-large ingestion stack.
- Search original question plus normalized standalone queries; bounded planning, not an open-ended agent loop.
- No neighboring-chunk expansion. Preserve bounds: <=4 semantic searches, <=30 merged candidates per search, <=8 Qdrant `query_points` calls, one global rerank, <=12 final chunks, <=24,000 evidence characters.
- Inline citations plus cited-only sources; reference validity is not proof of semantic support.
- Separate chatbot and ingestion fixes. User authorized Kubernetes testing, isolated reingestion/reindexing, and eventual test rollout.
- GPT subagents are allowed; do **not** use Astra. VT-ARC subagents were also allowed, but the latest actual host-side attempt still returned campus-VPN HTTP 403. Application model calls **inside Kubernetes work**. GPT workers sometimes failed with usage-limit errors and left partial edits; inspect files, not merely worker completion claims.

## 3. What is implemented

### Chatbot / frontend

- Shared `/ask`, `/chat`, `/chat/stream` planning → retrieval → evidence → generation pipeline.
- ARC defaults: `GLM-5.3` answer model, `DeepSeek-V4.1-Flash` planner.
- History/query validation, bounded history, standalone follow-up planning, original-query recall, dense/sparse RRF and one cross-encoder rerank.
- Catalog program metadata lane and conservative wrong-source filtering (program identity, level, kind, edition, governance documents). Specialization qualifiers must not collapse to their base major; generic degree-planning advice is not automatically a conflicting program.
- Evidence budgeting, page-ID normalization/deduplication, safe URLs, catalog metadata/heading labels, fail-closed unknown/missing citations.
- SSE sends provisional tokens, then authoritative `answer`, `sources`, `done`. UI replaces provisional text, renders safe citation links and cited-only collapsed sources, and excludes failed turns from history.
- `/ready` added; public `/logs` removed. Frozen-lock, non-root chatbot image; separate test ConfigMap/Secret references.

### Ingestion

- Real CourseLeaf tables, headings, alternatives, electives and footnotes preserved in markdown; actual edition preserved, never inferred as student cohort.
- Handles VT's `programcurriculumtextcontainer` and `graduationrequirementstextcontainer`, not only the stock requirement pane; roadmap excluded from required-course extraction.
- Large `requirements` YAML field removed: tables belong in the body. That field overflowed the deployed ingestor's frontmatter parser. Source-derived total-credit summary is placed near the top of a program card.
- Repair runner retains originals/raw HTML and reports in an isolated bucket; checks robots, rejects bad/shell responses and suspicious loss, and does not overwrite the original corpus.
- Actual coverage: **8 repaired CS/math cards out of 622 source program objects**. 13 URLs attempted; five graduate URLs disallowed by robots. This is NOT a full catalog recrawl.
- Pinned deployed ingestor processed eight documents, zero failed, 52 points upserted. Idempotence rerun retained the same points. CS has 13 indexed chunks, with all required math/alternative identifiers and 123 total. Bad empty-URL fallback document `e3b0c44298fc1c14` has zero points at final verification.

## 4. Crucial current source vs tested-image distinction

**The branch's final source is ahead of, and less verified than, the r2 canary image.**

After the r2 tests, a worker started an exact-program-filter refinement and then failed. Its partial edits remain in:

- `services/chatbot/src/chatbot/program_metadata.py`: `program_filter_names()`.
- `services/chatbot/src/chatbot/retriever.py`: exact `program_name` condition, `program_exact` lane and changed reservation cap.

Those edits were discovered during the commit checkpoint. They are intentionally preserved as unfinished work, not represented as accepted fixes. Fresh backend results: **173 passed, 1 failed**:

`tests/test_retriever.py::test_catalog_lane_reuses_embedding_and_shares_candidate_budget`

The earlier **174 passed** result refers to the pre-refinement code used for r2. Do not simply change the assertion to hide the failure: first review the new name inference, lane labels, reservation policy, fallback/recall behavior and agreed bounds. Either finish and test the refinement or explicitly revert its delta against r2.

The failed worker had permission to add an index only on the isolated collection but did not return a handoff. Inspect actual payload indexes before assuming that step happened.

## 5. Kubernetes / registry state

Always distinguish context from namespace:

```bash
kubectl --context endeavour -n test ...
```

### Preserved originals and isolated target

| Resource | Last verified state |
|---|---|
| Original Qdrant collection `hokiehelp-test` | **268330** points; unchanged by repair |
| Original MinIO bucket `test-crawled-pages-0705` | Read-only repair source |
| Target collection `hokiehelp-quality-20260922` | **268367** points after repair |
| Target bucket `hokiehelp-quality-20260922` | Eight repaired cards, raw/original/report objects, snapshot backup |

`docs/evaluations/test-collection-clone.json` records snapshot/clone verification. Approximately 2.18 GB of snapshot data is archived in the isolated bucket; it is not in Git. **Ingestion prefix must be `catalog.vt.edu/programs/`**, never the entire bucket (which also contains backups/raw/originals/reports).

### Built and pushed images

- Chatbot `ghcr.io/prakharmodi26/hokiehelp-chatbot:quality-20260922-r2`
  - `sha256:c544157c1fda379cc26a1a787cf77564c7ca628cf9075fd5f8b747c7632d57d3`
- Frontend `ghcr.io/prakharmodi26/hokiehelp-frontend:quality-20260922`
  - `sha256:5eaf63da9e39c0df2cb0143db2e2470e1073c97eb24c947ed1c0cb696e0267b2`
- Compatible ingestor used by the repair:
  - `ghcr.io/prakharmodi26/hokiehelp-ingestor@sha256:28a5c5f5372ec8cd42e441ffcc7966f7385c4ecb698a905809ad09013add3df9`
- Deployed crawler baseline used with a ConfigMap source overlay:
  - `ghcr.io/prakharmodi26/hokiehelp-crawler@sha256:835d33b7878770ad064ba65072b7e32c3efc5614baae815edccedff0f5ca8801`

### What actually serves traffic

- `chatbot-quality-canary` runs r2 against the isolated collection, with labels that **do not match either live chatbot Service**. Its `activeDeadlineSeconds` is 14400; it may have expired by the next session. `/ready` was 200 and `/logs` 404.
- New ConfigMap: `chatbot-quality-20260922`; new Secret: `chatbot-arc-quality` (`llm-api-key`). Values are not committed. Credentials were explicitly provisioned for the cluster; application code does not read Pi credentials.
- Existing website deployments **were not rolled forward**: chatbot `:0716` (3 replicas), frontend `:0713` (2 replicas), as last inspected.
- Website: https://hokiehelp.endeavour.cs.vt.edu/
- `k8s/quality/chatbot-rollout-patch.yaml` is **prepared, not applied**. It still pins r2, not the unfinished source at this branch tip.
- Private rollback manifests: `~/.local/state/hokiehelp/quality-20260922/{deploy,configmap,svc}.json`. Private Docker auth is below that directory; never commit it.
- Existing `quality-inspect-*` / `quality-catalog-fetch` pods had completed and cannot be used for exec. Repair/ingest jobs completed; check exact resource names before reusing/deleting anything.

If later rolling forward, preserve the previous ReplicaSets and original config/secret references. `rollout undo deployment/chatbot` and `rollout undo deployment/hokiehelp-frontend` are the planned rollback path; original collection remains available. Do not run rollback now—the live deployments were not changed.

## 6. Actual validation, not just unit tests

### Successful observations

- Three generic CS credit paraphrases returned **123**, before and after ingestion; none cited the known Statistics/CALS/sibling-program sources.
- Greeting, unknown person/program refusal, unresolved-reference clarification, parking and SSE worked in the recorded probes.
- Switching to Mathematics returned 120 with a Mathematics catalog citation; this is a recorded answer, not a broad semantic-acceptance claim.
- Six Chromium **mocked** frontend regression tests previously passed. These cover rendering/history/SSE handling, not real deployed UI integration.
- Crawler checkpoint tests: **16 passed** (`test_program_tables_repair.py` + `test_courseleaf.py`). One obsolete legacy assertion was updated to check body requirements and absent frontmatter requirements, matching the intentional parser contract. Full restored crawler suite has not been certified.
- Dependency lock / whitespace checks pass. `/tmp/hokiehelp-quality-venv` became unusable during final checks; use the project environment instead.

### Open blockers / failed acceptance

1. **Required CS math is indexed but not consistently retrieved.**
   - `d3556d267f8ac571#0002`: core/additional math requirements and alternatives.
   - `d3556d267f8ac571#0003`: MATH 1225 / 1226.
   - Actual follow-up normalized to `Virginia Tech undergraduate CS Computer Science BS degree checksheet required mathematics courses`, but evidence was dominated by non-technical-course exclusions/general advising, and the answer refused to list requirements.
   - Explicit 2026–2027 question obtained calculus evidence but missed the other required-course table. Fixing source extraction alone did not fix candidate/evidence recall.
   - The partial exact-program refinement above was an attempt to address this, not a proven solution.
2. **Detailed math request returned HTTP 502 once.**
   - Upstream HTTP 200 followed by completion-validation `ValueError`; raw finish reason was not captured for that failure.
   - A separate diagnostic with `max_tokens=4096` completed, but still gave incomplete answers from incomplete evidence. This does NOT prove token budget was the original cause or fix retrieval.
   - Serving config still uses default `LLM_MAX_TOKENS=2048`; diagnostic override did not change the canary. Investigate finish metadata and latency before changing model defaults or silently accepting truncation.
3. **Dated PDF totals are indexed but missed by normal retrieval.**
   - CS 2023–24: `21d8ef4e32ef41d9#0000` has `C REDITS R EQUIRED FOR G RADUATION : 123`; `#0004` mentions minimum 123.
   - Normal top-30 candidate generation misses those passages. An OCR-shaped diagnostic query recovers `#0000`; rerank 0.8895 and evidence retention then succeed.
   - DCC `2e350e9088bedeb9#0000` has the same heading/total; its case also has a reranking interaction.
   - Current answers safely abstain but are unhelpful. No OCR-specific query hack or hardcoded total was installed. Catalog repair does not repair PDF embeddings.
4. **Edition qualification / verbosity still varies.** Valid citations do not guarantee that every claim is supported or appropriately cohort-qualified.
5. **No full website rollout or real-browser end-to-end acceptance yet.** The post-ingest API run stopped after the detailed-math 502; it did not finish all planned cases.

## 7. Files to read first

On this branch:

- `docs/evaluations/cluster-canary-post-ingest.json` — actual failures as well as successes.
- `docs/evaluations/cluster-math-context-probe.json` — actual standalone queries, full selected evidence, provider usage/finish metadata, and incomplete answers; diagnostic 4096-token override.
- `docs/evaluations/cs-curriculum-review.md` — required vs alternative vs elective reference.
- `docs/evaluations/dated-checksheet-recall.md` and `cluster-canary-edition-controls.json`.
- `docs/evaluations/cluster-canary-pre-ingest.json`, `cluster-canary-controls.json` — earlier HTTP/SSE probes.
- `docs/answer-quality.md` — architecture/history; historical test counts are not current-HEAD guarantees.
- `services/chatbot/{src/chatbot/{retriever,program_metadata,evidence,llm,app,config}.py,tests/}`.
- `k8s/quality/` — isolated configuration/canary/prepared rollout patch.

On the ingestion branch:

- `services/crawler/k8s/quality-20260922/artifacts/final-validation-20260924.json`.
- `artifacts/repaired-cs-card.md`, `artifacts/indexed-cs-chunks.txt`, `artifacts/ingest-idempotence.log`.
- `src/crawler/{courseleaf,program_content}.py`, `scripts/repair_program_cards.py`, the two targeted test files and real HTML fixture.

Historical retrieval reports include implementation hashes; do not compare them as if they all describe current HEAD. Some early provider evaluations were host-side 403 failures, not model-quality measurements.

## 8. Recommended continuation order

1. Inspect branch tips, partial edit and current cluster state. Do not accidentally run the old main manifests against the new code.
2. Resolve/review the partial exact-program refinement and the failing backend test. Keep literal identity/edition safeguards and bounded searches; no hardcoded CS answer, course list or neighbor expansion.
3. Prove that the core curriculum and calculus passages reach final evidence for both real-history follow-ups and explicit 2026–2027 queries. Rerun the true planner, not only hand-authored retrieval queries. Verify alternatives and elective distinctions, not just identifier presence.
4. Diagnose the model error with sanitized finish/usage metadata. Never accept a truncated completion as a successful answer.
5. Decide separately whether to normalize small-caps PDF headings/reembed isolated PDF copies. Preserve originals and provenance; normalization only after retrieval cannot recover a missing candidate. Do not bypass robots to expand graduate catalog coverage.
6. Repeat actual HTTP + SSE acceptance, review the full answers and citations, then build a **new immutable image/tag** from the reviewed code. Update canary/rollout pins; r2 is not current HEAD.
7. Only after acceptance, apply the explicit test rollout (backend and frontend). Run the real public browser test, inspect console/network/citations/history, and save results. Retain rollback data and clearly report any safe-but-incomplete answers.
8. Keep production concurrency/deadline/rate-limit hardening separate but visible. The rewritten backend does not restore the old global concurrency/deadline controls; it is not production-certified.

## 9. Useful commands (read before running)

Offline backend validation from the chatbot worktree:

```bash
PYTHONPATH=services/chatbot/src uv run --project services/chatbot --extra dev pytest services/chatbot/tests -q
uv lock --check --project services/chatbot
git diff --check
```

Crawler validation from the ingestion worktree:

```bash
cd services/crawler
PYTHONPATH=src uv run --extra dev pytest -q tests/test_program_tables_repair.py tests/test_courseleaf.py
uv lock --check
```

Frontend regression suite requires local `@playwright/test` and installed Chromium:

```bash
npx playwright test --config services/frontend/playwright.config.cjs
```

Actual API evaluation (requires a running canary; uses ARC and the live isolated index, incurs provider work):

```bash
kubectl --context endeavour -n test exec -i chatbot-quality-canary -- \
  python - --stream --output /tmp/post-ingest-api.json \
  < services/chatbot/scripts/evaluate_deployed_api.py
```

Fetch its report even if the evaluator exits nonzero. Heuristic checks are not semantic proof. `probe_deployed_context.py` can inspect the same public cases with explicit token-budget override without writing the index. Do not publish provider credentials or reasoning.

After—not before—successful frontend/backend rollout:

```bash
node services/frontend/tests/verify-deployed.cjs \
  https://hokiehelp.endeavour.cs.vt.edu \
  docs/evaluations/live-browser.json
```

That browser script has **no fetch/model mocks** and checks final SSE answers, citations, source toggles and real follow-up history. It has been written but not yet run against the new deployed website.
