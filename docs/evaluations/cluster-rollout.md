# Answer-quality cluster test — checkpoint, rollout blocked

**Authoritative continuation document: [session handoff](../HANDOFF-answer-quality.md).**
The current branch also contains an unfinished post-r2 retrieval refinement
(173 backend tests pass, one fails). The r2 image below predates that edit.

Context: `endeavour`; namespace: **`test`** (not production).

## Isolation and rollback

- Original collection: `hokiehelp-test`; original crawler bucket: `test-crawled-pages-0705`.
- Isolated working collection/bucket: `hokiehelp-quality-20260922`.
- `test-collection-clone.json` records a verified 268,330-point clone and a 2.18 GB archived snapshot. The original collection/bucket are not repair targets.
- Pre-change deployment/config/service manifests are saved privately under `~/.local/state/hokiehelp/quality-20260922/`.
- The new chatbot configuration is a separate ConfigMap, `chatbot-quality-20260922`, and its ARC credential is a separate Secret, `chatbot-arc-quality`. Credential values are not stored in this repository.
- The canary Pod does not match either live chatbot Service selector. Its image is pinned in `k8s/quality/chatbot-canary.yaml`.
- Planned rollout patch: `k8s/quality/chatbot-rollout-patch.yaml`. **Not applied yet.**
- After rollout, `kubectl --context endeavour -n test rollout undo deployment/chatbot` restores the previous image, ConfigMap and Secret references; the original index remains available. Frontend rollback uses `rollout undo deployment/hokiehelp-frontend`.

## Verified so far (2026-09-24 UTC)

- Chatbot image `quality-20260922-r2`: `sha256:c544157c1fda379cc26a1a787cf77564c7ca628cf9075fd5f8b747c7632d57d3`.
- Frontend image `quality-20260922`: `sha256:5eaf63da9e39c0df2cb0143db2e2470e1073c97eb24c947ed1c0cb696e0267b2` (built, not rolled out).
- Canary `/ready` succeeds; unauthenticated `/logs` is removed (404).
- ARC works **inside Kubernetes**. Host-side ARC subagents still returned a campus-VPN 403; this is distinct from application availability.
- 174 backend tests and six mocked Chromium frontend regression tests pass. Lock and whitespace checks pass.
- Independent read-only GPT review found no remaining must-fix source-filter issues for a low-traffic test rollout after specialization and generic-degree-advice regressions were fixed.
- Actual HTTP canary requests, without mocked planner/retrieval/model: three CS credit paraphrases all returned **123** with citations, and none cited the known sibling/statistics/CALS sources. See `cluster-canary-pre-ingest.json`.
- Greeting, unknown person/program, and unresolved pronoun behavior passed the recorded checks. Parking returned cited parking sources. Real SSE emitted tokens followed by authoritative `answer`, `sources`, `done`. See `cluster-canary-controls.json`.

Additional dated/program probes safely abstained rather than supplying the requested total: CS 2023–24 failed its 123-credit check, and Data-Centric Computing also gave an incomplete answer. `dated-checksheet-recall.md` records that the PDF totals are indexed but missed by normal candidate generation (with an additional reranking issue for DCC). This is not fixed by catalog-table extraction.

These are not full answer-quality acceptance: the credit answers still vary in edition qualification, and one declines to confirm an edition's total. Count/citation heuristics do not establish passage-level entailment. A source-family filter cannot guarantee grounding or currency.

## Pending gates

Catalog repair/reingestion is now verified for **eight** CS/math cards, not the entire catalog. Target collection has 268367 points, original remains 268330. CS has 13 chunks with all required math/alternative identifiers; the compatible ingestor processed eight documents, zero failed, 52 upserts. Idempotence and source-card/index fidelity reports are on the separate ingestion branch.

1. Resolve/review the unfinished exact-program retrieval refinement and failing test.
2. Fix remaining math evidence recall: the post-ingest API follow-up still refuses to list required courses despite their presence in Qdrant. The explicit 2026–2027 math request also returned 502 once. See `cluster-canary-post-ingest.json` and `cluster-math-context-probe.json`.
3. Diagnose provider finish metadata/token-budget behavior; a diagnostic 4096-token override completed but did not fix missing evidence or change serving defaults. Rerun complete HTTP/SSE acceptance and manually review citations, alternatives and editions.
4. Build a newly reviewed image, update manifest pins, and roll out backend/frontend only after acceptance. Then drive the real public UI using `services/frontend/tests/verify-deployed.cjs`. That browser test has no API or model mocks and checks SSE rendering, citations, source toggling and conversation history; it has not yet been run against the new public deployment.

This is a low-traffic test, not load/production acceptance. The rewritten backend does not yet restore the previous deployment's global concurrency/deadline controls; dependency timeouts and per-session request limits are not a substitute for them.
