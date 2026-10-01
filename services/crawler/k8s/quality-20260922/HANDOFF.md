# Catalog repair branch handoff

Branch: `fix/catalog-curriculum-ingestion`.

The main session handoff is on the paired chatbot branch:
[docs/HANDOFF-answer-quality.md](https://github.com/prakharmodi26/HokieHelpFinal/blob/feat/chatbot-answer-quality/docs/HANDOFF-answer-quality.md).
Read it before resuming: the catalog data is repaired for a small acceptance set, but chatbot retrieval still misses required-course evidence and the website rollout remains blocked.

## Source provenance — why this branch has a broad diff

This worktree began at `origin/feat/vtedu-full-domain-ingestion`, commit `e0515fd`. The actual deployed crawler source, config and pyproject were then copied from:

`ghcr.io/prakharmodi26/hokiehelp-crawler@sha256:835d33b7878770ad064ba65072b7e32c3efc5614baae815edccedff0f5ca8801`

The live crawler differed substantially from that Git branch. Changes in crawler config, engine, discovery, storage, URL/scope handling and related support modules largely reflect that recovered deployed baseline. They are **not** all newly authored curriculum fixes. Full equivalence/full-suite acceptance of that restored baseline is not claimed.

The repair itself centers on `src/crawler/courseleaf.py`, new `src/crawler/program_content.py`, `scripts/repair_program_cards.py`, targeted tests/real HTML fixture, and the manifests in this directory. Deployment used the pinned crawler image with a ConfigMap source overlay, not a fresh unverified full image rebuild. The lockfile was refreshed for the recovered dependency declaration.

## Completed and last verified (2026-09-24 UTC)

Context **endeavour**, namespace **test**. Endeavour is not a namespace.

- Source bucket `test-crawled-pages-0705` was read-only.
- Target bucket/collection: `hokiehelp-quality-20260922`.
- Eight program cards repaired from 13 CS/math URLs attempted. Five graduate pages were disallowed by robots. Source bucket contains 622 program objects: this is a targeted repair, **not** a full catalog pass.
- Table rows, heading context, required courses, alternatives, elective distinctions and footnotes retained; roadmap excluded from requirements. Actual `2026-2027` edition retained.
- VT-specific requirement pane IDs handled. Empty/challenge/error HTML and suspicious loss are rejected.
- Requirements/tables moved out of YAML into the body: oversized YAML exceeded the deployed ingestor's bounded frontmatter parser. A source-derived total-credit summary is near the top of the card.
- Pinned compatible ingestor processed eight documents, zero failed, 52 upserts. Idempotence rerun preserved the same 52 points.
- Original collection `hokiehelp-test`: **268330** points, unchanged. Target: **268367**.
- CS document `d3556d267f8ac571`: 13 chunks. Required MATH 1225/1226/2114, MATH 2204 or CMDA 2005, MATH 2534 or 3034, MATH 3134 or 3124, CS 1114, statistics elective and 123 total are present with metadata/heading context.
- Empty-URL fallback document `e3b0c44298fc1c14`: zero points at final verification. The underlying ingestor's fail-open malformed-frontmatter behavior merits separate hardening; this repair avoids oversized frontmatter rather than redesigning that ingestor.
- Fresh commit-checkpoint tests: **16 passed**, across `test_program_tables_repair.py` and `test_courseleaf.py`. The legacy test was adjusted to verify body requirements and absent YAML requirements, matching the intentional new format. Full crawler suite not certified. Lock and whitespace checks pass.

## Artifacts

All paths below are relative to this directory:

- `artifacts/final-validation-20260924.json`: coverage, IDs, hashes, required-course contexts, counts and idempotence validation.
- `artifacts/program-repair-20260924.json`: exact successes and robots exclusions.
- `artifacts/repaired-cs-card.md`: rendered source card.
- `artifacts/indexed-cs-chunks.txt`: actual stored passages.
- `artifacts/ingest-idempotence.log`, `artifacts/repair-inspection.txt`: supporting operational evidence.

The isolated bucket also contains original cards, raw HTML and reports. Snapshot backup exists there from the main session; do not ingest those objects.

## Operational constraints

- Follow `README.md` here only after checking current resources. Its commands recreate repair resources and **perform writes**; they are not necessary merely to resume code review.
- Always `kubectl --context endeavour -n test ...`.
- **Mandatory ingest prefix:** `catalog.vt.edu/programs/`. Never ingest the whole repair bucket.
- Use the pinned compatible ingestor in `program-ingest-job.yaml` (deployed `:0716` digest), remote BGE-M3 embeddings and the named dense/sparse collection. Do not substitute the old checkout's ingestion stack.
- Manifests use Secret references, not credential values. No credentials/backups are committed.
- Do not write repairs into the original bucket/collection or bypass robots exclusions.

## What remains

1. Main blocker is now chatbot candidate/evidence recall, not absent CS table rows. Do not recrawl blindly to solve a retrieval failure.
2. The main branch contains an unfinished exact-program filter with one failing unit test, and a detailed-math generation error needs diagnosis. See its handoff/evaluations.
3. Dated CS/DCC PDF totals have small-caps text such as `C REDITS R EQUIRED FOR G RADUATION : 123`. They are already indexed but normal queries miss the relevant candidates; this HTML repair does not solve that. Any normalization/reembedding should be a separate isolated, provenance-preserving change with recall tests.
4. Decide whether/when to expand beyond the eight acceptance cards. Respect robots, preserve existing good cards on failed fetches, validate frontmatter and source-derived summaries, and verify index fidelity before switching serving configuration.
5. Validate the broader restored crawler baseline separately before treating this entire branch as production-ready.

## Local validation

From `services/crawler`:

```bash
PYTHONPATH=src uv run --extra dev pytest -q tests/test_program_tables_repair.py tests/test_courseleaf.py
uv lock --check
git diff --check
```
