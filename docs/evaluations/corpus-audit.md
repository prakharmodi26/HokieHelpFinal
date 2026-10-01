# Corpus and ingestion diagnosis

Read-only audit of `hokiehelp-test` and its source bucket `test-crawled-pages-0705`.
No crawl, reindex, deployment, allowlist/blocklist change, or generative-model call was performed by these probes.

## Follow-up implementation

The user chose **separate fixes**: retrieval in this chatbot branch, ingestion repair
independently. The subsequent [metadata retrieval change](metadata-retrieval.md)
recovers the catalog credit-total chunk in final evidence for all three tested CS
paraphrases. It does not repair the curriculum table or verify generated answers.
The diagnosis below records the pre-fix observations.

## Findings

| Regression / question | Evidence found | Classification |
|---|---|---|
| CS bachelor's total credits | Catalog page has 7 indexed chunks. `d3556d267f8ac571#0006` says “Complete 123 credit hours”; metadata identifies Computer Science Major, undergraduate, catalog edition 2026–2027. The 2023–24 checksheet has 20 chunks; `21d8ef4e32ef41d9#0000` contains its 123-credit headline. Both source cards exist in MinIO. | **Retrieval failure, not an absent-total ingestion gap.** This is source/year-specific evidence, not a hardcoded answer for every cohort. |
| CS required mathematics | The readable public catalog curriculum includes MATH 1225, 1226, and 2114, plus CS 1114. All four markers are absent from the stored catalog card body and all 7 corresponding indexed chunks. The older 2023–24 PDF card and chunks contain them. | **Specific-data gap in the catalog card, upstream of chunking/indexing**, plus retrieval limitations. It is not a complete corpus-wide absence of mathematics information. |
| Wrong program / old proposal sources | New diagnostic retrieval still selects Statistics, Secure Computing, graduate pages, and governance supporting documents alongside CS checksheets. | **Program/cohort selection and ranking failure.** More ingestion alone will not fix it. |
| University-wide completeness | The crawl summary records substantial failures/skips; source policy deliberately excludes some hosts and paths. | **Coverage is not established.** A `vt.edu` allow setting does not mean every VT topic was collected. |
| Raw catalog fetch on this host | Direct Python HTTP returned an empty HTTP 202; the separate readable web-fetch tool returned the curriculum. | **Fetch/access limitation**, not evidence of an empty or missing public page. Cause is unestablished. |

### Retrieval-stage localization

`corpus-retrieval-trace.json` runs the production `Retriever.search_many` with hand-specified standalone queries plus each original question. This is a retrieval diagnostic, **not** a successful ARC planner/answer acceptance run.

For all three credit paraphrases:

- Neither the catalog source nor the tested PDF's headline-total chunk entered either top-30 candidate list.
- Other chunks from the tested PDF sometimes entered the pool, but they did not contain the requested headline total.
- Some selected older CS checksheet copies do contain `123`; matching that number alone does not establish that the passage supports the requested program/year.
- Wrong-program chunks can still score about 0.97–0.98 after reranking.

This proves a candidate-recall problem for known strong evidence, in addition to the already observed ranking/answer-grounding problems. Citation validation cannot recover evidence that never reaches the model. The fix should not reintroduce neighboring-chunk expansion, which remains explicitly excluded.

## Source extraction

The catalog source is stored under a **special-handler key**:

`catalog.vt.edu/programs/undergraduate-college-engineering-computer-science-computer-science-bs.md`

Looking only for a path-shaped object key would incorrectly report it missing. The card body is 7,738 characters; its frontmatter repeats requirements, so metadata must not be mistaken for indexed body coverage.

Reviewed crawler source: `origin/feat/vtedu-full-domain-ingestion` at `e0515fdfb3bf460f39d8520e5436ffc8ed98bbd3`, not the older crawler in this working branch.

In `services/crawler/src/crawler/courseleaf.py`, `parse_programs` collects up to three paragraphs plus matching `h2`, `h3`, and `li` elements. It does not serialize the main curriculum table. A synthetic HTML fixture with MATH 2114 in a table and “Complete 123 credit hours” in a list reproduced exactly that loss: credits retained, curriculum row omitted. See `ingestion-rules-probe.json` for the fixture, source hash, and results.

**Provenance limit:** this is a parser reproduction from the archived branch, not verification of the exact code packaged in historical image `:0705`. The historical raw HTML was not compared. The live card/index gap is directly observed; the archived parser supplies a consistent mechanism, not complete historical attribution.

## Allow/block rules and collection coverage

Current live crawler configuration includes:

- `CRAWL_ALLOWED_DOMAINS=vt.edu`
- Empty `CRAWL_BLOCKED_DOMAINS`, but `HOST_BLOCKLIST_PATH=config/host-blocklist.yaml`
- `CRAWL_BLOCKED_PATHS=/content/(?!dam/),/editor.html`
- Default host cap 50; unlimited-host backstop 1,500

An empty environment blocklist does **not** disable the YAML rules. The archived registry explicitly seeds hosts; ordinary scope classification uses `vt.edu`, and special handlers have separate harvesting paths. Catalog frontier counts are not a count of all catalog cards harvested. In the reviewed `main.py`, `blocked_paths` is passed into `HostFilter`, whereas the configured `blocked_domains` field is not wired into that path; inspect the packaged version before relying on that setting operationally.

Applying the archived YAML plus current path settings:

- Tested CS catalog/checksheet, CS undergraduate, registrar checksheet, admissions deadline, and parking URLs are **not blocked**.
- `classes.vt.edu` is explicitly blocked; static course descriptions do not replace live section/schedule data.
- `ext.vt.edu` and its subdomains are blocked by pattern.
- `/wp-content/uploads/` is blocked, including a synthetic PDF under that path. This may exclude useful documents; it does not explain the tested CS checksheet, which uses `/content/dam/` and is present.
- `/content/` outside `/content/dam/` is blocked by the current path expression.
- Authentication, transactional, proxy, news/archive, and other host/path exclusions remain. They should not all be removed indiscriminately.

The reviewed registry gives core CS and registrar hosts `max_urls: -1`, but startup can replace `-1` with the configured backstop. Do not infer a particular URL was cap-dropped merely from these settings.

### Historical crawl outcome

The completed July job used crawler image `:0705`; its persisted summary reports:

- Frontier: **17,154 done; 6,298 failed; 2,852 skipped**.
- Document processor: **13,759 processed; 1,164 failed** (separate counters, not additive to frontier totals).
- Four wedged render batches.
- Sample hosts: parking 14 done / 26 failed; math 35 / 11; CS 13 done / 2 failed / 25 skipped; catalog 39 done / 11 skipped.

See `crawl-coverage-summary.json`. These aggregate figures cannot distinguish each URL's block rule, failed download, rendering failure, cap, or discovery failure. The historical frontier lived on `emptyDir`, so this audit did not recover per-URL frontier reasons from it. Current ConfigMaps are not proof of the settings used by that July run.

## Acceptance work

1. **Preserve curriculum tables in the actual ingestion implementation**, with fixtures covering program identity, edition, total credits, required courses, alternatives, and electives. Verify source card → chunk fidelity before proposing a targeted reingest. Do not run the older BGE-large ingestion code against the live BGE-M3 collection.
2. **Improve source-aware candidate recall and selection**, using program/level/edition metadata already present on catalog cards. Known exact-program credit chunks must become retrievable without relying on raw-question phrasing or neighbors.
3. **Add coverage checks for each failing intent**: expected authoritative URL, public fetch result, rule eligibility, discovery/fetch status if known, stored object, indexed chunks, candidate rank, selected evidence, then supported answer. Keep access errors and unknown provenance separate from confirmed absence.
4. **Review exclusions and failed URLs selectively**, particularly live class data and university-wide services. Decide scope before changing policy; no blanket unblock or full recrawl is justified by the current evidence.
5. Restore ARC access and rerun answer tests. Provider HTTP 403 is an access problem, not model-quality evidence. Independent model review remains incomplete.

## Reproduce

```bash
# Source/index checks; credentials remain inside the existing admin pod.
kubectl --context endeavour -n test exec -i ADMIN_POD -- \
  python - --minio-bucket test-crawled-pages-0705 \
  < services/chatbot/scripts/audit_corpus.py \
  > docs/evaluations/corpus-audit.json

# Production retrieval, no planner or answer calls.
PYTHONPATH=services/chatbot/src python \
  services/chatbot/scripts/audit_retrieval_coverage.py --port-forward --disable-program-candidates \
  --output docs/evaluations/corpus-retrieval-control.json
```

The corpus script distinguishes failed lookups from missing objects, marks truncated index scans incomplete, checks URL variants, and excludes frontmatter from body-marker tests. Its offline tests validate those diagnostic distinctions, not answer quality.
