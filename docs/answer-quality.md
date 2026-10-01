# Answer-quality iteration

Branch: `feat/chatbot-answer-quality`, based on this repository's older `main`,
**not** the newer unmerged live stack described in GitHub issues. An isolated
canary and catalog reingestion now exist, but the website deployments remain
unchanged and the branch is not accepted as an answer-quality fix.

**Current checkpoint: [read the handoff](HANDOFF-answer-quality.md).** The latest
source contains a partial exact-program retrieval refinement left by a failed
worker: 173 backend tests pass and one fails. Earlier counts and retrieval bounds
below describe the tested pre-refinement iterations, not acceptance of current
HEAD. Eight catalog cards were repaired/reingested, but live math-answer recall
still fails; the prepared rollout patch has not been applied.

## Agreed design and implementation

- Virginia Tech-wide scope, limited by the actual corpus.
- VT-ARC for generative AI only. Preserve the **existing live BGE-M3 embedding
  service, BGE reranker, and named dense/sparse index**. The older checkout's
  BGE-large ingestion code is NOT compatible with that live collection.
- Plan all requests, including first turns and `/ask`. Resolve follow-ups;
  preserve program, person, degree level, term, and year without inventing them.
  Produce one primary query covering the full question, plus at most two focused
  queries for comparisons/multipart questions. Greet or clarify without retrieval.
- **User-approved revision after live testing:** search the original question
  alongside the planned queries, unless identical. Maximum four searches total,
  not an open-ended agent loop. This preserves wording lost through normalization.
- Each general search prefetches 30 dense and 30 sparse hits through Qdrant RRF.
  Academic questions additionally use a bounded `page_type=program` metadata view
  with the **same embedding** and an explicit degree-level filter when available.
  Up to 6 literal program-name/level/edition matches share the existing 30-candidate
  budget; other slots remain general retrieval. No additional generated queries.
- Merge/deduplicate at most 120 candidates, then rerank once against the resolved
  primary query. Raw pronouns do not determine final relevance. Apply the same
  cross-encoder floor of 0.3 in hybrid and dense-only modes. Within the final 12
  slots, reserve up to 3 for matching catalog pages (one chunk/page), then fill by
  reranker order. This is explicit source-identity priority, not a higher score or
  semantic-support guarantee. `PROGRAM_CANDIDATES_ENABLED=false` disables it.
- **No neighboring chunks.** No local model loading or index writes.
- Evidence is bounded to 24,000 rendered characters, including metadata, and
  4,000 characters per chunk. These are character limits, not exact token counts.
  History has separate count/character limits and a 4,000-character per-message
  cap so a long assistant reply does not evict the user's referent.
- Inline numeric citations and a collapsed list of only cited pages. No arbitrary
  additional source-list cap. RRF is never displayed as a relevance percentage.

## Pipeline and contracts

`question + history -> validated plan -> original + planned searches -> merged
reranking -> bounded evidence/page IDs -> ARC answer -> citation reference check`

Model-generated queries are limited to 500 characters each. Malformed, timed-out,
empty, truncated, or rejected planning calls fall back to the original question
(up to the API's 2,000-character input limit). This is fail-open availability, not
proof of successful history resolution. Database/embedding/reranking failures
are errors, not successful empty retrievals.

Page IDs are assigned after evidence selection. URL variants share IDs; duplicate
passages, images-only content, and unsafe/non-web source URLs are excluded. Full
retrieval objects are not mutated. Heading context and decoded titles accompany
evidence; catalog program, level, and edition labels also share the same budget.
The cited-source response is `{id, title, url}`, without retrieval scores.

Citation validation requires at least one supplied numeric ID and rejects unknown
IDs. Missing/invalid citations produce an explicit insufficient-evidence fallback
with no sources. **This checks reference integrity, not semantic entailment.** A
model can still attach a valid ID to an unsupported assertion.

SSE emits provisional `token` events, then an authoritative `answer`, `sources`,
and `done`. The frontend replaces provisional text with the final answer. Failed
or incomplete streams show an error and do not enter successful conversation
history. Provisional text is not guaranteed grounded or citation-valid.

`RETRIEVAL_TRACE` logs contain all merged candidate IDs/URLs, score provenance,
query ranks, and the selected rerank scores, separately from the source response.
They omit questions, history, and passage bodies. Evidence selection logs its
budget/counts. These are bounded server diagnostics, not a new user-facing trace
viewer or durable evaluation store.

## Review and acceptance status

Worker output was reviewed rather than accepted on completion alone. Several
workers failed; the primary agent integrated the implementation and corrected
issues found in remaining worker files, including the frontend test server,
uncited source display, obsolete retriever tests, and stale configuration docs.

Latest backend run: **158 tests passed**; the Chromium suite passed **6 tests**;
dependency lock and whitespace checks also passed. These are not live acceptance.

Completed checks include:

- Combined backend unit/API tests: planner schema/fallback, provider failures,
  current service/index contracts, original-query preservation, global reranking,
  no neighbors, deduplication, budgets, citation references, and SSE failures.
- Chromium browser tests against the real frontend with mocked API responses:
  final-answer replacement, citations/XSS safety, collapsed sources, interrupted
  streams, UTF-8 boundaries, history behavior, and disabled in-flight controls.
- Actual VT-ARC calls to both configured models; actual live corpus retrieval;
  one successful real provider streaming probe before access failed.
- Dependency lock consistency and whitespace checks.

**Not complete:** independent final review (worker/provider failures prevented a
completed review), final live evaluation of the revised pipeline, and source-level
acceptance on the GitHub regressions. Passing tests does not resolve #29/#30/#50.

## What live testing actually found

Reports are in `docs/evaluations/`:

| Artifact | Meaning |
| --- | --- |
| `arc-smoke.json` | Both ARC models accepted JSON planning requests |
| `arc-preimplementation-benchmark.json` | Tiny preliminary synthetic sample, not the final prompt |
| `arc-model-benchmark.json` | Broader synthetic benchmark before later prompt refinements; contains failures and raw outputs |
| `retrieval-baseline.json` | Read-only live corpus/index compatibility and retrieval probes |
| `pipeline-normalized-only.json` | Actual earlier production-code pipeline using only normalized queries; exposes serious quality failures |
| `pipeline-evaluation.json` | Revised original-plus-normalized attempt; provider errors blocked its answer checks; NOT a successful acceptance run |
| `corpus-audit.md` / `corpus-audit.json` | Source-object/index diagnosis: indexed CS total exists; specific catalog curriculum rows are absent upstream of chunking |
| `corpus-retrieval-trace.json` | Production retrieval with hand-specified queries; localizes known evidence misses to the candidate stage, with no model calls |
| `crawl-coverage-summary.json` / `ingestion-rules-probe.json` | Historical crawl counters, archived rule/parser probes, and explicit provenance/access limitations |
| `corpus-retrieval-control.json` / `corpus-retrieval-metadata.json` | Fresh 11-case disabled/enabled comparison; CS catalog total recovered in final evidence for all 3 credit paraphrases |
| `metadata-retrieval.md` | Bounded metadata-path design, observed improvements, and unresolved limitations |

The live collection `hokiehelp-test` had 268,330 points, BGE-M3 identity-composed
embeddings, and named dense/sparse vectors. Re-embedding sampled passages matched
stored vectors. Using the older unnamed BGE-large query interface failed outright.

Normalized CS bachelor's-credit questions frequently retrieved master's/PhD
pages; wrong-program rerank scores were often 0.96–0.99. Across earlier CS
paraphrases, source overlap was poor. A 0.3 floor is not a program-identity filter.

GLM-5.3 generally declined to invent missing program totals, but full-evidence
answers took roughly 31–43 seconds in several cases and one hit the 60-second
read timeout. DeepSeek was faster, but one answer incorrectly treated general
undergraduate policy as a CS-specific total, and another cited a sibling program.
Unknown-person refusal, greeting routing, and unresolved-reference clarification
worked in the successful pipeline sample. Different unknown names can still
retrieve above-floor candidates; the threshold is not a universal guarantee.

After revising retrieval, a direct live probe with an explicit undergraduate CS
checksheet query plus the original wording recovered CS checksheets near the top.
However, Statistics, graduate-program, old proposal, and sibling-program documents
still remained. This is an observed recall improvement, not proof of answer
correctness or a controlled attribution to one change.

The revised full-answer rerun then encountered provider failures. A separate tiny
chat-completion probe confirmed **HTTP 403** from VT-ARC, with no Retry-After header.
The reason was not established. Further generation calls were stopped. Prompt
changes made afterward, including stronger no-other-person and no-general-minimum
rules, have unit coverage but have not been revalidated against the provider.

### Ingestion versus retrieval diagnosis

The [corpus audit](evaluations/corpus-audit.md) distinguishes missing data from
missed indexed evidence. Before the metadata change, the CS catalog's explicit
123-credit statement and the 2023–24 checksheet headline were already indexed, but
neither target chunk reached the diagnostic candidate pools. The [metadata-path
comparison](evaluations/metadata-retrieval.md) now recovers the catalog total in
final evidence for all three tested CS credit paraphrases. Separately, the catalog source card omits core
curriculum rows present in the readable public page; the archived parser reproduces
table loss. Older PDF evidence still contains those course rows.

The tested CS URLs pass the reviewed block rules. Other hosts/paths are deliberately
excluded, and the historical crawl contains thousands of failures/skips. Neither
university-wide coverage nor the exact historical reason for each missing URL is
established. No ingestion or policy change has been applied.

### Defaults and remaining work

Provisional defaults are DeepSeek-V4.1-Flash for planning and GLM-5.3 for answering:
Flash is faster; GLM was more conservative in the tested real-source answers.
This is not a claim that either model eliminates hallucinations or that the
latency/defaults are production-ready.

Next acceptance work:

1. Restore authorized ARC access and rerun both benchmarks and the actual pipeline.
2. Evaluate and refine the metadata-aware path beyond the current sample, including
   wrong-program PDFs, program aliases, and cohort semantics. Merely increasing
   top-k or the reranker threshold is insufficient. Separately repair curriculum
   extraction in the actual ingestion implementation, testing source-card/chunk
   fidelity before reingesting.
3. Inspect actual cited passages against claims, including cohort distinctions,
   the three #29 paraphrases, #16 follow-ups, and #30 sibling-program leakage.
4. Complete an independent review, then reconsider model/context defaults.

MCP/live information, reconstruction of unmerged services, deployment hardening,
corpus-wide metadata/edition repair, and embedding changes remain separate work.

## Configuration and reproduction

Set `LLM_API_KEY` through the environment/secret manager. Runtime code never reads
pi's personal config. Evaluation scripts can explicitly opt in using `--pi-config`.
No credentials belong in tracked files.

Defaults: `LLM_BASE_URL=https://llm-api.arc.vt.edu/api/v1`,
`EMBEDDINGS_URL=http://embeddings-service`, `QDRANT_COLLECTION=hokiehelp-test`,
`EMBEDDING_MODEL=BAAI/bge-m3`, `CANDIDATE_K=30`, `TOP_K=12`, `MIN_SCORE=0.3`.
`MIN_SCORE` now means cross-encoder relevance, **not dense cosine similarity**.

Other controls: `LLM_MODEL`, `REWRITER_MODEL`, `HYBRID_ENABLED`,
`PROGRAM_CANDIDATES_ENABLED` (provisionally true),
`MAX_CONTEXT_CHARS`, `MAX_CHUNK_CHARS`, `MAX_HISTORY_MESSAGES`,
`MAX_HISTORY_CHARS`, `LLM_TIMEOUT`, `REWRITE_TIMEOUT`, `DEPENDENCY_TIMEOUT`,
`LLM_MAX_TOKENS`, and `REWRITE_MAX_TOKENS`. Old keyword-scroll/RRF settings are no
longer used. Do not blindly reuse an old ConfigMap: explicit old values override
new defaults. The checked-in legacy ingestion/deployment manifests are not an
integrated deployment recipe for this branch.

```bash
# Use a Python environment containing services/chatbot's declared dependencies.
PYTHONPATH=services/chatbot/src python -m pytest services/chatbot/tests -q
npx playwright install chromium --only-shell
npx playwright test --config services/frontend/playwright.config.cjs --workers=1
uv lock --check --project services/chatbot

# Only after provider access is restored. Constructed questions, no user logs.
PYTHONPATH=services/chatbot/src python services/chatbot/scripts/benchmark_arc.py --pi-config --all-cases
PYTHONPATH=services/chatbot/src python services/chatbot/scripts/evaluate_pipeline.py --pi-config --port-forward --stream-probe
```

The pipeline script uses read-only port-forwards to context `endeavour`, namespace
`test`, and cleans them up afterward. Both evaluation scripts checkpoint results;
future runs record prompt hashes and stop on provider authorization/rate-limit
errors. Browser tests mock backend replies: they are not a deployed end-to-end test.
