# Bounded program-metadata retrieval

User decision: keep the ingestion repair separate. This change is confined to the
chatbot branch; no crawler merge, deploy, index write, or reindex was performed.

## Implementation

- `services/chatbot/src/chatbot/program_metadata.py`: conservative recognition of
  academic queries and literal program names from existing catalog payloads.
  Preserve major/minor/option distinctions and explicit level/edition constraints.
  No program ontology, acronym guessing, or model-generated filters.
- `retriever.py`: reuse each query's BGE-M3 embedding for one additional bounded
  metadata view (`page_type=program`, optionally `program_level`). Both dense and
  sparse prefetches receive the filter. The general view remains unfiltered.
- Keep **at most 30 merged candidates per query**: up to 6 program matches,
  round-robin across pages, plus general results. At most four semantic searches
  and eight Qdrant `query_points` calls; no neighbors or open-ended searches.
- Rerank the global pool **once**, retaining the existing 0.3 floor. At the default
  final limit of 12, reserve up to 3 slots for matching program pages, one passage
  per page, before filling by ordinary rerank order. Lower configured limits reduce
  this reservation. Matching metadata never bypasses the relevance floor.
- `evidence.py`: include bounded program, level, and catalog-edition labels inside
  the existing character budget. The answer prompt asks for the source edition,
  not an assumed student cohort or unsupported “latest” designation.
- `PROGRAM_CANDIDATES_ENABLED=false` disables both the extra view and priority
  reservation for controlled comparison. Provisional branch default is true.

**Why reserve a slot after reranking?** The known CS catalog total scores about
0.575 while unrelated Statistics PDFs score about 0.97. Simply adding the correct
chunk to the pool still loses it to the final top-k. Priority is a transparent
source-identity rule, not a claim that these scores are comparable confidence
probabilities or that metadata proves semantic support.

## Live read-only comparison

Artifacts: `corpus-retrieval-control.json` and `corpus-retrieval-metadata.json`.
Both use the same production implementation (hash recorded), hand-specified
standalone queries plus original wording, and the same collection/budgets.
**No planner or answer calls were made.** The comparison is one pass per setting,
not a robust latency or statistical quality benchmark.

| Case | Observed result with metadata path |
|---|---|
| Three CS credit paraphrases | `d3556d267f8ac571#0006`, “Complete 123 credit hours,” selected first and retained in bounded evidence in **3/3**. None selected this chunk in the disabled control. Metadata says undergraduate Computer Science Major, 2026–2027. |
| CS math follow-up | Matching catalog page recovered, but the selected passage concerns electives. **Not an answer-quality pass**: the required-course table is still absent from this card. |
| Switch to mathematics major | Matching Mathematics Major catalog card recovered. Required-total claims still need passage-level review. |
| Data-Centric Computing | No priority match; selected chunk-ID set unchanged. |
| Explicit CS 2023–24 request | No priority match to the 2026–2027 catalog; selected chunk-ID set unchanged. |
| Admissions and parking | Selected chunk-ID sets unchanged. |
| Unknown person and unknown program | Selected chunk-ID sets unchanged; not necessarily empty. **Refusal remains an answer-model requirement**, not something this change proves. |

All candidate counts stayed within 30/search and final evidence within 12 chunks.
Mean measured retrieval/evidence time was roughly one second in both settings on
this small sample. These calls did not exercise generation latency.

## Remaining limitations

- Wrong-program, old-proposal, and sibling-program PDFs still occupy general
  slots. Sparse/absent metadata on those sources is not repaired here.
- Strict name matching deliberately misses aliases and some specialized-program
  naming variants. A no-match leaves general retrieval available; it does not
  declare a program nonexistent.
- Year matching is conservative lexical edition matching, not a resolver for
  graduation year versus entering cohort, catalog policy, or semester applicability.
  Other editions can remain in the general evidence set; this is not a global
  edition filter.
- A program-matched passage may discuss electives instead of required courses.
  Prioritization does not verify claim entailment, and the catalog extraction gap
  remains a separate ingestion repair.
- The unknown-program probe retrieved unrelated general-policy chunks well above
  the score floor. Do not treat absence of a matching name or a high score as an
  existence test.
- ARC access remains blocked; end-to-end answers and the updated edition wording
  have not been provider-validated. Independent final review is still incomplete.

Offline checks: **158 backend tests passed**, including candidate budgets, shared
embeddings, filtered dense/sparse views, page diversity, wrong-name/year handling,
major/minor/option distinctions, score-floor enforcement, and metadata budgets.
The last Chromium run passed **6 tests**. Python 3.11 syntax, dependency lock, and
whitespace checks passed.

## Reproduce

```bash
PYTHONPATH=services/chatbot/src python services/chatbot/scripts/audit_retrieval_coverage.py \
  --port-forward --disable-program-candidates \
  --output docs/evaluations/corpus-retrieval-control.json
PYTHONPATH=services/chatbot/src python services/chatbot/scripts/audit_retrieval_coverage.py \
  --port-forward --output docs/evaluations/corpus-retrieval-metadata.json
```
