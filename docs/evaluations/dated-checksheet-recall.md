# Remaining dated-checksheet recall gap

Read-only GPT diagnosis, 2026-09-24; context `endeavour`, namespace `test`, original collection `hokiehelp-test`. No source/index edits or model calls were made for the diagnostic retrieval probe.

## CS 2023–24

The total is **already indexed**:

- `21d8ef4e32ef41d9#0000`: `C REDITS R EQUIRED FOR G RADUATION : 123`.
- `21d8ef4e32ef41d9#0004`: `minimum of 123 credits`.

A normal primary query, `Virginia Tech undergraduate CS (Computer Science) BS degree checksheet total required credit hours for 2023-24`, plus original `How many credits for the 2023-24 CS major?`, misses both passages in its top-30 candidates. Selected passages include `#0005`, `#0006`, `#0015`, `#0001`, which do not state the total.

An **offline diagnostic only** query, `CS Major 2023 24 C REDITS R EQUIRED FOR G RADUATION`, retrieves `#0000` at candidate rank 2. It then survives reranking (0.8895), selection (rank 6), and the evidence budget. This identifies candidate recall, not citation filtering or evidence truncation, as the demonstrated failure stage.

## Data-Centric Computing

`2e350e9088bedeb9#0000` contains the same title-page heading and total. Current retrieval selects a passage such as `#0004` mentioning a minimum of 123 credits, but the real API probe still declines to state an overall total. The diagnostic OCR-shaped query recovers the title-page candidate, but it reranks too low against the original primary query; this case also has a reranking interaction.

## Status

- Actual API answers and failed CS-total check: `cluster-canary-edition-controls.json`.
- These are safe but unhelpful partial answers, **not** successful credit-total answers.
- Repairing HTML catalog tables does not repair these PDFs. The original bucket/collection remain unchanged.
- No hardcoded credit answer or OCR-shaped supplemental query has been added to the application.
- A follow-up should evaluate conservative PDF heading normalization and reembedding in the isolated collection, preserving source text/provenance and checking actual semantic recall. Simply normalizing text at answer time cannot recover a passage that never entered the candidate pool.
