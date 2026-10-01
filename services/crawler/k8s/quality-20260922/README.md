# hokiehelp-quality-20260922 catalog repair

Reproducible manifests for the isolated CourseLeaf program-card repair. These target only:

- bucket: `hokiehelp-quality-20260922`
- Qdrant collection: `hokiehelp-quality-20260922`

Do not run these against `test-crawled-pages-0705` or `hokiehelp-test` as targets.

## 1) Create/refresh patched crawler overlay

```bash
kubectl --context endeavour -n test delete configmap quality-catalog-repair-overlay --ignore-not-found
kubectl --context endeavour -n test create configmap quality-catalog-repair-overlay \
  --from-file=courseleaf.py=services/crawler/src/crawler/courseleaf.py \
  --from-file=program_content.py=services/crawler/src/crawler/program_content.py \
  --from-file=validate.py=services/crawler/src/crawler/validate.py \
  --from-file=repair_program_cards.py=services/crawler/scripts/repair_program_cards.py
kubectl --context endeavour -n test label configmap quality-catalog-repair-overlay \
  managed-by=hokiehelp-pipeline repair-target=hokiehelp-quality-20260922 --overwrite
```

## 2) Acceptance-first repair (CS/Math)

Set `REPAIR_ARGS` in `catalog-repair-job.yaml` to:

```text
--include-url-regex '(computer-science|mathematics)'
```

Then run:

```bash
kubectl --context endeavour -n test delete job quality-catalog-repair --ignore-not-found
kubectl --context endeavour -n test apply -f services/crawler/k8s/quality-20260922/catalog-repair-job.yaml
```

## 3) Full program repair

Set `REPAIR_ARGS` back to the empty string in the manifest and re-apply the same job. The script writes `reports/program-repair.json` into the target bucket as it progresses.

## 4) Program-prefix ingest

After repaired cards are in the target bucket:

```bash
kubectl --context endeavour -n test delete job quality-program-ingest --ignore-not-found
kubectl --context endeavour -n test apply -f services/crawler/k8s/quality-20260922/program-ingest-job.yaml
```

The ingest job uses `INGEST_PREFIX=catalog.vt.edu/programs/`; `backups/` is outside that prefix.
