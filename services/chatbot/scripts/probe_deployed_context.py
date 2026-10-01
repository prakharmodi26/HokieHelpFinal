#!/usr/bin/env python3
"""Cluster-only diagnostic for public synthetic cases; never writes the index.

Records actual planner/evidence/provider finish metadata. Does NOT record API keys
or model reasoning. Optional token-budget override is explicit in the report.
"""
import argparse
import json
from pathlib import Path
import time

from chatbot.config import ChatbotConfig
from chatbot.evidence import select_evidence, finalize_answer
from chatbot.llm import LLMClient
from chatbot.retriever import Retriever


def main(args):
    cfg = ChatbotConfig.from_env()
    rows = json.loads(Path(args.input).read_text())['cases']
    wanted = set(args.cases.split(','))
    llm = LLMClient(cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model, cfg.rewriter_model,
                    timeout=150, max_tokens=args.max_tokens)
    retriever = Retriever(cfg.embedding_model, cfg.qdrant_host, cfg.qdrant_port, cfg.qdrant_collection,
                          embeddings_url=cfg.embeddings_url, candidate_k=cfg.candidate_k, top_k=cfg.top_k,
                          min_score=cfg.min_score, program_candidates_enabled=cfg.program_candidates_enabled)
    report = {'collection': cfg.qdrant_collection, 'model': cfg.llm_model,
              'max_tokens_override': args.max_tokens, 'cases': []}
    try:
        for case in rows:
            if case['id'] not in wanted: continue
            question, history = case['question'], case['history']
            plan = llm.plan_query(question, history)
            record = {'id': case['id'], 'question': question, 'history': history, 'plan': plan.model_dump()}
            if plan.intent == 'search':
                chunks = select_evidence(retriever.search_many(plan.queries, original_query=question))
                record['evidence'] = chunks
                # Intentionally inspect the raw response's finish metadata rather
                # than suppress a truncated completion through normal finalization.
                started = time.monotonic()
                response = llm._client.post('chat/completions', json=llm._body(question, chunks, history))
                record['seconds'] = round(time.monotonic() - started, 3)
                record['http_status'] = response.status_code
                response.raise_for_status()
                data = response.json()
                choice = data['choices'][0]
                record['finish_reason'] = choice.get('finish_reason')
                record['usage'] = data.get('usage')
                raw = choice['message'].get('content') or ''
                record['content'] = raw
                record['reasoning_chars'] = len(choice['message'].get('reasoning_content') or '')
                if record['finish_reason'] == 'stop' and raw.strip():
                    record['answer'], record['sources'] = finalize_answer(raw, chunks)
            report['cases'].append(record)
            Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: record.get(k) for k in ('id','plan','http_status','finish_reason','usage','seconds')}), flush=True)
    finally:
        retriever.close()
        llm.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='/tmp/post-ingest-api.json')
    parser.add_argument('--cases', default='math-followup,math-catalog-2026')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--output', default='/tmp/context-probe.json')
    main(parser.parse_args())
