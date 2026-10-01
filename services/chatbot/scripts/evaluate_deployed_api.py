#!/usr/bin/env python3
"""Public synthetic questions against the actual deployed HTTP + SSE API.

Run in the cluster. Does not read credentials, modify the index, or mock models.
Checks are regression signals; saved answers still require passage-level review.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import httpx


CASES = [
    ('greeting', 'Hi!', []),
    ('credits-complete', 'how many credits do i need to complete bs in cs', []),
    ('credits-graduate', 'how many credits do i need to graduate BS in CS', []),
    ('credits-undergraduate', 'How many credits do i need to complete undergraduate in cs', []),
    ('math-followup', 'What math courses does it require?', 'cs'),
    ('math-major-switch', 'What about the Mathematics major instead?', 'cs'),
    ('math-catalog-2026', 'For the 2026-2027 Computer Science major catalog, which mathematics courses are required, including allowed alternatives?', []),
    ('cs-2023', 'How many credits does the 2023-24 CS major checksheet require?', []),
    ('data-centric', 'How many credits for the Data-Centric Computing major?', []),
    ('unknown-person', 'What courses does Professor Quasar Noodlewick teach?', []),
    ('unknown-program', 'What are the Chronoengineering degree requirements?', []),
    ('ambiguous', 'What does it require?', []),
    ('parking', 'Where can I get a Virginia Tech student parking permit?', []),
]


def run(args):
    report = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'url': args.url,
              'method': 'Actual deployed /chat and /chat/stream; no mocked planner, retrieval, evidence or model',
              'limitations': ['Heuristic checks do not prove semantic entailment.',
                              'Single sequential pass, not a load test.'], 'cases': []}
    cs_history = []
    wanted = set(args.cases.split(',')) if args.cases else None
    if wanted and wanted - {case[0] for case in CASES}:
        raise ValueError('Unknown case ID')

    def checkpoint():
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')

    with httpx.Client(base_url=args.url, timeout=200) as client:
        health = client.get('/ready')
        health.raise_for_status()
        report['readiness'] = health.json()
        for cid, question, history in CASES:
            if wanted and cid not in wanted:
                continue
            if history == 'cs':
                if not cs_history:
                    cs_history = [{'role': 'user', 'content': 'Tell me about the undergraduate Computer Science major at Virginia Tech.'}]
                history = cs_history
            started = time.monotonic()
            row = {'id': cid, 'question': question, 'history': history}
            try:
                response = client.post('/chat', json={'question': question, 'history': history})
                row['http_status'] = response.status_code
                response.raise_for_status()
                result = response.json()
                row.update(result)
                answer, sources = result['answer'], result['sources']
                citations = set(map(int, re.findall(r'\[(\d+)\]', answer)))
                ids = {s['id'] for s in sources}
                checks = {'nonempty': bool(answer.strip()), 'source_ids_match_citations': citations == ids}
                if cid.startswith('credits-') or cid == 'cs-2023':
                    checks['mentions_123'] = bool(re.search(r'\b123\b', answer))
                    checks['no_wrong_program_url'] = not any(re.search(r'secure|data.centric|statistic|cos_stat|cals_', s['url'], re.I) for s in sources)
                    checks['has_cited_sources'] = bool(sources)
                if cid == 'math-followup':
                    checks['contains_three_expected_math_codes'] = all(re.search(r'MATH\s+' + code, answer, re.I) for code in ('1225', '1226', '2114'))
                    checks['no_math_major_checksheet'] = not any('cos_math' in s['url'].lower() for s in sources)
                if cid == 'math-catalog-2026':
                    checks['required_and_alternative_math_codes_present'] = all(
                        re.search(r'MATH\s+' + code, answer, re.I)
                        for code in ('1225', '1226', '2114', '2204', '2534', '3034', '3134', '3124')
                    ) and bool(re.search(r'CMDA\s+2005', answer, re.I))
                    checks['catalog_cited'] = any('/computer-science-bs/' in s['url'] for s in sources)
                    checks['edition_identified'] = '2026' in answer and '2027' in answer
                    # Presence does not verify 'OR' relationships or distinguish
                    # required courses from elective examples: review full answer.
                if cid in {'greeting', 'ambiguous', 'unknown-person', 'unknown-program'}:
                    checks['no_sources'] = not sources
                row['checks'] = checks
                if cid == 'credits-complete':
                    cs_history = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            except Exception as exc:
                row['error_type'] = type(exc).__name__
            row['seconds'] = round(time.monotonic() - started, 3)
            report['cases'].append(row)
            checkpoint()
            print(json.dumps({'id': cid, 'seconds': row['seconds'], 'checks': row.get('checks'), 'error_type': row.get('error_type')}), flush=True)
            if row.get('http_status') in {401, 403, 429, 502, 503}:
                report['stopped_on_error'] = cid
                checkpoint()
                break

        if args.stream:
            started = time.monotonic()
            events = []
            with client.stream('POST', '/chat/stream', json={'question': 'How many credits for the undergraduate Computer Science major?', 'history': []}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: '): events.append(json.loads(line[6:]))
            final = next((e for e in events if e['type'] == 'answer'), {})
            sources = next((e for e in events if e['type'] == 'sources'), {})
            report['stream'] = {'seconds': round(time.monotonic() - started, 3),
                'event_types': [e['type'] for e in events], 'answer': final.get('content'),
                'sources': sources.get('sources'),
                'checks': {'authoritative_answer': bool(final.get('content')),
                           'done_last': bool(events) and events[-1]['type'] == 'done',
                           'no_error': all(e['type'] != 'error' for e in events)}}
            checkpoint()
    report['heuristic_pass'] = bool(report['cases']) and all(
        row.get('http_status') == 200 and bool(row.get('checks')) and all(row['checks'].values())
        for row in report['cases']
    ) and not report.get('stopped_on_error')
    if args.stream:
        report['heuristic_pass'] = report['heuristic_pass'] and all(report['stream']['checks'].values())
    checkpoint()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', default='/tmp/deployed-api-evaluation.json')
    parser.add_argument('--cases', help='Optional comma-separated case IDs')
    parser.add_argument('--stream', action='store_true')
    result = run(parser.parse_args())
    raise SystemExit(0 if result['heuristic_pass'] else 1)
