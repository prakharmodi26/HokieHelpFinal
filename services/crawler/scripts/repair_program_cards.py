#!/usr/bin/env python3
"""Re-render existing catalog program URLs into an ISOLATED repair bucket.

Never writes to the source bucket or Qdrant. Saves original cards and fetched HTML;
fails individual URLs closed on missing requirements/edition or course-code loss.
Use the pinned repair crawler image, then the compatible deployed BGE-M3 ingestor.
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import yaml
from bs4 import BeautifulSoup
from minio import Minio
from playwright.async_api import async_playwright
from crawler.courseleaf import (
    detect_catalog_edition,
    parse_program_detail,
    program_card_markdown,
    program_requirement_containers,
)

PREFIX = "catalog.vt.edu/programs/"
CODE = re.compile(r"\b([A-Z&]{2,8})\s+(\d{4}[A-Z]?)\b")


def codes(text):
    return {a + " " + b for a, b in CODE.findall(text)}


def parse_card(raw):
    parts = re.split(r"^---\s*$", raw, maxsplit=2, flags=re.MULTILINE)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("Invalid markdown frontmatter")
    return yaml.safe_load(parts[1]), parts[2]


def validated_card(html, url, old_body):
    soup = BeautifulSoup(html, "html.parser")
    panes = program_requirement_containers(soup)
    source_text = " ".join(pane.get_text(" ", strip=True) for pane in panes)
    if not panes or len(source_text) < 100:
        raise ValueError("Missing or empty requirements/curriculum pane")
    edition = detect_catalog_edition(html)
    if not edition:
        raise ValueError("No source edition")
    record = parse_program_detail(html, url)
    if record is None:
        raise ValueError("Rejected program record")
    card = program_card_markdown(record, edition)
    frontmatter, body = parse_card(card)
    missing = codes(source_text) - codes(body)
    if missing:
        raise ValueError("Course-code loss: " + ", ".join(sorted(missing)))
    if len(body) < min(500, len(old_body) * 0.5):
        raise ValueError("Unexpected body shrinkage")
    table_count = sum(len(pane.find_all("table")) for pane in panes)
    if table_count and "| ---" not in body:
        raise ValueError("Source tables missing from body")
    return card, {"title": record.title, "edition": edition, "table_count": table_count,
                  "source_course_codes": len(codes(source_text)),
                  "body_chars": len(body), "doc_id": frontmatter["doc_id"],
                  "content_hash": frontmatter["content_hash"]}


async def fetch_robots_policy(context):
    """Fetch robots.txt through the browser session, not raw HTTP.

    catalog.vt.edu can return an initial anti-bot HTTP 202 to non-browser
    requests.  Navigating with the warmed Chromium context respects the normal
    challenge flow while still failing closed if robots cannot be established.
    """
    robots_page = await context.new_page()
    try:
        response = await robots_page.goto(
            'https://catalog.vt.edu/robots.txt',
            wait_until='domcontentloaded',
            timeout=30000,
        )
        status = response.status if response is not None else None
        if status not in (200, 404):
            raise ValueError(f'Cannot establish robots policy: {status}')
        body = await robots_page.text_content('body') or ''
        robots = RobotFileParser()
        robots.parse(body.splitlines() if status == 200 else [])
        return robots
    finally:
        await robots_page.close()


async def run(args):
    if args.bucket == args.source_bucket or not args.bucket.startswith("hokiehelp-quality-"):
        raise ValueError("Target must be an isolated repair bucket")
    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    client = Minio(endpoint.removeprefix("http://").removeprefix("https://"),
                   access_key=os.environ["MINIO_ACCESS_KEY"], secret_key=os.environ["MINIO_SECRET_KEY"],
                   secure=endpoint.startswith("https://"))
    if not client.bucket_exists(args.bucket):
        client.make_bucket(args.bucket)

    def upload(key, raw, content_type):
        if isinstance(raw, str): raw = raw.encode()
        client.put_object(args.bucket, key, io.BytesIO(raw), len(raw), content_type=content_type)

    records = []
    for obj in client.list_objects(args.source_bucket, prefix=PREFIX, recursive=True):
        if not obj.object_name.endswith('.md'): continue
        response = client.get_object(args.source_bucket, obj.object_name)
        try: raw = response.read().decode()
        finally: response.close(); response.release_conn()
        frontmatter, body = parse_card(raw)
        url = frontmatter.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "catalog.vt.edu": continue
        if not (parsed.path.startswith('/undergraduate/') or parsed.path.startswith('/graduate/')): continue
        records.append((obj.object_name, url, raw, body))
    if args.include_url_regex:
        include = re.compile(args.include_url_regex)
        records = [record for record in records if include.search(record[1])]
    # Regression sources first, then stable order. One browser request at a time.
    records.sort(key=lambda r: (not any(k in r[1] for k in ("computer-science", "mathematics")), r[0]))
    if args.limit: records = records[:args.limit]
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "source_bucket": args.source_bucket,
              "target_bucket": args.bucket, "attempted": len(records), "ok": [], "failed": []}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        context = await browser.new_context()
        page = await context.new_page()
        # Establish the normal browser session before reading site robots policy.
        await page.goto('https://catalog.vt.edu/', wait_until='networkidle', timeout=60000)
        await page.wait_for_selector('#site-title', timeout=30000)
        robots = await fetch_robots_policy(context)
        for key, url, original, old_body in records:
            try:
                if not robots.can_fetch('*', url): raise ValueError('Disallowed by robots')
                statuses = []
                def response_seen(response):
                    if response.request.resource_type == 'document': statuses.append(response.status)
                page.on('response', response_seen)
                try:
                    await page.goto(url, wait_until='networkidle', timeout=60000)
                    # VT's live CourseLeaf program pages do not expose a single
                    # #requirementstextcontainer.  Wait for any source-of-record
                    # requirements/curriculum pane that program_requirement_containers()
                    # will preserve; intentionally exclude #roadmaptextcontainer so
                    # recommended semester plans are not conflated with requirements.
                    await page.wait_for_selector(
                        '#requirementstextcontainer, '
                        '#programcurriculumtextcontainer, '
                        '#graduationrequirementstextcontainer, '
                        '[id$="curriculumtextcontainer"], '
                        '[id$="requirementstextcontainer"], '
                        '[id$="planofstudytextcontainer"]',
                        timeout=20000,
                    )
                    html = await page.content()
                finally:
                    page.remove_listener('response', response_seen)
                if not statuses or statuses[-1] != 200:
                    raise ValueError('Final document response not HTTP 200')
                if page.url.rstrip('/') != url.rstrip('/'):
                    raise ValueError('Unexpected final URL')
                card, detail = validated_card(html, url, old_body)
                upload('originals/' + key, original, 'text/markdown')
                upload('raw/' + key.removesuffix('.md') + '.html', html, 'text/html')
                upload(key, card, 'text/markdown')
                report['ok'].append(dict(key=key, url=url, html_sha256=hashlib.sha256(html.encode()).hexdigest(), **detail))
                print(json.dumps({'ok': url, 'tables': detail['table_count']}), flush=True)
            except Exception as exc:
                report['failed'].append({'url': url, 'error_type': type(exc).__name__, 'reason': str(exc)[:240]})
                print(json.dumps({'failed': url, 'error_type': type(exc).__name__}), flush=True)
            upload('reports/program-repair.json', json.dumps(report, indent=2), 'application/json')
        await browser.close()
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    upload('reports/program-repair.json', json.dumps(report, indent=2), 'application/json')
    print(json.dumps({'succeeded': len(report['ok']), 'failed': len(report['failed'])}), flush=True)
    # Failed pages remain untouched in the cloned index, not erased as empty data.
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bucket', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--include-url-regex', default='', help='Only repair source cards whose URL matches this regex')
    asyncio.run(run(parser.parse_args()))
