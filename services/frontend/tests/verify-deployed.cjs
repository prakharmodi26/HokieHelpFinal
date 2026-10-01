// Real browser -> deployed nginx -> actual backend/models/index. No routes or fetch mocks.
// node services/frontend/tests/verify-deployed.cjs https://hokiehelp.endeavour.cs.vt.edu OUTPUT.json
const { chromium, expect } = require('@playwright/test');
const fs = require('node:fs');

(async () => {
  const url = process.argv[2] || 'https://hokiehelp.endeavour.cs.vt.edu';
  const output = process.argv[3] || '/tmp/hokiehelp-live-browser.json';
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  const report = { generated_utc: new Date().toISOString(), url, mocked: false,
    console_errors: [], page_errors: [], failed_requests: [], turns: [] };
  page.on('console', m => { if (m.type() === 'error') report.console_errors.push(m.text()); });
  page.on('pageerror', e => report.page_errors.push(e.message));
  page.on('requestfailed', r => report.failed_requests.push({ url: r.url(), error: r.failure()?.errorText }));
  try {
    await page.goto(url, { waitUntil: 'networkidle' });
    await expect(page).toHaveTitle('HokieHelp — Virginia Tech Assistant');
    for (const question of ['How many credits do I need for a BS in Computer Science?', 'What math courses does it require?']) {
      const started = Date.now();
      const responsePromise = page.waitForResponse(r => r.url().endsWith('/api/chat/stream') && r.request().method() === 'POST', { timeout: 180000 });
      await page.locator('#question-input').fill(question);
      await page.locator('#send-btn').click();
      await expect(page.locator('#question-input')).toBeDisabled();
      const response = await responsePromise;
      const body = await response.text();
      const events = body.split('\n').filter(l => l.startsWith('data: ')).map(l => JSON.parse(l.slice(6)));
      await expect(page.locator('#send-btn')).toBeEnabled({ timeout: 180000 });
      const bubble = page.locator('.message.assistant').last();
      await expect(bubble).not.toHaveClass(/error/);
      const final = events.find(e => e.type === 'answer');
      const sources = events.find(e => e.type === 'sources')?.sources || [];
      const turn = { question, seconds: (Date.now() - started) / 1000, status: response.status(),
        content_type: response.headers()['content-type'], request: response.request().postDataJSON(),
        final_answer: final?.content, sources, rendered_text: await bubble.innerText(),
        event_types: [...new Set(events.map(e => e.type))],
        citations: await bubble.locator('a.citation').evaluateAll(as => as.map(a => ({ text: a.textContent, href: a.href }))) };
      report.turns.push(turn);
      expect(response.status()).toBe(200);
      expect(turn.content_type).toContain('text/event-stream');
      expect(final?.content).toBeTruthy();
      expect(events.at(-1)?.type).toBe('done');
      expect(events.some(e => e.type === 'error')).toBe(false);
      expect(turn.citations.length).toBeGreaterThan(0);
      const supplied = new Set(sources.map(s => s.url));
      expect(turn.citations.every(c => supplied.has(c.href))).toBe(true);
      await expect(bubble.locator('.sources-toggle')).toHaveAttribute('aria-expanded', 'false');
      await bubble.locator('.sources-toggle').click();
      await expect(bubble.locator('.sources-list')).toBeVisible();
      expect(await bubble.locator('.source-link').count()).toBe(sources.length);
      if (report.turns.length === 1) expect(turn.rendered_text).toMatch(/\b123\b/);
      else {
        for (const course of ['1225', '1226', '2114']) expect(turn.rendered_text).toMatch(new RegExp('MATH\\s+' + course));
        expect(turn.request.history).toHaveLength(2);
        expect(turn.request.history[1].content).toBe(report.turns[0].final_answer);
      }
    }
    await page.screenshot({ path: output.replace(/\.json$/, '') + '.png', fullPage: true });
    expect(report.page_errors).toEqual([]);
    expect(report.console_errors).toEqual([]);
    expect(report.failed_requests).toEqual([]);
    report.passed = true;
  } catch (e) {
    report.passed = false;
    report.error = e.message;
    process.exitCode = 1;
  } finally {
    fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
    await browser.close();
    console.log(JSON.stringify({ passed: report.passed, turns: report.turns.length, output, error: report.error }));
  }
})();
