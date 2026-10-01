const { test, expect } = require('@playwright/test');

function sse(events) {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');
}

async function openApp(page) {
  await page.route('**/*', (route) => {
    const url = new URL(route.request().url());
    if (url.hostname === '127.0.0.1' || url.hostname === 'localhost') {
      return route.continue();
    }
    return route.abort();
  });
  await page.goto('/');
}

async function installFetchQueue(page, responses) {
  await page.evaluate((mockResponses) => {
    window.__mockFetchQueue = mockResponses;
    window.__mockFetchRequests = [];
    window.__mockFetchResolvers = {};

    window.fetch = async (input, init) => {
      const spec = window.__mockFetchQueue.shift();
      if (!spec) throw new Error('Unexpected fetch: ' + input);

      let body = null;
      try { body = init && init.body ? JSON.parse(init.body) : null; } catch (_) { body = init && init.body; }
      window.__mockFetchRequests.push({
        url: String(input),
        method: init && init.method,
        headers: init && init.headers,
        body,
      });

      if (spec.holdKey) {
        await new Promise((resolve) => { window.__mockFetchResolvers[spec.holdKey] = resolve; });
      }

      const headers = spec.headers || { 'content-type': 'text/event-stream; charset=utf-8' };
      const status = spec.status || 200;
      const text = spec.body || '';
      const encoded = new TextEncoder().encode(text);
      const splitAt = (spec.splitAt || []).filter((n) => n > 0 && n < encoded.length).sort((a, b) => a - b);

      const stream = new ReadableStream({
        start(controller) {
          let start = 0;
          for (const end of splitAt) {
            controller.enqueue(encoded.slice(start, end));
            start = end;
          }
          controller.enqueue(encoded.slice(start));
          controller.close();
        },
      });

      return new Response(stream, { status, headers });
    };
  }, responses);
}

async function ask(page, question) {
  await page.locator('#question-input').fill(question);
  await page.locator('#send-btn').click();
}

test('index copy presents HokieHelp as a university-wide VT assistant', async ({ page }) => {
  await openApp(page);

  await expect(page).toHaveTitle('HokieHelp — Virginia Tech Assistant');
  await expect(page.locator('.logo-subtitle')).toHaveText('VT AI Assistant');
  await expect(page.locator('#question-input')).toHaveAttribute('placeholder', 'Ask about Virginia Tech...');
  await expect(page.locator('.input-hint')).toContainText('Virginia Tech knowledge sources');
  await expect(page.locator('body')).not.toContainText(/CS AI Assistant|CS Assistant|Virginia Tech CS|website\.cs\.vt\.edu/);
});

test('renders authoritative final answer with safe citations and collapsed supplied sources only', async ({ page }) => {
  await openApp(page);
  await installFetchQueue(page, [{
    body: sse([
      { type: 'token', content: 'Provisional fallback text that should disappear.' },
      {
        type: 'answer',
        content: 'Authoritative <img src=x onerror="window.__xss=1"> answer with [1], [2], [99], and a bad [link](javascript:window.__xss=2).',
      },
      {
        type: 'sources',
        sources: [
          { id: 1, title: '<img src=x onerror="window.__xss=3"> Catalog', url: 'https://vt.edu/catalog?source=1' },
          { id: 2, title: 'Unsafe source', url: 'javascript:window.__xss=4' },
          { id: 3, title: 'Faculty <script>window.__xss=5</script>', url: 'https://www.vt.edu/faculty' },
        ],
      },
      { type: 'done' },
    ]),
  }]);

  await ask(page, 'What sources are used?');
  const bubble = page.locator('.message.assistant').last();
  await expect(bubble).toContainText('Authoritative <img src=x onerror="window.__xss=1"> answer');
  await expect(bubble).not.toContainText('Provisional fallback text');

  await expect(bubble.locator('img[src="x"]')).toHaveCount(0);
  await expect(bubble.locator('script')).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => window.__xss)).toBe(undefined);

  await expect(bubble.locator('a.citation')).toHaveCount(1);
  await expect(bubble.locator('a.citation')).toHaveText('[1]');
  await expect(bubble.locator('a.citation')).toHaveAttribute('href', 'https://vt.edu/catalog?source=1');

  const hrefs = await bubble.locator('a').evaluateAll((anchors) => anchors.map((a) => a.getAttribute('href')));
  expect(hrefs).not.toContain('javascript:window.__xss=2');
  expect(hrefs).not.toContain('javascript:window.__xss=4');

  await expect(bubble.locator('.sources-toggle')).toHaveAttribute('aria-expanded', 'false');
  await expect(bubble.locator('.sources-list')).not.toBeVisible();
  await expect(bubble.locator('.source-link')).toHaveCount(1);
  await expect(bubble.locator('.source-link').nth(0)).toContainText('[1] <img src=x onerror="window.__xss=3"> Catalog');
  await expect(bubble.locator('.source-link').nth(0)).toHaveAttribute('href', 'https://vt.edu/catalog?source=1');
  await expect(bubble.locator('.sources')).not.toContainText('[3] Faculty');
  await expect(bubble.locator('.sources')).not.toContainText('[2] Unsafe source');
  await expect(bubble.locator('.sources')).not.toContainText('[99]');

  await bubble.locator('.sources-toggle').click();
  await expect(bubble.locator('.sources-toggle')).toHaveAttribute('aria-expanded', 'true');
  await expect(bubble.locator('.sources-list')).toBeVisible();
});

test('server error events show an error and failed turns are not sent as next-request history', async ({ page }) => {
  await openApp(page);
  await installFetchQueue(page, [
    { body: sse([{ type: 'token', content: 'partial text' }, { type: 'error', content: 'backend exploded' }]) },
    { body: sse([{ type: 'answer', content: 'Recovered answer.' }, { type: 'sources', sources: [] }, { type: 'done' }]) },
  ]);

  await ask(page, 'first failing question');
  await expect(page.locator('.message.error').last()).toContainText('Error: backend exploded');

  await ask(page, 'second clean question');
  await expect(page.locator('.message.assistant').last()).toContainText('Recovered answer.');

  const requests = await page.evaluate(() => window.__mockFetchRequests);
  expect(requests).toHaveLength(2);
  expect(requests[1].body.history).toEqual([]);
});

test('premature EOF shows an incomplete-response error and is excluded from next-request history', async ({ page }) => {
  await openApp(page);
  await installFetchQueue(page, [
    { body: sse([{ type: 'answer', content: 'Incomplete answer without done.' }]) },
    { body: sse([{ type: 'answer', content: 'Clean answer after EOF.' }, { type: 'sources', sources: [] }, { type: 'done' }]) },
  ]);

  await ask(page, 'question that ends early');
  await expect(page.locator('.message.error').last()).toContainText('Error: The response was incomplete. Please try again.');

  await ask(page, 'question after EOF');
  await expect(page.locator('.message.assistant').last()).toContainText('Clean answer after EOF.');

  const requests = await page.evaluate(() => window.__mockFetchRequests);
  expect(requests).toHaveLength(2);
  expect(requests[1].body.history).toEqual([]);
});

test('new chat and inputs are disabled while a response is in flight and re-enabled afterwards', async ({ page }) => {
  await openApp(page);
  await installFetchQueue(page, [{
    holdKey: 'releaseResponse',
    body: sse([{ type: 'answer', content: 'Delayed final answer.' }, { type: 'sources', sources: [] }, { type: 'done' }]),
  }]);

  await ask(page, 'hold open');
  await page.waitForFunction(() => window.__mockFetchRequests.length === 1);
  await expect(page.locator('#new-chat-btn')).toBeDisabled();
  await expect(page.locator('#question-input')).toBeDisabled();
  await expect(page.locator('#send-btn')).toBeDisabled();

  await page.evaluate(() => window.__mockFetchResolvers.releaseResponse());
  await expect(page.locator('.message.assistant').last()).toContainText('Delayed final answer.');
  await expect(page.locator('#new-chat-btn')).toBeEnabled();
  await expect(page.locator('#question-input')).toBeEnabled();
  await expect(page.locator('#send-btn')).toBeEnabled();
});

test('handles UTF-8 characters and SSE records split across byte/chunk boundaries', async ({ page }) => {
  await openApp(page);
  const answer = 'Authoritative UTF-8 🦃 answer with Café [1].';
  const body = sse([
    { type: 'token', content: 'Draft' },
    { type: 'answer', content: answer },
    { type: 'sources', sources: [{ id: 1, title: 'UTF-8 source', url: 'https://vt.edu/utf8' }] },
    { type: 'done' },
  ]);
  const emojiByte = Buffer.from(body.slice(0, body.indexOf('🦃'))).length + 2;
  const splitAt = [1, 7, emojiByte, emojiByte + 1, body.length - 3];
  await installFetchQueue(page, [{ body, splitAt }]);

  await ask(page, 'split stream');
  const bubble = page.locator('.message.assistant').last();
  await expect(bubble).toContainText(answer);
  await expect(bubble.locator('a.citation')).toHaveText('[1]');
  await expect(bubble.locator('a.citation')).toHaveAttribute('href', 'https://vt.edu/utf8');
});
