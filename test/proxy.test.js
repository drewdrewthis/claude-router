'use strict';
const test = require('node:test');
const assert = require('node:assert');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { startServer } = require('../router');

delete process.env.ROUTER_ANTHROPIC_API_KEY; // force incoming-header credential reuse

const TIERS = { light: 'claude-haiku-4-5', standard: 'claude-sonnet-5', heavy: 'claude-opus-4-8' };

function baseConfig(over = {}) {
  return {
    mode: 'session',
    tiers: TIERS,
    pinned: [],
    classifierModel: 'claude-haiku-4-5',
    classifierTimeoutMs: 3000,
    maxHaikuInputTokens: 150000,
    upstream: 'http://unused',
    port: 0,
    ...over,
  };
}

function isClassifier(bodyStr) {
  try {
    const j = JSON.parse(bodyStr);
    return typeof j.system === 'string' && j.system.includes('exactly one word');
  } catch {
    return false;
  }
}

// Mock upstream. `respond(ctx)` is called for FORWARDED (non-classifier) requests.
// Classifier requests are auto-answered with `label` (default 'heavy').
function startMock({ label = 'heavy', respond } = {}) {
  const requests = [];
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const body = Buffer.concat(chunks);
      const str = body.toString('utf8');
      const rec = { method: req.method, url: req.url, headers: req.headers, body, classifier: isClassifier(str) };
      requests.push(rec);
      if (rec.classifier) {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ content: [{ type: 'text', text: label }] }));
        return;
      }
      if (respond) return respond({ req, body, res });
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true, seen_model: safeModel(str) }));
    });
  });
  return new Promise((r) => {
    server.listen(0, '127.0.0.1', () =>
      r({
        url: `http://127.0.0.1:${server.address().port}`,
        requests,
        forwarded: () => requests.filter((x) => !x.classifier),
        classifierCalls: () => requests.filter((x) => x.classifier),
        close: () => new Promise((rr) => server.close(rr)),
      })
    );
  });
}

function safeModel(str) {
  try {
    return JSON.parse(str).model;
  } catch {
    return null;
  }
}

function tmpLog() {
  return path.join(os.tmpdir(), `router-decisions-${process.pid}-${Math.random().toString(36).slice(2)}.jsonl`);
}

// Minimal client that returns status, headers, raw body, and per-chunk timings.
function request(port, { method = 'POST', path: p = '/v1/messages', headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port, method, path: p, headers },
      (res) => {
        const chunks = [];
        const timings = [];
        const t0 = Date.now();
        res.on('data', (c) => {
          chunks.push(c);
          timings.push({ t: Date.now() - t0, s: c.toString('utf8') });
        });
        res.on('end', () =>
          resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks), timings })
        );
      }
    );
    req.on('error', reject);
    if (body != null) req.write(body);
    req.end();
  });
}

test('rewrite: complex body (tool_use, base64 image, cache_control) deep-equals original except model', async () => {
  const mock = await startMock({ label: 'standard' });
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile: tmpLog() });
  const original = {
    model: 'claude-3-5-sonnet-20241022',
    max_tokens: 1024,
    system: [{ type: 'text', text: 'you are helpful', cache_control: { type: 'ephemeral' } }],
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'look at this' },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgoAAAANS' } },
          { type: 'tool_use', id: 'tu_1', name: 'calc', input: { a: 1 } },
        ],
      },
    ],
    tools: [{ name: 'calc', input_schema: { type: 'object' } }],
  };
  await request(srv.port, { headers: { 'content-type': 'application/json' }, body: JSON.stringify(original) });

  const fwd = mock.forwarded();
  assert.strictEqual(fwd.length, 1);
  const got = JSON.parse(fwd[0].body.toString('utf8'));
  assert.strictEqual(got.model, TIERS.standard);
  const a = { ...got, model: '_' };
  const b = { ...original, model: '_' };
  assert.deepStrictEqual(a, b);

  await srv.close();
  await mock.close();
});

test('passthrough (already-haiku) forwards byte-identical body, no classify', async () => {
  const mock = await startMock();
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile: tmpLog() });
  const raw = JSON.stringify({ model: 'claude-haiku-4-5', messages: [{ role: 'user', content: 'hi' }] });
  await request(srv.port, { headers: { 'content-type': 'application/json' }, body: raw });

  assert.strictEqual(mock.classifierCalls().length, 0);
  const fwd = mock.forwarded();
  assert.strictEqual(fwd.length, 1);
  assert.strictEqual(fwd[0].body.toString('utf8'), raw);

  await srv.close();
  await mock.close();
});

test('pinned forwards unchanged and never classifies', async () => {
  const mock = await startMock();
  const srv = await startServer({ config: baseConfig({ pinned: ['claude-opus'] }), upstream: mock.url, logFile: tmpLog() });
  const raw = JSON.stringify({ model: 'claude-opus-4-8', messages: [{ role: 'user', content: 'hi' }] });
  await request(srv.port, { headers: { 'content-type': 'application/json' }, body: raw });

  assert.strictEqual(mock.classifierCalls().length, 0);
  assert.strictEqual(mock.forwarded()[0].body.toString('utf8'), raw);

  await srv.close();
  await mock.close();
});

test('auth fidelity: authorization + anthropic-beta arrive byte-identical', async () => {
  const mock = await startMock({ label: 'standard' });
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile: tmpLog() });
  const auth = 'Bearer sk-ant-oat-TESTTOKEN';
  const beta = 'oauth-2025-04-20';
  await request(srv.port, {
    headers: { 'content-type': 'application/json', authorization: auth, 'anthropic-beta': beta },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'hi' }] }),
  });
  const fwd = mock.forwarded()[0];
  assert.strictEqual(fwd.headers['authorization'], auth);
  assert.strictEqual(fwd.headers['anthropic-beta'], beta);

  await srv.close();
  await mock.close();
});

test('SSE: event order preserved AND streamed (first delta before upstream finishes)', async () => {
  const mock = await startMock({
    label: 'heavy',
    respond: ({ res }) => {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
      res.write('event: message_start\ndata: {"type":"message_start"}\n\n');
      setTimeout(() => {
        res.write('event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n');
        setTimeout(() => {
          res.write('event: message_stop\ndata: {"type":"message_stop"}\n\n');
          res.end();
        }, 150);
      }, 150);
    },
  });
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile: tmpLog() });
  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'stream please' }] }),
  });
  const text = r.body.toString('utf8');
  const events = [...text.matchAll(/event: (\w+)/g)].map((m) => m[1]);
  assert.deepStrictEqual(events, ['message_start', 'content_block_delta', 'message_stop']);

  // Streaming proof: first chunk lands well before the last one. If the proxy
  // buffered, all chunks would arrive together (delta ~0).
  const firstT = r.timings[0].t;
  const lastT = r.timings[r.timings.length - 1].t;
  assert.ok(lastT - firstT >= 100, `expected streaming gap, got ${lastT - firstT}ms`);

  await srv.close();
  await mock.close();
});

test('fail-open: classifier hangs -> original model forwarded within budget, decision fallback', async () => {
  const mock = await startMock({
    respond: ({ res }) => {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
    },
  });
  // Override classifier handling: hang forever on classifier calls.
  // startMock auto-answers classifier calls, so build a dedicated server here.
  await mock.close();

  const requests = [];
  const hangServer = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const str = Buffer.concat(chunks).toString('utf8');
      requests.push({ classifier: isClassifier(str), body: str });
      if (isClassifier(str)) return; // never respond -> AbortController fires
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true, seen_model: safeModel(str) }));
    });
  });
  await new Promise((r) => hangServer.listen(0, '127.0.0.1', r));
  const upstream = `http://127.0.0.1:${hangServer.address().port}`;

  const logFile = tmpLog();
  const srv = await startServer({ config: baseConfig({ classifierTimeoutMs: 250 }), upstream, logFile });
  const t0 = Date.now();
  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'hi' }] }),
  });
  const elapsed = Date.now() - t0;
  assert.strictEqual(r.status, 200);
  assert.ok(elapsed < 2000, `fail-open should be quick, took ${elapsed}ms`);

  const fwd = requests.filter((x) => !x.classifier);
  assert.strictEqual(fwd.length, 1);
  assert.strictEqual(safeModel(fwd[0].body), 'claude-3-5-sonnet'); // ORIGINAL preserved

  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n');
  const entry = JSON.parse(lines[lines.length - 1]);
  assert.strictEqual(entry.decision, 'fallback');
  assert.ok(entry.fallback_reason);

  await srv.close();
  await new Promise((r) => hangServer.close(r));
});

test('session stickiness at proxy: same key classifies once, second is cache-hit', async () => {
  const mock = await startMock({ label: 'heavy' });
  const logFile = tmpLog();
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile });
  const raw = JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'sticky' }] });
  await request(srv.port, { headers: { 'content-type': 'application/json' }, body: raw });
  await request(srv.port, { headers: { 'content-type': 'application/json' }, body: raw });

  assert.strictEqual(mock.classifierCalls().length, 1);
  const fwd = mock.forwarded();
  assert.strictEqual(fwd.length, 2);
  assert.strictEqual(safeModel(fwd[0].body.toString('utf8')), TIERS.heavy);
  assert.strictEqual(safeModel(fwd[1].body.toString('utf8')), TIERS.heavy);

  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').map(JSON.parse);
  assert.strictEqual(lines[0].decision, 'routed');
  assert.strictEqual(lines[1].decision, 'cache-hit');

  await srv.close();
  await mock.close();
});

test('context guard at proxy: oversized body + light label -> standard tier forwarded', async () => {
  const mock = await startMock({ label: 'light' });
  const srv = await startServer({ config: baseConfig({ maxHaikuInputTokens: 10 }), upstream: mock.url, logFile: tmpLog() });
  await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'this body easily exceeds ten tokens worth of bytes' }] }),
  });
  assert.strictEqual(safeModel(mock.forwarded()[0].body.toString('utf8')), TIERS.standard);

  await srv.close();
  await mock.close();
});

test('upstream 429 with retry-after relayed unchanged', async () => {
  const mock = await startMock({
    label: 'standard',
    respond: ({ res }) => {
      res.writeHead(429, { 'retry-after': '42', 'anthropic-ratelimit-requests-remaining': '0', 'content-type': 'application/json' });
      res.end(JSON.stringify({ type: 'error', error: { type: 'rate_limit_error' } }));
    },
  });
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile: tmpLog() });
  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'hi' }] }),
  });
  assert.strictEqual(r.status, 429);
  assert.strictEqual(r.headers['retry-after'], '42');
  assert.strictEqual(r.headers['anthropic-ratelimit-requests-remaining'], '0');

  await srv.close();
  await mock.close();
});

test('malformed JSON body forwarded raw, upstream response relayed (no 500)', async () => {
  const mock = await startMock();
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile: tmpLog() });
  const raw = 'definitely not json {';
  const r = await request(srv.port, { headers: { 'content-type': 'application/json' }, body: raw });
  assert.strictEqual(r.status, 200);
  assert.strictEqual(mock.classifierCalls().length, 0);
  assert.strictEqual(mock.forwarded()[0].body.toString('utf8'), raw);

  await srv.close();
  await mock.close();
});

test('20 concurrent requests -> 20 individually-parseable log lines', async () => {
  const mock = await startMock({ label: 'standard' });
  const logFile = tmpLog();
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile });
  await Promise.all(
    Array.from({ length: 20 }, (_, i) =>
      request(srv.port, {
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'req-' + i }] }),
      })
    )
  );
  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n');
  assert.strictEqual(lines.length, 20);
  for (const l of lines) assert.doesNotThrow(() => JSON.parse(l));

  await srv.close();
  await mock.close();
});

test('non-/v1/messages path (count_tokens) untouched passthrough, no log, no classify', async () => {
  const mock = await startMock();
  const logFile = tmpLog();
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile });
  const raw = JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'count me' }] });
  const r = await request(srv.port, { method: 'POST', path: '/v1/messages/count_tokens', headers: { 'content-type': 'application/json' }, body: raw });
  assert.strictEqual(r.status, 200);
  assert.strictEqual(mock.classifierCalls().length, 0);
  const fwd = mock.forwarded()[0];
  assert.strictEqual(fwd.url, '/v1/messages/count_tokens');
  assert.strictEqual(fwd.body.toString('utf8'), raw);
  assert.strictEqual(fs.existsSync(logFile), false); // nothing logged

  await srv.close();
  await mock.close();
});

test('decision log contains no message/prompt text (sentinel absent)', async () => {
  const mock = await startMock({ label: 'heavy' });
  const logFile = tmpLog();
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile });
  const SENTINEL = 'SUPER_SECRET_PROMPT_SENTINEL_9f3a';
  await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: SENTINEL }] }),
  });
  const contents = fs.readFileSync(logFile, 'utf8');
  assert.ok(!contents.includes(SENTINEL), 'log must not contain prompt text');

  await srv.close();
  await mock.close();
});
