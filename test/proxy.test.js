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

test('mid-SSE client abort does not crash the server; follow-up request succeeds', async () => {
  const mock = await startMock({
    label: 'heavy',
    respond: ({ res }) => {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write('event: message_start\ndata: {}\n\n');
      let n = 0;
      let alive = true;
      res.on('close', () => {
        alive = false;
      });
      const tick = () => {
        if (!alive) return;
        if (n >= 5) {
          try {
            res.write('event: message_stop\ndata: {}\n\n');
            res.end();
          } catch {}
          return;
        }
        try {
          res.write(`event: content_block_delta\ndata: {"i":${n++}}\n\n`);
        } catch {}
        setTimeout(tick, 30);
      };
      setTimeout(tick, 30);
    },
  });
  const srv = await startServer({ config: baseConfig(), upstream: mock.url, logFile: tmpLog() });

  // Fire a streaming request and destroy the client socket after the first chunk.
  await new Promise((resolve) => {
    const req = http.request(
      { host: '127.0.0.1', port: srv.port, method: 'POST', path: '/v1/messages', headers: { 'content-type': 'application/json' } },
      (res) => {
        res.once('data', () => {
          req.destroy();
          resolve();
        });
        res.on('error', () => {});
      }
    );
    req.on('error', () => {}); // ECONNRESET after destroy
    req.end(JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'stream then abort' }] }));
  });

  await new Promise((r) => setTimeout(r, 50)); // let the server process the abort

  // Server must still be alive: a follow-up request completes normally.
  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'follow up after abort' }] }),
  });
  assert.strictEqual(r.status, 200);
  assert.ok(r.body.toString('utf8').includes('message_stop'));

  await srv.close();
  await mock.close();
});

test('non-allowlisted upstream host is rejected by validateConfig', async () => {
  const { validateConfig, mergeConfig } = require('../router');
  const original = process.exit;
  let exited = 0;
  process.exit = () => {
    exited++;
    throw new Error('exit'); // stop execution the way process.exit would
  };
  try {
    assert.throws(() =>
      validateConfig(mergeConfig({ upstream: 'https://evil.example.com', tiers: TIERS }))
    );
    assert.strictEqual(exited, 1);
    // loopback + api.anthropic.com pass
    exited = 0;
    validateConfig(mergeConfig({ upstream: 'http://127.0.0.1:9', tiers: TIERS }));
    assert.strictEqual(exited, 0);
  } finally {
    process.exit = original;
  }
});

test('partial tiers deep-merge over defaults; missing tier is fatal', async () => {
  const { mergeConfig, validateConfig } = require('../router');
  const merged = mergeConfig({ tiers: { heavy: 'my-opus' } });
  assert.strictEqual(merged.tiers.heavy, 'my-opus');
  assert.strictEqual(merged.tiers.light, 'claude-haiku-4-5'); // preserved from defaults
  assert.strictEqual(merged.tiers.standard, 'claude-sonnet-5');

  const original = process.exit;
  let exited = 0;
  process.exit = () => {
    exited++;
    throw new Error('exit');
  };
  try {
    assert.throws(() =>
      validateConfig({ upstream: 'http://127.0.0.1:9', tiers: { light: '', standard: 'x', heavy: 'y' } })
    );
    assert.ok(exited >= 1);
  } finally {
    process.exit = original;
  }
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

// ---------- cross-provider (OpenAI-compatible) routing ----------

// Mock OpenAI-compatible upstream. Records POST /chat/completions requests and
// replies with either a streaming chat.completions SSE (default) or canned JSON.
function startOpenaiMock({ stream = true } = {}) {
  const requests = [];
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      requests.push({ method: req.method, url: req.url, headers: req.headers, body: Buffer.concat(chunks) });
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const send = (o) => res.write(`data: ${JSON.stringify(o)}\n\n`);
        send({ id: 'cmpl-x', choices: [{ delta: { role: 'assistant', content: '' } }] });
        setTimeout(() => {
          send({ choices: [{ delta: { content: 'Hello from ' } }] });
          setTimeout(() => {
            send({ choices: [{ delta: { content: 'the provider' } }] });
            send({ choices: [{ delta: {}, finish_reason: 'stop' }] });
            send({ choices: [], usage: { prompt_tokens: 9, completion_tokens: 3 } });
            res.write('data: [DONE]\n\n');
            res.end();
          }, 60);
        }, 60);
      } else {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            id: 'cmpl-y',
            choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'non-stream reply' } }],
            usage: { prompt_tokens: 4, completion_tokens: 2 },
          })
        );
      }
    });
  });
  return new Promise((r) => {
    server.listen(0, '127.0.0.1', () =>
      r({
        url: `http://127.0.0.1:${server.address().port}`,
        requests,
        close: () => new Promise((rr) => server.close(rr)),
      })
    );
  });
}

function providerConfig(openaiUrl, over = {}) {
  return baseConfig({
    tiers: { light: 'claude-haiku-4-5', standard: 'claude-sonnet-5', heavy: 'mockai,demo-model' },
    providers: {
      anthropic: { type: 'anthropic' },
      mockai: { type: 'openai', base_url: openaiUrl, api_key_env: 'MOCKAI_KEY' },
    },
    ...over,
  });
}

test('openai provider: SSE round-trip yields a valid Anthropic event stream; client creds never cross', async () => {
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'heavy' }); // classifier + (unused) anthropic upstream
  const oai = await startOpenaiMock({ stream: true });
  const srv = await startServer({ config: providerConfig(oai.url), upstream: anth.url, logFile: tmpLog() });

  const clientAuth = 'Bearer sk-ant-CLIENT-TOKEN';
  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json', authorization: clientAuth, 'anthropic-beta': 'oauth-2025-04-20', 'x-api-key': 'client-xkey' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', stream: true, messages: [{ role: 'user', content: 'route me cross-provider' }] }),
  });

  // Client receives a well-formed Anthropic SSE.
  const text = r.body.toString('utf8');
  const events = [...text.matchAll(/event: (\w+)/g)].map((m) => m[1]);
  assert.strictEqual(events[0], 'message_start');
  assert.strictEqual(events[events.length - 1], 'message_stop');
  assert.ok(events.includes('content_block_delta'));
  const startData = JSON.parse(text.match(/event: message_start\ndata: (.*)/)[1]);
  assert.strictEqual(startData.message.model, 'mockai/demo-model');

  // SECURITY: the OpenAI upstream saw ONLY the provider bearer, never client creds.
  assert.strictEqual(oai.requests.length, 1);
  const h = oai.requests[0].headers;
  assert.strictEqual(h['authorization'], 'Bearer provider-secret-key');
  assert.notStrictEqual(h['authorization'], clientAuth);
  assert.strictEqual(h['x-api-key'], undefined);
  assert.strictEqual(h['anthropic-beta'], undefined);
  assert.strictEqual(oai.requests[0].url, '/chat/completions');

  await srv.close();
  await anth.close();
  await oai.close();
  delete process.env.MOCKAI_KEY;
});

test('openai provider: missing credential env fails open to Anthropic passthrough with fallback_reason', async () => {
  delete process.env.MOCKAI_KEY; // key intentionally absent
  const anth = await startMock({ label: 'heavy' });
  const oai = await startOpenaiMock({ stream: true });
  const logFile = tmpLog();
  const srv = await startServer({ config: providerConfig(oai.url), upstream: anth.url, logFile });

  await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'route but key missing' }] }),
  });

  // Fell open to Anthropic: original model reached the anthropic mock; openai mock untouched.
  const fwd = anth.forwarded();
  assert.strictEqual(fwd.length, 1);
  assert.strictEqual(safeModel(fwd[0].body.toString('utf8')), 'claude-3-5-sonnet');
  assert.strictEqual(oai.requests.length, 0);

  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').map(JSON.parse);
  const entry = lines[lines.length - 1];
  assert.strictEqual(entry.decision, 'fallback');
  assert.strictEqual(entry.fallback_reason, 'provider-key-missing:mockai');

  await srv.close();
  await anth.close();
  await oai.close();
});

test('openai provider: non-stream response is translated into an Anthropic message envelope', async () => {
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'heavy' });
  const oai = await startOpenaiMock({ stream: false });
  const srv = await startServer({ config: providerConfig(oai.url), upstream: anth.url, logFile: tmpLog() });

  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'non stream please' }] }),
  });
  assert.strictEqual(r.status, 200);
  const env = JSON.parse(r.body.toString('utf8'));
  assert.strictEqual(env.type, 'message');
  assert.strictEqual(env.role, 'assistant');
  assert.strictEqual(env.model, 'mockai/demo-model');
  assert.deepStrictEqual(env.content, [{ type: 'text', text: 'non-stream reply' }]);
  assert.strictEqual(env.stop_reason, 'end_turn');
  assert.deepStrictEqual(env.usage, { input_tokens: 4, output_tokens: 2 });

  await srv.close();
  await anth.close();
  await oai.close();
  delete process.env.MOCKAI_KEY;
});

test('modality gate: an image request routed to a text-only openai provider fails open to Anthropic', async () => {
  // Key PRESENT on purpose: proves the MODALITY gate diverts, not a missing credential.
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'standard' }); // classifier + Anthropic fallback upstream
  const prov = await startProviderMock(({ res }) => {
    // Must never be reached. If it is, answer so the test fails on the routing assert (not a hang).
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        id: 'cmpl-should-not-happen',
        choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'blind' } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      })
    );
  });
  const logFile = tmpLog();
  const srv = await startServer({
    config: providerConfig(prov.url, {
      tiers: { light: 'claude-haiku-4-5', standard: 'mockai,demo-model', heavy: 'claude-opus-4-8' },
    }),
    upstream: anth.url,
    logFile,
  });

  await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-3-5-sonnet',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } },
            { type: 'text', text: 'what is this?' },
          ],
        },
      ],
    }),
  });

  // Served by ANTHROPIC (fail-open), NOT the text-only provider.
  const fwd = anth.forwarded();
  assert.strictEqual(fwd.length, 1, 'the image request must be served by the Anthropic upstream');
  assert.strictEqual(safeModel(fwd[0].body.toString('utf8')), 'claude-3-5-sonnet'); // ORIGINAL model
  assert.strictEqual(prov.requests.length, 0, 'the text-only provider must never receive a non-text request');

  // Decision log records an Anthropic modality fallback.
  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').map(JSON.parse);
  const last = lines[lines.length - 1];
  assert.strictEqual(last.decision, 'fallback');
  assert.strictEqual(last.provider, 'anthropic');
  assert.ok(
    typeof last.fallback_reason === 'string' && last.fallback_reason.startsWith('modality:'),
    `fallback_reason should start with 'modality:', got ${JSON.stringify(last.fallback_reason)}`
  );

  await srv.close();
  await anth.close();
  await prov.close();
  delete process.env.MOCKAI_KEY;
});

test('privacy gate: a request containing a secret routed to a text provider fails open to Anthropic', async () => {
  // Key PRESENT on purpose: proves the PRIVACY gate diverts, not a missing credential.
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'standard' }); // classifier + Anthropic fallback upstream
  const prov = await startProviderMock(({ res }) => {
    // Must never be reached. If it is, answer so the test fails on the routing assert (not a hang).
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        id: 'cmpl-should-not-happen',
        choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'leaked' } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      })
    );
  });
  const logFile = tmpLog();
  const srv = await startServer({
    config: providerConfig(prov.url, {
      tiers: { light: 'claude-haiku-4-5', standard: 'mockai,demo-model', heavy: 'claude-opus-4-8' },
    }),
    upstream: anth.url,
    logFile,
  });

  // Canonical fake AWS access key id -> hasSensitiveContent matches -> privacy gate fires.
  await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-3-5-sonnet',
      messages: [{ role: 'user', content: 'Is this AWS key still active? AKIAIOSFODNN7EXAMPLE' }],
    }),
  });

  // Served by ANTHROPIC (fail-closed), NOT the text provider that would have seen the secret.
  const fwd = anth.forwarded();
  assert.strictEqual(fwd.length, 1, 'the secret-bearing request must be served by the Anthropic upstream');
  assert.strictEqual(safeModel(fwd[0].body.toString('utf8')), 'claude-3-5-sonnet'); // ORIGINAL model
  assert.strictEqual(prov.requests.length, 0, 'the third-party provider must never receive a secret');

  // Decision log records an Anthropic privacy fallback.
  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').map(JSON.parse);
  const last = lines[lines.length - 1];
  assert.strictEqual(last.decision, 'fallback');
  assert.strictEqual(last.provider, 'anthropic');
  assert.ok(
    typeof last.fallback_reason === 'string' && last.fallback_reason.startsWith('privacy:'),
    `fallback_reason should start with 'privacy:', got ${JSON.stringify(last.fallback_reason)}`
  );

  await srv.close();
  await anth.close();
  await prov.close();
  delete process.env.MOCKAI_KEY;
});

// ---------- hardening: provider timeout + runtime fail-open + non-SSE re-envelope ----------

// Fully controllable OpenAI-compatible provider mock: the handler decides status,
// delay, content-type, or a mid-stream socket cutoff. Records every request seen.
function startProviderMock(handler) {
  const requests = [];
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('error', () => {});
    res.on('error', () => {});
    req.on('end', () => {
      requests.push({ method: req.method, url: req.url, headers: req.headers, body: Buffer.concat(chunks) });
      try {
        handler({ req, res });
      } catch {
        try { res.destroy(); } catch {}
      }
    });
  });
  return new Promise((r) => {
    server.listen(0, '127.0.0.1', () =>
      r({
        url: `http://127.0.0.1:${server.address().port}`,
        requests,
        close: () => new Promise((rr) => server.close(rr)),
      })
    );
  });
}

test('provider HTTP 500 fails open to Anthropic at runtime; provider-http-500 logged', async () => {
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'heavy' }); // classifier + fallback upstream
  const prov = await startProviderMock(({ res }) => {
    res.writeHead(500, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'kaboom' } }));
  });
  const logFile = tmpLog();
  const srv = await startServer({ config: providerConfig(prov.url), upstream: anth.url, logFile });

  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'route then 500' }] }),
  });

  // Client received the Anthropic fallback response, not a provider error.
  assert.strictEqual(r.status, 200);
  const fwd = anth.forwarded();
  assert.strictEqual(fwd.length, 1);
  assert.strictEqual(safeModel(fwd[0].body.toString('utf8')), 'claude-3-5-sonnet'); // ORIGINAL model
  assert.strictEqual(prov.requests.length, 1); // provider attempted exactly once

  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').map(JSON.parse);
  const last = lines[lines.length - 1];
  assert.strictEqual(last.decision, 'fallback');
  assert.strictEqual(last.fallback_reason, 'provider-http-500');
  assert.strictEqual(last.provider, 'anthropic');
  assert.strictEqual(last.routed_model, last.original_model);

  await srv.close();
  await anth.close();
  await prov.close();
  delete process.env.MOCKAI_KEY;
});

test('provider timeout (no response before providerTimeoutMs) fails open to Anthropic', async () => {
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'heavy' });
  const prov = await startProviderMock(({ res }) => {
    // Never respond within the timeout window; unref so the timer can't hold the loop.
    const t = setTimeout(() => {
      try { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{}'); } catch {}
    }, 4000);
    if (t.unref) t.unref();
  });
  const logFile = tmpLog();
  const srv = await startServer({ config: providerConfig(prov.url, { providerTimeoutMs: 150 }), upstream: anth.url, logFile });

  const t0 = Date.now();
  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'slow provider' }] }),
  });
  const elapsed = Date.now() - t0;

  assert.strictEqual(r.status, 200);
  assert.ok(elapsed < 1500, `abort+fallback should be fast, took ${elapsed}ms`);
  const fwd = anth.forwarded();
  assert.strictEqual(fwd.length, 1);
  assert.strictEqual(safeModel(fwd[0].body.toString('utf8')), 'claude-3-5-sonnet');

  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').map(JSON.parse);
  const last = lines[lines.length - 1];
  assert.strictEqual(last.decision, 'fallback');
  assert.match(last.fallback_reason, /provider-fetch|abort/);
  assert.strictEqual(last.provider, 'anthropic');

  await srv.close();
  await anth.close();
  await prov.close();
  delete process.env.MOCKAI_KEY;
});

test('stream requested but provider returns non-SSE JSON is re-enveloped as a single-shot Anthropic stream', async () => {
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'heavy' });
  const prov = await startProviderMock(({ res }) => {
    // Provider ignored stream:true and answered with a single JSON body.
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        id: 'cmpl-ns',
        choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'single shot text' } }],
        usage: { prompt_tokens: 5, completion_tokens: 3 },
      })
    );
  });
  const srv = await startServer({ config: providerConfig(prov.url), upstream: anth.url, logFile: tmpLog() });

  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', stream: true, messages: [{ role: 'user', content: 'stream me' }] }),
  });

  assert.strictEqual(r.status, 200);
  assert.match(r.headers['content-type'], /text\/event-stream/);
  const text = r.body.toString('utf8');
  const events = [...text.matchAll(/event: (\w+)/g)].map((m) => m[1]);
  assert.strictEqual(events[0], 'message_start');
  assert.strictEqual(events[events.length - 1], 'message_stop');
  assert.ok(text.includes('"text_delta"'), 'must contain a text_delta');
  assert.ok(text.includes('single shot text'), 'must carry the provider text');
  assert.strictEqual(prov.requests.length, 1);
  assert.strictEqual(anth.forwarded().length, 0); // provider succeeded: no fallback

  await srv.close();
  await anth.close();
  await prov.close();
  delete process.env.MOCKAI_KEY;
});

test('provider SSE that dies mid-stream still terminates the client stream and does not double-serve', async () => {
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'heavy' });
  const prov = await startProviderMock(({ res }) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write(`data: ${JSON.stringify({ id: 'x', choices: [{ delta: { role: 'assistant', content: 'partial ' } }] })}\n\n`);
    setTimeout(() => { try { res.destroy(); } catch {} }, 50); // kill the socket mid-stream
  });
  const srv = await startServer({ config: providerConfig(prov.url), upstream: anth.url, logFile: tmpLog() });

  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-3-5-sonnet', stream: true, messages: [{ role: 'user', content: 'stream then die' }] }),
  });

  assert.strictEqual(r.status, 200);
  const text = r.body.toString('utf8');
  assert.ok(text.includes('message_stop'), 'stream must be cleanly terminated');
  const stops = [...text.matchAll(/event: message_stop/g)].length;
  assert.strictEqual(stops, 1, 'exactly one termination, not double-served');
  assert.strictEqual(anth.forwarded().length, 0); // no fallback once streaming began

  await srv.close();
  await anth.close();
  await prov.close();
  delete process.env.MOCKAI_KEY;
});

test('non-stream provider that sends headers then stalls the body fails open to Anthropic within providerTimeoutMs', async () => {
  process.env.MOCKAI_KEY = 'provider-secret-key';
  const anth = await startMock({ label: 'heavy' }); // classifier + fallback upstream
  const prov = await startProviderMock(({ res }) => {
    // 200 + headers arrive (so fetch resolves), then a partial body that never
    // completes. The non-stream branch must stay bounded through the body read so
    // this still fails open — no client bytes have been written yet.
    res.writeHead(200, { 'content-type': 'application/json' });
    res.write('{"id":"cmpl-stall",'); // never res.end() — body hangs forever
  });
  const logFile = tmpLog();
  const srv = await startServer({ config: providerConfig(prov.url, { providerTimeoutMs: 150 }), upstream: anth.url, logFile });

  const t0 = Date.now();
  const r = await request(srv.port, {
    headers: { 'content-type': 'application/json' }, // NOT stream:true -> non-stream path
    body: JSON.stringify({ model: 'claude-3-5-sonnet', messages: [{ role: 'user', content: 'stalled body' }] }),
  });
  const elapsed = Date.now() - t0;

  assert.strictEqual(r.status, 200);
  assert.ok(elapsed < 1500, `stalled-body fail-open should be fast, took ${elapsed}ms`);
  const fwd = anth.forwarded();
  assert.strictEqual(fwd.length, 1);
  assert.strictEqual(safeModel(fwd[0].body.toString('utf8')), 'claude-3-5-sonnet'); // ORIGINAL model, failed open
  assert.strictEqual(prov.requests.length, 1); // provider attempted exactly once

  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').map(JSON.parse);
  const last = lines[lines.length - 1];
  assert.strictEqual(last.decision, 'fallback');
  assert.strictEqual(last.provider, 'anthropic');
  assert.strictEqual(last.routed_model, last.original_model);
  assert.match(last.fallback_reason, /provider-body|abort|provider-fetch/);

  await srv.close();
  await anth.close();
  await prov.close();
  delete process.env.MOCKAI_KEY;
});
