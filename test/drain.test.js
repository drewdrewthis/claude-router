'use strict';
// The SIGINT/SIGTERM drain in router.js's `require.main === module` block is the PR's own
// motivating fix (a killed router must not silently lose the LAST turn's queued trace). It only
// runs as `main`, so require()-based tests can't reach it — this exercises it via a real child
// process against a mock upstream and a mock LangWatch collector.

const test = require('node:test');
const assert = require('node:assert');
const http = require('node:http');
const path = require('node:path');
const { spawn } = require('node:child_process');

const ROUTER = path.join(__dirname, '..', 'router.js');

function listen(server) {
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r(server.address().port)));
}
function post(port, pathName, bodyObj) {
  return new Promise((resolve, reject) => {
    const body = Buffer.from(JSON.stringify(bodyObj));
    const req = http.request(
      { host: '127.0.0.1', port, method: 'POST', path: pathName,
        headers: { 'content-type': 'application/json', 'content-length': body.length } },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks) }));
      }
    );
    req.on('error', reject);
    req.end(body);
  });
}

test('SIGTERM drains the queued telemetry span before exit (the last-turn trace is not lost)', async (t) => {
  const traces = [];

  // Mock Anthropic upstream: answer any POST /v1/messages with a minimal message body.
  const upstream = http.createServer((req, res) => {
    req.on('error', () => {});
    req.resume();
    req.on('end', () => {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        id: 'msg_drain', type: 'message', role: 'assistant', model: 'claude-haiku-4-5',
        content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn',
        usage: { input_tokens: 1, output_tokens: 1 },
      }));
    });
  });

  // Mock LangWatch collector: record every POST /v1/traces body, answer the OTLP 200 shape.
  const collector = http.createServer((req, res) => {
    const chunks = [];
    req.on('error', () => {});
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      if (req.method === 'POST' && req.url === '/v1/traces') traces.push(Buffer.concat(chunks).toString('utf8'));
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ message: 'Trace received successfully.', partialSuccess: { rejectedSpans: 0 } }));
    });
  });

  const upstreamPort = await listen(upstream);
  const collectorPort = await listen(collector);

  const childEnv = {
    ...process.env,
    ROUTER_UPSTREAM: `http://127.0.0.1:${upstreamPort}`,
    LANGWATCH_ENDPOINT: `http://127.0.0.1:${collectorPort}`,
    LANGWATCH_API_KEY: 'lw-test-fake',
    ROUTER_TELEMETRY_CAPTURE_CONTENT: '0',
    ROUTER_PORT: '0', // ephemeral; parse the actual port from the "listening" line on stderr
    ROUTER_ANTHROPIC_API_KEY: '', // no boot self-check against a real credential
  };
  delete childEnv.OTEL_EXPORTER_OTLP_HEADERS;
  delete childEnv.OTEL_EXPORTER_OTLP_ENDPOINT;
  delete childEnv.ROUTER_TELEMETRY;

  let child;
  try {
    child = spawn(process.execPath, [ROUTER], { env: childEnv, stdio: ['ignore', 'ignore', 'pipe'] });
    child.on('error', () => {});

    // Wait (bounded) for the router to announce its bound port on stderr.
    const routerPort = await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('router did not announce a port within 8s')), 8000);
      let buf = '';
      child.stderr.on('data', (d) => {
        buf += d.toString('utf8');
        const m = buf.match(/listening on http:\/\/127\.0\.0\.1:(\d+)/);
        if (m) { clearTimeout(t); resolve(Number(m[1])); }
      });
      child.on('exit', (code) => { clearTimeout(t); reject(new Error('router exited early (code ' + code + ')')); });
    });

    // One already-light passthrough turn: forwards to the mock upstream AND records a span.
    const r = await post(routerPort, '/v1/messages', {
      model: 'claude-haiku-4-5',
      messages: [{ role: 'user', content: 'hi' }],
    });
    assert.strictEqual(r.status, 200);

    // Kill and time the drain. The queued span (flushIntervalMs=2000, not yet flushed) must be
    // flushed by the drain's telemetry.shutdown() before the process exits.
    const killedAt = Date.now();
    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('router did not exit within 8s of SIGTERM')), 8000);
      child.on('exit', () => { clearTimeout(t); resolve(); });
      child.kill('SIGTERM');
    });
    const drainMs = Date.now() - killedAt;
    t.diagnostic(`drain exit in ${drainMs}ms; collector received ${traces.length} trace POST(s)`);

    assert.ok(drainMs < 8000, `drain took ${drainMs}ms (must be under the 8s hard cap)`);
    assert.ok(traces.length >= 1, `mock collector received ${traces.length} trace POST(s); expected >= 1 (span lost)`);
  } finally {
    if (child && child.exitCode == null && child.signalCode == null) child.kill('SIGKILL');
    await new Promise((r) => upstream.close(r));
    await new Promise((r) => collector.close(r));
  }
});
