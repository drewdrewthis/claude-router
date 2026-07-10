'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { createTelemetry, resolveTelemetryOptions } = require('../lib/telemetry');

const ENDPOINT = 'https://telemetry.test/v1/traces';
const KEY = 'lw-test-key'; // fake, never a real credential

// A fetchImpl that records every call and returns a JSON 200. Inject it so tests never
// touch the network.
function captureFetch(responseBody) {
  const calls = [];
  const impl = async (url, opts) => {
    calls.push({ url, headers: opts.headers, body: JSON.parse(opts.body) });
    return {
      ok: true,
      status: 200,
      json: async () =>
        responseBody || { message: 'Trace received successfully.', partialSuccess: { rejectedSpans: 0 } },
    };
  };
  impl.calls = calls;
  return impl;
}

const DIGEST = 'refactor the auth module\n[meta est_tokens=6 tools=0 thinking=false images=0]';

function sampleDecision(over = {}) {
  const base = {
    routed: { provider: 'nvidia', model: 'meta/llama-3.3-70b-instruct' },
    routedModel: 'meta/llama-3.3-70b-instruct',
    rewrite: false,
    hasNonText: false,
    hasSensitive: false,
    digest: DIGEST,
    log: {
      key: 'abc123def456',
      original_model: 'claude-sonnet-4',
      routed_model: 'meta/llama-3.3-70b-instruct',
      provider: 'nvidia',
      label: 'heavy',
      decision: 'routed',
      est_input_tokens: 6,
      cache_hit: false,
      classifier_ms: 21,
    },
  };
  return { ...base, ...over, log: { ...base.log, ...(over.log || {}) } };
}

function sampleEntry(over = {}) {
  const decision = over.decision || sampleDecision();
  return {
    decision,
    classifierMs: decision.log.classifier_ms,
    digest: decision.digest,
    label: decision.log.label,
    classifierModel: 'claude-haiku-4-5',
    upstream: {
      provider: 'nvidia',
      model: decision.log.routed_model,
      status: 200,
      ms: 42,
      usage: { input_tokens: 6, output_tokens: 18 },
    },
    startMs: 1000,
    endMs: 1200,
    httpStatus: 200,
    ...over,
    decision,
  };
}

// Record one entry through an active client, force a flush, return the captured payload.
async function exportOne(entry, opts = {}) {
  const fetchImpl = captureFetch(opts.responseBody);
  const t = createTelemetry({
    enabled: true,
    endpoint: ENDPOINT,
    apiKey: KEY,
    fetchImpl,
    flushIntervalMs: 1e9, // never auto-flush mid-test
    ...opts,
  });
  t.recordRequest(entry);
  await t.shutdown();
  return { fetchImpl, telemetry: t };
}

function spansOf(payload) {
  return payload.resourceSpans[0].scopeSpans[0].spans;
}
function findSpan(spans, name) {
  return spans.find((s) => s.name === name);
}
function attrMap(span) {
  const m = {};
  for (const a of span.attributes) m[a.key] = a.value;
  return m;
}
function hasAttrAnywhere(spans, key) {
  return spans.some((s) => s.attributes.some((a) => a.key === key));
}

// ---------------- resolveTelemetryOptions ----------------

test('resolveTelemetryOptions: picks up LANGWATCH_API_KEY and enables', () => {
  const o = resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY });
  assert.equal(o.apiKey, KEY);
  assert.equal(o.enabled, true);
  assert.equal(o.endpoint, 'https://app.langwatch.ai/api/otel/v1/traces');
});

test('resolveTelemetryOptions: falls back to parsing OTEL_EXPORTER_OTLP_HEADERS', () => {
  const o = resolveTelemetryOptions(
    {},
    { OTEL_EXPORTER_OTLP_HEADERS: 'Authorization=Bearer ' + KEY + ',x-extra=1' }
  );
  assert.equal(o.apiKey, KEY);
  assert.equal(o.enabled, true);
});

test('resolveTelemetryOptions: disabled when no key resolves', () => {
  const o = resolveTelemetryOptions({}, {});
  assert.equal(o.apiKey, '');
  assert.equal(o.enabled, false);
});

test('resolveTelemetryOptions: disabled by ROUTER_TELEMETRY=0 even with a key', () => {
  const o = resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY, ROUTER_TELEMETRY: '0' });
  assert.equal(o.enabled, false);
});

test('resolveTelemetryOptions: disabled by config.telemetry.enabled === false', () => {
  const o = resolveTelemetryOptions({ telemetry: { enabled: false } }, { LANGWATCH_API_KEY: KEY });
  assert.equal(o.enabled, false);
});

test('resolveTelemetryOptions: endpoint /v1/traces appended exactly once', () => {
  // default base -> appended
  assert.equal(
    resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY }).endpoint,
    'https://app.langwatch.ai/api/otel/v1/traces'
  );
  // trailing slash stripped before appending
  assert.equal(
    resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY, OTEL_EXPORTER_OTLP_ENDPOINT: 'https://x.io/api/otel/' }).endpoint,
    'https://x.io/api/otel/v1/traces'
  );
  // already a /v1/traces URL -> NOT double-appended
  assert.equal(
    resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY, OTEL_EXPORTER_OTLP_ENDPOINT: 'https://x.io/api/otel/v1/traces' }).endpoint,
    'https://x.io/api/otel/v1/traces'
  );
  // LANGWATCH_ENDPOINT wins over OTEL_EXPORTER_OTLP_ENDPOINT
  assert.equal(
    resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY, LANGWATCH_ENDPOINT: 'https://lw.test/otel', OTEL_EXPORTER_OTLP_ENDPOINT: 'https://ignored/otel' }).endpoint,
    'https://lw.test/otel/v1/traces'
  );
});

test('resolveTelemetryOptions: captureContent from config or env', () => {
  assert.equal(resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY }).captureContent, false);
  assert.equal(
    resolveTelemetryOptions({ telemetry: { captureContent: true } }, { LANGWATCH_API_KEY: KEY }).captureContent,
    true
  );
  assert.equal(
    resolveTelemetryOptions({}, { LANGWATCH_API_KEY: KEY, ROUTER_TELEMETRY_CAPTURE_CONTENT: '1' }).captureContent,
    true
  );
});

// ---------------- payload shape ----------------

test('payload: every span has a non-empty attributes array, hex ids, and children parent the root', async () => {
  const { fetchImpl } = await exportOne(sampleEntry());
  assert.equal(fetchImpl.calls.length, 1);
  const call = fetchImpl.calls[0];
  assert.equal(call.url, ENDPOINT);
  assert.equal(call.headers.authorization, 'Bearer ' + KEY);
  assert.equal(call.headers['content-type'], 'application/json');

  const payload = call.body;
  const resAttrs = payload.resourceSpans[0].resource.attributes;
  assert.ok(resAttrs.some((a) => a.key === 'service.name' && a.value.stringValue === 'claude-router'));
  assert.ok(resAttrs.some((a) => a.key === 'host.name'));

  const spans = spansOf(payload);
  assert.equal(spans.length, 3); // root + classifier + upstream

  for (const s of spans) {
    assert.match(s.traceId, /^[0-9a-f]{32}$/);
    assert.match(s.spanId, /^[0-9a-f]{16}$/);
    assert.ok(Array.isArray(s.attributes) && s.attributes.length > 0, s.name + ' has non-empty attributes');
    // proto3 JSON: int64 timestamps are strings
    assert.equal(typeof s.startTimeUnixNano, 'string');
    assert.equal(typeof s.endTimeUnixNano, 'string');
  }

  const root = findSpan(spans, 'claude_router.request');
  const classifier = findSpan(spans, 'claude_router.classifier');
  const upstream = findSpan(spans, 'claude_router.upstream');

  assert.equal(root.kind, 2); // SERVER
  assert.equal(root.parentSpanId, undefined); // root has no parent
  assert.equal(root.startTimeUnixNano, '1000000000'); // 1000ms * 1e6, via BigInt

  for (const child of [classifier, upstream]) {
    assert.equal(child.kind, 3); // CLIENT
    assert.equal(child.traceId, root.traceId); // shared trace
    assert.equal(child.parentSpanId, root.spanId); // parented to root
  }

  // proto3 JSON: int64 attribute values are strings; bools are boolValue
  const rm = attrMap(root);
  assert.equal(rm['router.est_input_tokens'].intValue, '6');
  assert.equal(rm['http.response.status_code'].intValue, '200');
  assert.equal(rm['router.rewrite'].boolValue, false);
  assert.equal(rm['langwatch.thread_id'].stringValue, 'abc123def456');
  assert.equal(rm['router.decision'].stringValue, 'routed');
  assert.equal(rm['langwatch.span.type'].stringValue, 'span');

  const cm = attrMap(classifier);
  assert.equal(cm['langwatch.span.type'].stringValue, 'llm');
  assert.equal(cm['gen_ai.request.model'].stringValue, 'claude-haiku-4-5');
  assert.equal(cm['gen_ai.system'].stringValue, 'anthropic');

  const um = attrMap(upstream);
  assert.equal(um['gen_ai.request.model'].stringValue, 'meta/llama-3.3-70b-instruct');
  assert.equal(um['gen_ai.system'].stringValue, 'nvidia');
  assert.equal(um['gen_ai.usage.input_tokens'].intValue, '6');
  assert.equal(um['gen_ai.usage.output_tokens'].intValue, '18');
});

test('payload: omitted attributes are absent (never emitted as "null" strings)', async () => {
  // A malformed-passthrough-style decision: null key, null label, no classifier, no upstream.
  const decision = {
    routed: null,
    routedModel: null,
    rewrite: false,
    hasNonText: false,
    hasSensitive: false,
    log: {
      key: null,
      original_model: null,
      routed_model: null,
      provider: 'anthropic',
      label: null,
      decision: 'malformed-passthrough',
      est_input_tokens: 3,
      cache_hit: false,
    },
  };
  const { fetchImpl } = await exportOne(
    sampleEntry({ decision, classifierMs: null, upstream: null, label: null })
  );
  const spans = spansOf(fetchImpl.calls[0].body);
  const rm = attrMap(findSpan(spans, 'claude_router.request'));
  assert.equal('langwatch.thread_id' in rm, false); // null key omitted
  assert.equal('router.label' in rm, false); // null label omitted
  // No attribute value anywhere is the literal string "null".
  for (const s of spans) {
    for (const a of s.attributes) {
      assert.notEqual(a.value.stringValue, 'null');
    }
  }
});

// ---------------- content-privacy gate ----------------

test('captureContent:false -> no langwatch.input/output anywhere and content_captured=false', async () => {
  const { fetchImpl } = await exportOne(sampleEntry(), { captureContent: false });
  const spans = spansOf(fetchImpl.calls[0].body);
  assert.equal(hasAttrAnywhere(spans, 'langwatch.input'), false);
  assert.equal(hasAttrAnywhere(spans, 'langwatch.output'), false);
  const rm = attrMap(findSpan(spans, 'claude_router.request'));
  assert.equal(rm['router.content_captured'].boolValue, false);
});

test('captureContent:true + hasSensitive===true -> privacy gate wins, still NO content, content_captured=false', async () => {
  const decision = sampleDecision({ hasSensitive: true });
  const { fetchImpl } = await exportOne(sampleEntry({ decision }), { captureContent: true });
  const spans = spansOf(fetchImpl.calls[0].body);
  assert.equal(hasAttrAnywhere(spans, 'langwatch.input'), false);
  assert.equal(hasAttrAnywhere(spans, 'langwatch.output'), false);
  const rm = attrMap(findSpan(spans, 'claude_router.request'));
  assert.equal(rm['router.content_captured'].boolValue, false);
});

test('captureContent:true + hasSensitive:false -> classifier span carries langwatch.input/output', async () => {
  const { fetchImpl } = await exportOne(sampleEntry(), { captureContent: true });
  const spans = spansOf(fetchImpl.calls[0].body);
  const cm = attrMap(findSpan(spans, 'claude_router.classifier'));
  assert.ok(cm['langwatch.input'], 'classifier has langwatch.input');
  assert.ok(cm['langwatch.output'], 'classifier has langwatch.output');
  assert.deepEqual(JSON.parse(cm['langwatch.input'].stringValue), { type: 'text', value: DIGEST });
  assert.deepEqual(JSON.parse(cm['langwatch.output'].stringValue), { type: 'text', value: 'heavy' });
  // the ROOT span never carries prompt text
  const rm = attrMap(findSpan(spans, 'claude_router.request'));
  assert.equal('langwatch.input' in rm, false);
  assert.equal(rm['router.content_captured'].boolValue, true);
});

// ---------------- conditional child spans ----------------

test('classifier span omitted when classifierMs is null', async () => {
  const { fetchImpl } = await exportOne(sampleEntry({ classifierMs: null }));
  const spans = spansOf(fetchImpl.calls[0].body);
  assert.equal(findSpan(spans, 'claude_router.classifier'), undefined);
  assert.ok(findSpan(spans, 'claude_router.request'));
  assert.ok(findSpan(spans, 'claude_router.upstream'));
});

test('upstream span omitted when upstream is null', async () => {
  const { fetchImpl } = await exportOne(sampleEntry({ upstream: null }));
  const spans = spansOf(fetchImpl.calls[0].body);
  assert.equal(findSpan(spans, 'claude_router.upstream'), undefined);
  assert.ok(findSpan(spans, 'claude_router.request'));
  assert.ok(findSpan(spans, 'claude_router.classifier'));
});

// ---------------- robustness / lifecycle ----------------

test('a rejecting fetchImpl never throws out of recordRequest/shutdown and increments failed', async () => {
  const t = createTelemetry({
    enabled: true,
    endpoint: ENDPOINT,
    apiKey: KEY,
    flushIntervalMs: 1e9,
    fetchImpl: async () => {
      throw new Error('network down');
    },
  });
  const rootOnly = sampleEntry({ classifierMs: null, upstream: null }); // 1 span
  assert.doesNotThrow(() => t.recordRequest(rootOnly));
  await assert.doesNotReject(() => t.shutdown());
  assert.equal(t.stats().failed, 1);
});

test('queue cap drops the oldest spans and increments dropped', async () => {
  const fetchImpl = captureFetch();
  const t = createTelemetry({
    enabled: true,
    endpoint: ENDPOINT,
    apiKey: KEY,
    flushIntervalMs: 1e9,
    maxQueue: 3,
    fetchImpl,
  });
  for (let i = 0; i < 5; i++) t.recordRequest(sampleEntry({ classifierMs: null, upstream: null })); // 1 span each
  assert.equal(t.stats().dropped, 2);
  assert.equal(t.stats().queued, 3);
  assert.equal(fetchImpl.calls.length, 0); // never reached the 32-span flush threshold
  await t.shutdown();
});

test('partialSuccess.rejectedSpans is surfaced in stats().rejected', async () => {
  const { fetchImpl, telemetry } = await exportOne(sampleEntry(), {
    responseBody: { message: 'ok', partialSuccess: { rejectedSpans: 2 } },
  });
  assert.equal(fetchImpl.calls.length, 1);
  assert.equal(telemetry.stats().sent, 3); // 3 spans posted
  assert.equal(telemetry.stats().rejected, 2);
});

test('a disabled client performs zero fetchImpl calls', async () => {
  const fetchImpl = captureFetch();
  const t = createTelemetry({ enabled: false, endpoint: ENDPOINT, apiKey: KEY, fetchImpl });
  t.recordRequest(sampleEntry());
  await t.shutdown();
  assert.equal(fetchImpl.calls.length, 0);
  assert.deepEqual(t.stats(), { queued: 0, sent: 0, failed: 0, dropped: 0, rejected: 0 });
});

test('a client with no apiKey is inert (no fetchImpl calls)', async () => {
  const fetchImpl = captureFetch();
  const t = createTelemetry({ enabled: true, endpoint: ENDPOINT, apiKey: '', fetchImpl });
  t.recordRequest(sampleEntry());
  await t.shutdown();
  assert.equal(fetchImpl.calls.length, 0);
});

test('shutdown is safe to call twice', async () => {
  const fetchImpl = captureFetch();
  const t = createTelemetry({ enabled: true, endpoint: ENDPOINT, apiKey: KEY, flushIntervalMs: 1e9, fetchImpl });
  t.recordRequest(sampleEntry());
  await t.shutdown();
  await assert.doesNotReject(() => t.shutdown());
  assert.equal(fetchImpl.calls.length, 1); // second shutdown does not re-flush
});

test('shutdown() drains a sub-threshold queue the interval has not yet flushed', async () => {
  const fetchImpl = captureFetch();
  const t = createTelemetry({
    enabled: true,
    endpoint: ENDPOINT,
    apiKey: KEY,
    flushIntervalMs: 60000, // long enough that the interval never fires during the test
    fetchImpl,
  });
  t.recordRequest(sampleEntry({ classifierMs: null, upstream: null })); // 1 span, far under the 32-span threshold
  assert.equal(fetchImpl.calls.length, 0); // proves neither the threshold nor the interval flushed
  await t.shutdown();
  assert.equal(fetchImpl.calls.length, 1); // the drain flushed the queued span
  const spans = spansOf(fetchImpl.calls[0].body);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].name, 'claude_router.request');
  await t.shutdown(); // memoized promise: no second POST
  assert.equal(fetchImpl.calls.length, 1);
});
