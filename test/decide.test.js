'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { decide } = require('../lib/decide');

const TIERS = { light: 'claude-haiku-4-5', standard: 'claude-sonnet-5', heavy: 'claude-opus-4-8' };

function baseConfig(over = {}) {
  return {
    mode: 'session',
    tiers: TIERS,
    pinned: [],
    classifierModel: 'claude-haiku-4-5',
    classifierTimeoutMs: 3000,
    maxHaikuInputTokens: 150000,
    upstream: 'http://mock',
    ...over,
  };
}

function bodyBuf(obj) {
  return Buffer.from(JSON.stringify(obj));
}

function fixedClock() {
  return () => 1000;
}

test('label -> configured tier mapping (read from config, not hardcoded)', async () => {
  const edited = baseConfig({ tiers: { light: 'L', standard: 'S', heavy: 'H' } });
  for (const [label, model] of [['light', 'L'], ['standard', 'S'], ['heavy', 'H']]) {
    const cache = new Map();
    const d = await decide({
      rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'hi' }] }),
      config: edited,
      classify: async () => label,
      now: fixedClock(),
      cache,
    });
    assert.strictEqual(d.routedModel, model, `${label} -> ${model}`);
    assert.strictEqual(d.log.decision, 'routed');
    assert.strictEqual(d.log.label, label);
  }
});

test('pinned model -> passthrough, classify never called', async () => {
  let called = false;
  const d = await decide({
    rawBody: bodyBuf({ model: 'claude-opus-4-8', messages: [{ role: 'user', content: 'hi' }] }),
    config: baseConfig({ pinned: ['claude-opus'] }),
    classify: async () => {
      called = true;
      return 'heavy';
    },
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(called, false);
  assert.strictEqual(d.routedModel, null);
  assert.strictEqual(d.log.decision, 'pinned');
});

test('already-haiku -> passthrough, no classify', async () => {
  let called = false;
  const d = await decide({
    rawBody: bodyBuf({ model: 'claude-haiku-4-5', messages: [{ role: 'user', content: 'hi' }] }),
    config: baseConfig(),
    classify: async () => {
      called = true;
      return 'heavy';
    },
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(called, false);
  assert.strictEqual(d.log.decision, 'already-light');
  assert.strictEqual(d.routedModel, null);
});

test('context guard: light label + oversized body -> standard tier, never haiku', async () => {
  const d = await decide({
    rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'x' }] }),
    config: baseConfig({ maxHaikuInputTokens: 1 }), // any real body exceeds this
    classify: async () => 'light',
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(d.routedModel, TIERS.standard);
  assert.notStrictEqual(d.routedModel, TIERS.light);
});

test('session stickiness: same key -> one classify, second is cache-hit', async () => {
  let calls = 0;
  const cache = new Map();
  const req = () => ({
    rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'same' }] }),
    config: baseConfig(),
    classify: async () => {
      calls++;
      return 'heavy';
    },
    now: fixedClock(),
    cache,
  });
  const a = await decide(req());
  const b = await decide(req());
  assert.strictEqual(calls, 1);
  assert.strictEqual(a.log.decision, 'routed');
  assert.strictEqual(b.log.decision, 'cache-hit');
  assert.strictEqual(b.log.cache_hit, true);
  assert.strictEqual(a.routedModel, b.routedModel);
});

test('midflight: messages.length > 2 with unseen key -> midflight-passthrough, no classify', async () => {
  let called = false;
  const d = await decide({
    rawBody: bodyBuf({
      model: 'claude-sonnet-4',
      messages: [
        { role: 'user', content: 'a' },
        { role: 'assistant', content: 'b' },
        { role: 'user', content: 'c' },
      ],
    }),
    config: baseConfig(),
    classify: async () => {
      called = true;
      return 'heavy';
    },
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(called, false);
  assert.strictEqual(d.log.decision, 'midflight-passthrough');
  assert.strictEqual(d.routedModel, null);
});

test('malformed JSON -> malformed-passthrough', async () => {
  const d = await decide({
    rawBody: Buffer.from('this is not json'),
    config: baseConfig(),
    classify: async () => 'heavy',
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(d.log.decision, 'malformed-passthrough');
  assert.strictEqual(d.routedModel, null);
});

test('missing model field -> malformed-passthrough', async () => {
  const d = await decide({
    rawBody: bodyBuf({ messages: [{ role: 'user', content: 'hi' }] }),
    config: baseConfig(),
    classify: async () => 'heavy',
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(d.log.decision, 'malformed-passthrough');
});

test('classify throws -> fallback, original model preserved', async () => {
  const d = await decide({
    rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'hi' }] }),
    config: baseConfig(),
    classify: async () => {
      throw new Error('boom');
    },
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(d.routedModel, null);
  assert.strictEqual(d.log.decision, 'fallback');
  assert.match(d.log.fallback_reason, /boom/);
  assert.strictEqual(d.log.routed_model, 'claude-sonnet-4');
});

test('unexpected label -> fallback', async () => {
  const d = await decide({
    rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'hi' }] }),
    config: baseConfig(),
    classify: async () => 'banana',
    now: fixedClock(),
    cache: new Map(),
  });
  assert.strictEqual(d.log.decision, 'fallback');
});

test('provider-prefixed tier resolves to {provider, model}; bare stays anthropic', async () => {
  const cfg = baseConfig({
    tiers: {
      light: 'claude-haiku-4-5',
      standard: 'claude-sonnet-5',
      heavy: 'nvidia,meta/llama-3.3-70b-instruct',
    },
  });
  const heavy = await decide({
    rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'complex' }] }),
    config: cfg,
    classify: async () => 'heavy',
    now: fixedClock(),
    cache: new Map(),
  });
  assert.deepStrictEqual(heavy.routed, { provider: 'nvidia', model: 'meta/llama-3.3-70b-instruct' });
  assert.strictEqual(heavy.routedModel, 'meta/llama-3.3-70b-instruct');
  assert.strictEqual(heavy.log.provider, 'nvidia');

  const light = await decide({
    rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'trivial' }] }),
    config: cfg,
    classify: async () => 'light',
    now: fixedClock(),
    cache: new Map(),
  });
  assert.deepStrictEqual(light.routed, { provider: 'anthropic', model: 'claude-haiku-4-5' });
  assert.strictEqual(light.log.provider, 'anthropic');
});

test('request mode classifies every request (no cache)', async () => {
  let calls = 0;
  const cache = new Map();
  const req = () => ({
    rawBody: bodyBuf({ model: 'claude-sonnet-4', messages: [{ role: 'user', content: 'same' }] }),
    config: baseConfig({ mode: 'request' }),
    classify: async () => {
      calls++;
      return 'heavy';
    },
    now: fixedClock(),
    cache,
  });
  await decide(req());
  await decide(req());
  assert.strictEqual(calls, 2);
  assert.strictEqual(cache.size, 0);
});
