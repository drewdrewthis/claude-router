'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { sanitizeForModel, capabilityBlockers } = require('../lib/params');

// Mirrors config.json's MEASURED seed — the only model with capability claims.
const CAPS = { 'claude-haiku-4-5': { effort: false, adaptiveThinking: false } };

test('haiku target strips output_config.effort and drops output_config when it empties', () => {
  const { body, stripped } = sanitizeForModel(
    { model: 'claude-haiku-4-5', output_config: { effort: 'high' }, messages: [] },
    'claude-haiku-4-5',
    CAPS
  );
  assert.deepEqual(stripped, ['output_config.effort']);
  assert.equal('output_config' in body, false); // emptied by removing effort -> removed entirely
});

test('haiku target strips effort but preserves other output_config keys', () => {
  const { body, stripped } = sanitizeForModel(
    { model: 'claude-haiku-4-5', output_config: { effort: 'high', foo: 'bar' } },
    'claude-haiku-4-5',
    CAPS
  );
  assert.deepEqual(stripped, ['output_config.effort']);
  assert.deepEqual(body.output_config, { foo: 'bar' }); // effort gone, foo kept
});

test('haiku target strips adaptive thinking but PRESERVES enabled+budget_tokens', () => {
  const adaptive = sanitizeForModel(
    { model: 'claude-haiku-4-5', thinking: { type: 'adaptive' } },
    'claude-haiku-4-5',
    CAPS
  );
  assert.deepEqual(adaptive.stripped, ['thinking']);
  assert.equal('thinking' in adaptive.body, false);

  const enabled = sanitizeForModel(
    { model: 'claude-haiku-4-5', thinking: { type: 'enabled', budget_tokens: 4096 } },
    'claude-haiku-4-5',
    CAPS
  );
  assert.deepEqual(enabled.stripped, []);
  assert.deepEqual(enabled.body.thinking, { type: 'enabled', budget_tokens: 4096 }); // preserved
});

test('unknown model makes NO claims: body deep-equals input, nothing stripped', () => {
  const input = {
    model: 'claude-sonnet-5',
    output_config: { effort: 'high' },
    thinking: { type: 'adaptive' },
  };
  const { body, stripped } = sanitizeForModel(input, 'claude-sonnet-5', CAPS);
  assert.deepEqual(stripped, []);
  assert.deepEqual(body, input); // untouched
});

test('the caller input body is never mutated', () => {
  const input = {
    model: 'claude-haiku-4-5',
    output_config: { effort: 'high' },
    thinking: { type: 'adaptive' },
  };
  sanitizeForModel(input, 'claude-haiku-4-5', CAPS);
  // the original still carries everything it started with
  assert.deepEqual(input.output_config, { effort: 'high' });
  assert.deepEqual(input.thinking, { type: 'adaptive' });
});

test('effort + adaptive together are both stripped in one pass; the rest is intact', () => {
  const { body, stripped } = sanitizeForModel(
    {
      model: 'claude-haiku-4-5',
      output_config: { effort: 'high' },
      thinking: { type: 'adaptive' },
      messages: [{ role: 'user', content: 'hi' }],
    },
    'claude-haiku-4-5',
    CAPS
  );
  assert.deepEqual([...stripped].sort(), ['output_config.effort', 'thinking']);
  assert.equal('output_config' in body, false);
  assert.equal('thinking' in body, false);
  assert.deepEqual(body.messages, [{ role: 'user', content: 'hi' }]);
});

test('a missing capabilities map is a no-op (strips nothing)', () => {
  const input = { model: 'claude-haiku-4-5', output_config: { effort: 'high' } };
  const { body, stripped } = sanitizeForModel(input, 'claude-haiku-4-5', undefined);
  assert.deepEqual(stripped, []);
  assert.deepEqual(body, input);
});

// ---- dependency cascade: thinking removal must strip thinking-dependent context edits ----

test('haiku target strips thinking AND its dependent clear_thinking context-management edit', () => {
  const { body, stripped } = sanitizeForModel(
    {
      model: 'claude-haiku-4-5',
      thinking: { type: 'adaptive' },
      context_management: { edits: [{ type: 'clear_thinking_20251015', keep: 'all' }] },
    },
    'claude-haiku-4-5',
    CAPS
  );
  assert.equal('thinking' in body, false);
  assert.equal('context_management' in body, false); // its only edit depended on thinking -> gone
  assert.ok(stripped.includes('thinking'));
  assert.ok(stripped.includes('context_management.edits[clear_thinking_20251015]'));
  assert.ok(stripped.includes('context_management'));
});

test('a mixed context_management keeps unrelated edits and drops only the thinking-dependent one', () => {
  const { body, stripped } = sanitizeForModel(
    {
      model: 'claude-haiku-4-5',
      thinking: { type: 'adaptive' },
      context_management: {
        edits: [
          { type: 'clear_thinking_20251015', keep: 'all' },
          { type: 'clear_tool_uses_20250101', keep: 3 },
        ],
      },
    },
    'claude-haiku-4-5',
    CAPS
  );
  assert.deepEqual(body.context_management.edits, [{ type: 'clear_tool_uses_20250101', keep: 3 }]);
  assert.ok(stripped.includes('context_management.edits[clear_thinking_20251015]'));
  assert.equal(stripped.includes('context_management'), false); // survives with the remaining edit
});

test('a target that keeps thinking leaves context_management untouched', () => {
  const CAPS_KEEP = { 'model-x': { effort: false, adaptiveThinking: true } };
  const input = {
    model: 'model-x',
    thinking: { type: 'adaptive' },
    context_management: { edits: [{ type: 'clear_thinking_20251015', keep: 'all' }] },
  };
  const { body, stripped } = sanitizeForModel(input, 'model-x', CAPS_KEEP);
  assert.deepEqual(stripped, []);
  assert.deepEqual(body.context_management, { edits: [{ type: 'clear_thinking_20251015', keep: 'all' }] });

  // Unknown model: no claims, nothing touched (context_management preserved).
  const unknown = sanitizeForModel(input, 'nope', CAPS_KEEP);
  assert.deepEqual(unknown.stripped, []);
  assert.deepEqual(unknown.body, input);
});

test('a clear_thinking edit is dropped for a no-adaptive-thinking target even when the request had no thinking', () => {
  const { body, stripped } = sanitizeForModel(
    {
      model: 'claude-haiku-4-5',
      context_management: { edits: [{ type: 'clear_thinking_20251015', keep: 'all' }] },
      // NO thinking field at all
    },
    'claude-haiku-4-5',
    CAPS
  );
  assert.equal('context_management' in body, false); // invalid without thinking, regardless of who removed it
  assert.ok(stripped.includes('context_management.edits[clear_thinking_20251015]'));
  assert.equal(stripped.includes('thinking'), false); // there was no thinking to strip
});

test('the real Claude Code body (thinking:null, output_config:null, clear_thinking edit) sanitizes to a haiku-valid body', () => {
  const { body, stripped } = sanitizeForModel(
    {
      model: 'claude-haiku-4-5',
      max_tokens: 64000,
      stream: true,
      context_management: { edits: [{ type: 'clear_thinking_20251015', keep: 'all' }] },
      output_config: null,
      thinking: null,
    },
    'claude-haiku-4-5',
    CAPS
  );
  // context_management (the sole 400 cause) is gone; the legal nulls are left as-is, no clamp.
  assert.equal('context_management' in body, false);
  assert.equal(body.output_config, null);
  assert.equal(body.thinking, null);
  assert.equal(body.max_tokens, 64000);
  assert.ok(stripped.includes('context_management'));
});

// ---- capabilityBlockers: content-level pre-flight gate (strip params, never content) ----

const CAPS_SYS = { 'claude-haiku-4-5': { effort: false, adaptiveThinking: false, systemRole: false } };

test('capabilityBlockers flags a role:system message for a no-system-role target', () => {
  const body = { model: 'claude-haiku-4-5', messages: [{ role: 'user', content: 'hi' }, { role: 'system', content: 'reminder' }] };
  assert.deepEqual(capabilityBlockers(body, 'claude-haiku-4-5', CAPS_SYS), ['system-role-message']);
});

test('capabilityBlockers returns [] when there is no role:system message', () => {
  const body = { model: 'claude-haiku-4-5', messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'yo' }] };
  assert.deepEqual(capabilityBlockers(body, 'claude-haiku-4-5', CAPS_SYS), []);
});

test('capabilityBlockers makes NO claims for an unknown model (even with a role:system message)', () => {
  const body = { model: 'x', messages: [{ role: 'system', content: 'reminder' }] };
  assert.deepEqual(capabilityBlockers(body, 'x', CAPS_SYS), []);
});

test('capabilityBlockers returns [] when the target supports the system role', () => {
  const caps = { 'model-x': { systemRole: true } };
  const body = { model: 'model-x', messages: [{ role: 'system', content: 'reminder' }] };
  assert.deepEqual(capabilityBlockers(body, 'model-x', caps), []);
});
