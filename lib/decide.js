'use strict';
// Pure decision pipeline. No network, no fs: the classifier fn and clock are
// injected so this is fully unit-testable. Every path returns a decision object
// of the shape:
//   { routedModel: string|null, rewrite: bool, log: {...} }
// routedModel === null means "forward the original body untouched".

const crypto = require('node:crypto');

const TTL_MS = 8 * 60 * 60 * 1000; // 8h
const MAX_ENTRIES = 5000;
const LABELS = ['light', 'standard', 'heavy'];

function sha256(s) {
  return crypto.createHash('sha256').update(s).digest('hex');
}

function extractSystemText(system) {
  if (typeof system === 'string') return system;
  if (Array.isArray(system)) {
    return system.map((b) => (b && typeof b.text === 'string' ? b.text : '')).join('');
  }
  return '';
}

function extractText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((b) => (b && typeof b.text === 'string' ? b.text : '')).join(' ');
  }
  return '';
}

// Content-block types a text-only model cannot consume. A denylist (not an allowlist)
// so text / tool_use / thinking blocks never trip the modality gate; document/audio are
// pre-listed so they gate the instant a client sends them, before any provider-side
// vision flag exists.
const NON_TEXT_BLOCK_TYPES = new Set(['image', 'document', 'audio']);

// True if a single content block is non-text: a top-level image/document/audio, or a
// tool_result whose own content array embeds one (just as unroutable to a text model).
function isNonTextBlock(b) {
  if (!b || typeof b !== 'object') return false;
  if (NON_TEXT_BLOCK_TYPES.has(b.type)) return true;
  if (b.type === 'tool_result' && Array.isArray(b.content)) {
    return b.content.some((c) => c && typeof c === 'object' && NON_TEXT_BLOCK_TYPES.has(c.type));
  }
  return false;
}

// True if ANY message carries content a text-only provider cannot consume (an image now;
// document/audio pre-listed; or a tool_result embedding one). String content and pure
// text/tool_use blocks are text -> false. Defensive: a missing/non-array messages or
// content field collapses to false rather than throwing.
function hasNonTextContent(body) {
  const messages = body && Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!m || !Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (isNonTextBlock(b)) return true;
    }
  }
  return false;
}

// High-confidence credential shapes. A match forces the request onto the Anthropic
// (fail-closed) path so a secret never reaches a third-party provider (see router.js
// planForward). These are deliberately narrow, high-precision patterns (real key / token
// / SSN shapes); semantic PII (names, addresses, free-form secrets) is a future extension.
const SENSITIVE_PATTERNS = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/, // PEM private key block
  /\bAKIA[0-9A-Z]{16}\b/, // AWS access key id
  /\bsk_live_[0-9a-zA-Z]{16,}\b/, // Stripe live secret key
  /\bsk-ant-[0-9a-zA-Z_\-]{20,}\b/, // Anthropic API key
  /\bsk-[0-9a-zA-Z]{32,}\b/, // OpenAI-style secret key
  /\bgh[pousr]_[0-9a-zA-Z]{36,}\b/, // GitHub token
  /\bAIza[0-9A-Za-z_\-]{35}\b/, // Google API key
  /\bxox[baprs]-[0-9a-zA-Z-]{10,}\b/, // Slack token
  /\beyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\b/, // JWT
  /\b\d{3}-\d{2}-\d{4}\b/, // US SSN
];

// True if ANY request text matches a high-confidence credential shape: every message's
// text (string content, text blocks, and text nested in tool_result blocks) plus the
// system prompt. Fail-closed gate input: a match diverts the request to Anthropic in
// planForward. Defensive: missing/non-array fields collapse to false rather than throwing.
function hasSensitiveContent(body) {
  const messages = body && Array.isArray(body.messages) ? body.messages : [];
  let text = extractSystemText(body && body.system);
  for (const m of messages) {
    if (!m) continue;
    text += ' ' + extractText(m.content);
    // extractText does not descend into tool_result content; pull its text too.
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b && b.type === 'tool_result') {
          if (typeof b.content === 'string') text += ' ' + b.content;
          else if (Array.isArray(b.content)) text += ' ' + extractText(b.content);
        }
      }
    }
  }
  return SENSITIVE_PATTERNS.some((re) => re.test(text));
}

function buildDigest(body, est) {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let text = '';
  let images = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i] && messages[i].role === 'user') {
      const content = messages[i].content;
      text = extractText(content);
      // Count image blocks in the SAME (last user) message from the full content, BEFORE
      // the text is sliced — so tier selection stays modality-aware even when the visible
      // text is truncated or absent (an image-only turn).
      if (Array.isArray(content)) {
        images = content.reduce((n, b) => n + (b && b.type === 'image' ? 1 : 0), 0);
      }
      break;
    }
  }
  const tools = Array.isArray(body.tools) ? body.tools.length : 0;
  const thinking = body.thinking ? true : false;
  return `${text.slice(0, 2000)}\n[meta est_tokens=${est} tools=${tools} thinking=${thinking} images=${images}]`;
}

function getCache(cache, key, nowMs) {
  const e = cache.get(key);
  if (!e) return null;
  if (nowMs - e.ts > TTL_MS) {
    cache.delete(key);
    return null;
  }
  return e;
}

function setCache(cache, key, val) {
  if (cache.size >= MAX_ENTRIES && !cache.has(key)) {
    cache.delete(cache.keys().next().value); // evict oldest
  }
  cache.set(key, val);
}

function passthrough(decision, { key, model, est, hasNonText, hasSensitive }) {
  return {
    routed: null,
    routedModel: null,
    rewrite: false,
    hasNonText: !!hasNonText,
    hasSensitive: !!hasSensitive,
    log: {
      key: key || null,
      original_model: model || null,
      routed_model: model || null,
      provider: 'anthropic',
      label: null,
      decision,
      est_input_tokens: est,
      cache_hit: false,
    },
  };
}

async function decide({ rawBody, config, classify, now, cache }) {
  const est = Math.floor(rawBody.length / 4);

  let body;
  try {
    body = JSON.parse(rawBody.toString('utf8'));
  } catch {
    return passthrough('malformed-passthrough', { key: null, model: null, est });
  }
  if (!body || typeof body.model !== 'string') {
    return passthrough('malformed-passthrough', { key: null, model: null, est });
  }

  const original = body.model;
  // Computed once here and stashed on every decision so planForward need not re-parse
  // the body to run the modality gate.
  const hasNonText = hasNonTextContent(body);
  // Computed once here and stashed on every decision so planForward can run the
  // privacy gate without re-parsing the body. Fail-closed: high-confidence secrets
  // must never reach a third-party provider (see hasSensitiveContent).
  const hasSensitive = hasSensitiveContent(body);

  // Fast paths (order matters).
  if (/haiku/.test(original)) {
    return passthrough('already-light', { key: null, model: original, est, hasNonText, hasSensitive });
  }
  if ((config.pinned || []).some((p) => original.startsWith(p))) {
    return passthrough('pinned', { key: null, model: original, est, hasNonText, hasSensitive });
  }

  const forbidLight = est > (config.maxHaikuInputTokens ?? 150000);
  const tiers = config.tiers || {};

  // A tier value is either "provider,model" or a bare model (implies anthropic).
  const parseTier = (v) => {
    if (typeof v !== 'string' || v === '') return null;
    const i = v.indexOf(',');
    if (i === -1) return { provider: 'anthropic', model: v };
    return { provider: v.slice(0, i).trim(), model: v.slice(i + 1).trim() };
  };
  // Returns a routing target {provider, model}. Context guard: an oversized body
  // forbids the light tier, so we resolve the standard tier in its place.
  const resolveTarget = (label) => {
    const eff = label === 'light' && forbidLight ? 'standard' : label;
    return parseTier(tiers[eff]) || { provider: 'anthropic', model: original };
  };

  const messages = Array.isArray(body.messages) ? body.messages : [];
  const sysText = extractSystemText(body.system).slice(0, 1024);
  const firstContent = messages.length
    ? JSON.stringify(messages[0] ? messages[0].content : '').slice(0, 1024)
    : '';
  const key = sha256(sysText + firstContent);
  const keyShort = key.slice(0, 12);

  const sessionMode = (config.mode || 'session') === 'session';

  const rewriteDecision = (decision, target, label, extra) => ({
    routed: target, // {provider, model}
    routedModel: target.model, // backward-compat string
    rewrite: target.provider === 'anthropic' && target.model !== original,
    hasNonText,
    hasSensitive,
    log: {
      key: keyShort,
      original_model: original,
      routed_model: target.model,
      provider: target.provider,
      label,
      decision,
      est_input_tokens: est,
      cache_hit: !!(extra && extra.cache_hit),
      ...(extra && extra.classifier_ms != null ? { classifier_ms: extra.classifier_ms } : {}),
    },
  });

  const fallback = (err) => ({
    routed: null,
    routedModel: null,
    rewrite: false,
    hasNonText,
    hasSensitive,
    log: {
      key: keyShort,
      original_model: original,
      routed_model: original,
      provider: 'anthropic',
      label: null,
      decision: 'fallback',
      fallback_reason: String((err && err.message) || err),
      est_input_tokens: est,
      cache_hit: false,
    },
  });

  if (sessionMode) {
    const cached = getCache(cache, key, now());
    if (cached) {
      return rewriteDecision('cache-hit', resolveTarget(cached.label), cached.label, {
        cache_hit: true,
      });
    }
    // Unseen key with a deep conversation: mid-session with no prior decision
    // (e.g. proxy restarted). Switching models now would cold-restart the
    // model-scoped prompt cache -> passthrough unmodified.
    if (messages.length > 2) {
      return passthrough('midflight-passthrough', { key: keyShort, model: original, est, hasNonText, hasSensitive });
    }
  }

  // Classify (session mode: messages.length <= 2; request mode: always).
  // No in-flight lock by design: Claude Code issues turns sequentially per
  // session, so the only way two same-key first turns race here is unusual;
  // if they do, both classify to the same tier — harmless, just one wasted call.
  let label, ms;
  try {
    const t0 = now();
    label = await classify(buildDigest(body, est));
    ms = now() - t0;
  } catch (e) {
    return fallback(e);
  }
  if (!LABELS.includes(label)) return fallback(new Error('unexpected-label:' + label));

  if (sessionMode) setCache(cache, key, { label, ts: now() });
  return rewriteDecision('routed', resolveTarget(label), label, { cache_hit: false, classifier_ms: ms });
}

module.exports = {
  decide,
  buildDigest,
  hasNonTextContent,
  hasSensitiveContent,
  extractSystemText,
  extractText,
  sha256,
  TTL_MS,
  MAX_ENTRIES,
};
