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

// Session-scoped, unambiguously-encoded cache key.
//
// Two defects motivated this shape:
//   1. COLLISION. The old key was `sha256(sysText + firstContent)` — a raw
//      concatenation of two variable-length strings, so ("ab","c") and ("a","bc")
//      hash identically. Hashing a tuple unambiguously requires an injective encoding
//      of the fields (NIST SP 800-185 TupleHash: "moving bytes from one input string
//      to an adjacent one" must change the digest). `JSON.stringify` of an array gives
//      that for free — it quotes, escapes, and delimits each element.
//   2. CROSS-SESSION SHARING. Content-only keys mean two DIFFERENT sessions that
//      open with the same system prompt and the same first message (the norm for
//      one tool talking to one repo) share a cache entry — one session's tier
//      silently decides the other's. Scoping by session id keeps stickiness within
//      a session, which is all `mode: "session"` ever promised.
//
// `sessionId` is nullable: a client that supplies no session identity degrades to
// the previous content-only behaviour rather than losing caching entirely.
function cacheKey(sessionId, sysText, firstContent) {
  return sha256(JSON.stringify([sessionId || '', sysText, firstContent]));
}

// Resolve a stable per-session identifier, most-specific first. Both sources were
// confirmed present on every POST /v1/messages from a live Claude Code session:
//   - `x-claude-code-session-id`: the documented per-session id. Anthropic's LLM-gateway
//     protocol reference says to "use it to aggregate all requests from one session
//     without parsing request bodies" — so it is preferred over the body.
//   - `body.metadata.user_id`: fallback for any Anthropic client that sets it. Treated as
//     an OPAQUE scope token: never parsed, never logged (the Messages API defines it as
//     an opaque abuse-detection id that must carry no identifying information).
// Returns null when neither is present -> content-only key (previous behaviour).
function sessionIdFrom(body, headers) {
  const h = (headers && headers['x-claude-code-session-id']) || null;
  if (typeof h === 'string' && h !== '') return h;
  const uid = body && body.metadata && body.metadata.user_id;
  if (typeof uid === 'string' && uid !== '') return uid;
  return null;
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

// True if ANY request text matches a high-confidence credential shape. The scan must
// cover EVERYTHING translate.js can put on the provider wire, not just prose:
//   - the system prompt, and string / text-block message content;
//   - text nested in tool_result blocks (extractText does not descend into them);
//   - tool_use `input`, serialized. This is the one that bites: a tool argument is
//     structured JSON with no `.text`, so `extractText` returns '' for it, yet
//     requestAnthropicToOpenai serializes it straight into
//     `tool_calls[].function.arguments`. A `Bash` call carrying an AWS key in its
//     `command` would sail past a text-only gate and reach the third-party provider.
// Fail-closed gate input: a match diverts the request to Anthropic in planForward.
// Defensive: missing/non-array fields collapse to false rather than throwing.
function hasSensitiveContent(body) {
  const messages = body && Array.isArray(body.messages) ? body.messages : [];
  let text = extractSystemText(body && body.system);
  for (const m of messages) {
    if (!m) continue;
    text += ' ' + extractText(m.content);
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (!b) continue;
        if (b.type === 'tool_result') {
          if (typeof b.content === 'string') text += ' ' + b.content;
          else if (Array.isArray(b.content)) text += ' ' + extractText(b.content);
        } else if (b.type === 'tool_use' && b.input != null) {
          try {
            text += ' ' + JSON.stringify(b.input);
          } catch {
            /* circular/unserializable input: nothing to scan */
          }
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

async function decide({ rawBody, config, classify, now, cache, headers }) {
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
  const key = cacheKey(sessionIdFrom(body, headers), sysText, firstContent);
  const keyShort = key.slice(0, 12);

  const sessionMode = (config.mode || 'session') === 'session';

  const rewriteDecision = (decision, target, label, extra) => ({
    routed: target, // {provider, model}
    routedModel: target.model, // backward-compat string
    rewrite: target.provider === 'anthropic' && target.model !== original,
    hasNonText,
    hasSensitive,
    // digest present ONLY on the classify path (telemetry classifier-span input under the
    // content-privacy gate); never in decision.log — it embeds prompt text (see lib/log.js).
    ...(extra && extra.digest != null ? { digest: extra.digest } : {}),
    // Full sha256 cache key (session mode only) so the handler can pin this entry noRewrite
    // after a rewrite-rejected 400 — decision.log.key is only the 12-char short form.
    ...(extra && extra.cacheKey != null ? { cacheKey: extra.cacheKey } : {}),
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
      // A prior turn's rewrite drew a 400 and the handler pinned this session (router.js
      // markNoRewrite): forward original bytes untouched rather than rewrite into the same
      // 400 every turn.
      if (cached.noRewrite) {
        return passthrough('rewrite-rejected-passthrough', {
          key: keyShort,
          model: original,
          est,
          hasNonText,
          hasSensitive,
        });
      }
      return rewriteDecision('cache-hit', resolveTarget(cached.label), cached.label, {
        cache_hit: true,
        cacheKey: key,
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
  const digest = buildDigest(body, est);
  try {
    const t0 = now();
    label = await classify(digest);
    ms = now() - t0;
  } catch (e) {
    return fallback(e);
  }
  if (!LABELS.includes(label)) return fallback(new Error('unexpected-label:' + label));

  if (sessionMode) setCache(cache, key, { label, ts: now() });
  return rewriteDecision('routed', resolveTarget(label), label, {
    cache_hit: false,
    classifier_ms: ms,
    digest,
    ...(sessionMode ? { cacheKey: key } : {}),
  });
}

module.exports = {
  decide,
  buildDigest,
  cacheKey,
  sessionIdFrom,
  hasNonTextContent,
  hasSensitiveContent,
  extractSystemText,
  extractText,
  sha256,
  TTL_MS,
  MAX_ENTRIES,
};
