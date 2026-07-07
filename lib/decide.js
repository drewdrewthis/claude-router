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

function buildDigest(body, est) {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let text = '';
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i] && messages[i].role === 'user') {
      text = extractText(messages[i].content);
      break;
    }
  }
  const tools = Array.isArray(body.tools) ? body.tools.length : 0;
  const thinking = body.thinking ? true : false;
  return `${text.slice(0, 2000)}\n[meta est_tokens=${est} tools=${tools} thinking=${thinking}]`;
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

function passthrough(decision, { key, model, est }) {
  return {
    routedModel: null,
    rewrite: false,
    log: {
      key: key || null,
      original_model: model || null,
      routed_model: model || null,
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

  // Fast paths (order matters).
  if (/haiku/.test(original)) {
    return passthrough('already-light', { key: null, model: original, est });
  }
  if ((config.pinned || []).some((p) => original.startsWith(p))) {
    return passthrough('pinned', { key: null, model: original, est });
  }

  const forbidLight = est > (config.maxHaikuInputTokens ?? 150000);
  const tiers = config.tiers || {};
  const resolveModel = (label) => {
    let model = tiers[label] || original;
    if (label === 'light' && forbidLight) model = tiers.standard || model; // context guard
    return model;
  };

  const messages = Array.isArray(body.messages) ? body.messages : [];
  const sysText = extractSystemText(body.system).slice(0, 1024);
  const firstContent = messages.length
    ? JSON.stringify(messages[0] ? messages[0].content : '').slice(0, 1024)
    : '';
  const key = sha256(sysText + firstContent);
  const keyShort = key.slice(0, 12);

  const sessionMode = (config.mode || 'session') === 'session';

  const rewriteDecision = (decision, model, label, extra) => ({
    routedModel: model,
    rewrite: model !== original,
    log: {
      key: keyShort,
      original_model: original,
      routed_model: model,
      label,
      decision,
      est_input_tokens: est,
      cache_hit: !!(extra && extra.cache_hit),
      ...(extra && extra.classifier_ms != null ? { classifier_ms: extra.classifier_ms } : {}),
    },
  });

  const fallback = (err) => ({
    routedModel: null,
    rewrite: false,
    log: {
      key: keyShort,
      original_model: original,
      routed_model: original,
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
      return rewriteDecision('cache-hit', resolveModel(cached.label), cached.label, {
        cache_hit: true,
      });
    }
    // Unseen key with a deep conversation: mid-session with no prior decision
    // (e.g. proxy restarted). Switching models now would cold-restart the
    // model-scoped prompt cache -> passthrough unmodified.
    if (messages.length > 2) {
      return passthrough('midflight-passthrough', { key: keyShort, model: original, est });
    }
  }

  // Classify (session mode: messages.length <= 2; request mode: always).
  let label, ms;
  try {
    const t0 = now();
    label = await classify(buildDigest(body, est));
    ms = now() - t0;
  } catch (e) {
    return fallback(e);
  }
  if (!LABELS.includes(label)) return fallback(new Error('unexpected-label:' + label));

  if (sessionMode) setCache(cache, key, { label, model: tiers[label] || original, ts: now() });
  return rewriteDecision('routed', resolveModel(label), label, { cache_hit: false, classifier_ms: ms });
}

module.exports = {
  decide,
  buildDigest,
  extractSystemText,
  extractText,
  sha256,
  TTL_MS,
  MAX_ENTRIES,
};
