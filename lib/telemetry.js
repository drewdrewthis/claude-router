'use strict';
// Zero-dependency OTLP/HTTP (JSON-encoded) trace exporter for LangWatch. Node core
// only: node:crypto for span/trace IDs, node:os for host.name, global fetch for the POST.
//
// HARD INVARIANT: telemetry must NEVER break, block, or slow a proxied request.
// recordRequest() is synchronous and only enqueues; the real export is batched on an
// unref()'d interval and every network error is swallowed. Nothing here is ever awaited
// in the request path.
//
// Proto3 JSON encoding gotchas (LangWatch returns HTTP 200 but SILENTLY rejects a
// malformed span, counted in partialSuccess.rejectedSpans): int64 fields are STRINGS
// ({intValue:"3"}), trace/span IDs are hex strings, and every span MUST carry its own
// `attributes` array (`[]` if empty) or the span is dropped inside the 200.

const crypto = require('node:crypto');
const os = require('node:os');

const DEFAULT_ENDPOINT_BASE = 'https://app.langwatch.ai/api/otel';
const FLUSH_THRESHOLD = 32; // export eagerly once this many spans are queued
const EXPORT_TIMEOUT_MS = 5000; // hard bound on a single export POST

// --- OTLP attribute helpers (Proto3 JSON value shapes) ---------------------------
function strAttr(key, v) { return { key, value: { stringValue: String(v) } }; }
function boolAttr(key, v) { return { key, value: { boolValue: Boolean(v) } }; }
function intAttr(key, v) { return { key, value: { intValue: String(Math.trunc(Number(v))) } }; }

// Push helpers that OMIT null/undefined so we never emit a `"null"` string attribute.
// A false boolean or a 0 int is a real value and IS emitted.
function pushStr(arr, key, v) { if (v != null) arr.push(strAttr(key, v)); }
function pushBool(arr, key, v) { if (v != null) arr.push(boolAttr(key, v)); }
function pushInt(arr, key, v) {
  if (v == null) return;
  const n = Number(v);
  if (Number.isFinite(n)) arr.push(intAttr(key, n));
}

function traceIdHex() { return crypto.randomBytes(16).toString('hex'); } // 32 hex chars
function spanIdHex() { return crypto.randomBytes(8).toString('hex'); }   // 16 hex chars

// epoch ms -> int64 unix-nanos STRING. ms*1e6 overflows Number.MAX_SAFE_INTEGER, so the
// multiply is done in BigInt to avoid silent float precision loss.
function msToNano(ms) {
  const safe = Number.isFinite(ms) ? Math.round(ms) : Date.now();
  return (BigInt(safe) * 1000000n).toString();
}

// Append the OTLP traces path exactly once: `<base>/v1/traces`, trailing slash(es)
// stripped; if the base already targets /v1/traces (a user pasted the full URL) it is
// left untouched rather than double-appended.
function tracesUrl(base) {
  const b = String(base).replace(/\/+$/, '');
  return /\/v1\/traces$/.test(b) ? b : b + '/v1/traces';
}

// Resolve exporter options from config + env. Exported for unit testing.
function resolveTelemetryOptions(config, env) {
  env = env || {};
  const tconf = (config && config.telemetry) || {};

  // Endpoint: an explicit override, else LangWatch's default.
  const explicitEndpoint = env.LANGWATCH_ENDPOINT || env.OTEL_EXPORTER_OTLP_ENDPOINT || '';
  const endpoint = tracesUrl(explicitEndpoint || DEFAULT_ENDPOINT_BASE);

  // apiKey: explicit LANGWATCH_API_KEY (LangWatch-specific by name) is trusted unconditionally.
  // A token scavenged from OTEL_EXPORTER_OTLP_HEADERS is adopted ONLY when the operator has
  // EXPLICITLY pointed the exporter somewhere (LANGWATCH_ENDPOINT / OTEL_EXPORTER_OTLP_ENDPOINT).
  // Without an explicit endpoint the exporter defaults to LangWatch, so adopting a stray OTEL
  // Authorization header (meant for some OTHER collector) would misdirect that foreign token to
  // LangWatch — so there we adopt nothing and telemetry stays disabled.
  let apiKey = env.LANGWATCH_API_KEY || '';
  if (!apiKey && explicitEndpoint && typeof env.OTEL_EXPORTER_OTLP_HEADERS === 'string') {
    const m = env.OTEL_EXPORTER_OTLP_HEADERS.match(/Authorization\s*=\s*Bearer\s+([^\s,]+)/i);
    if (m) apiKey = m[1];
  }

  // Enabled unless: no key, explicit ROUTER_TELEMETRY=0, or config.telemetry.enabled===false.
  const enabled =
    Boolean(apiKey) && env.ROUTER_TELEMETRY !== '0' && tconf.enabled !== false;

  const captureContent =
    tconf.captureContent === true || env.ROUTER_TELEMETRY_CAPTURE_CONTENT === '1';

  return { enabled, endpoint, apiKey, captureContent };
}

// router.fallback_reason is emitted on the ROOT span, which the content gate does NOT cover.
// On the classifier-error path it is String(err.message) — potentially free-form text. Map it
// through a fixed allowlist so a third-party observability backend only ever sees a known reason
// CODE (or 'other'); every allowlisted value is static or config/status-derived, never content.
const FALLBACK_REASON_STATIC = new Set([
  'rewrite-rejected-400',
  'modality:non-text-to-text-model',
  'privacy:sensitive-content-to-thirdparty',
  'classifier-unparseable',
]);
// Prefixes whose tail is a provider name (config) or an HTTP status — safe to keep verbatim.
const FALLBACK_REASON_PREFIXES = [
  'provider-unknown:',
  'provider-key-missing:',
  'provider-fetch:',
  'provider-body:',
  'provider-http-',
  'classifier-http-',
];
function normalizeFallbackReason(reason) {
  if (typeof reason !== 'string' || reason === '') return null;
  if (FALLBACK_REASON_STATIC.has(reason)) return reason;
  for (const p of FALLBACK_REASON_PREFIXES) if (reason.startsWith(p)) return reason;
  return 'other'; // notably the classifier-error String(err.message) path
}

function createTelemetry(options) {
  options = options || {};
  const {
    enabled = false,
    endpoint,
    apiKey,
    captureContent = false,
    serviceName = 'claude-router',
    flushIntervalMs = 2000,
    maxQueue = 512,
    fetchImpl,
  } = options;

  const doFetch = fetchImpl || (typeof fetch === 'function' ? fetch : undefined);
  const counters = { sent: 0, failed: 0, dropped: 0, rejected: 0 };
  let queue = [];

  // A client is only "active" when it can actually export. When inactive we return a
  // no-op object — created unconditionally so the request handler needs no null checks,
  // and a disabled client performs ZERO fetch calls and creates NO interval.
  const active = Boolean(enabled && endpoint && apiKey && doFetch);
  if (!active) {
    return {
      recordRequest() {},
      shutdown() { return Promise.resolve(); },
      stats() { return { queued: 0, sent: 0, failed: 0, dropped: 0, rejected: 0 }; },
    };
  }

  function stats() {
    return {
      queued: queue.length,
      sent: counters.sent,
      failed: counters.failed,
      dropped: counters.dropped,
      rejected: counters.rejected,
    };
  }

  // Wrap a flat span list into a single OTLP ResourceSpans payload. Resource attributes
  // identify the service; spans from many requests share one scope (traceId groups them).
  function resourceSpans(spans) {
    return {
      resourceSpans: [
        {
          resource: {
            attributes: [strAttr('service.name', serviceName), strAttr('host.name', os.hostname())],
          },
          scopeSpans: [{ scope: { name: 'claude-router' }, spans }],
        },
      ],
    };
  }

  async function flush() {
    if (!queue.length) return;
    const batch = queue;
    queue = [];
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), EXPORT_TIMEOUT_MS);
    if (timer.unref) timer.unref();
    try {
      const res = await doFetch(endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
        body: JSON.stringify(resourceSpans(batch)),
        signal: ac.signal,
      });
      if (!res || !res.ok) {
        counters.failed += batch.length;
        return;
      }
      counters.sent += batch.length;
      // HTTP 200 can still carry per-span rejections; surface them for observability.
      let body = null;
      try { body = await res.json(); } catch { body = null; }
      const rejected = body && body.partialSuccess ? Number(body.partialSuccess.rejectedSpans) : 0;
      if (Number.isFinite(rejected) && rejected > 0) counters.rejected += rejected;
    } catch {
      // Fire-and-forget: swallow every network/timeout/abort error. A telemetry failure
      // must never surface into the request path.
      counters.failed += batch.length;
    } finally {
      clearTimeout(timer);
    }
  }

  function enqueue(spans) {
    for (const s of spans) queue.push(s);
    // Bounded queue: drop the OLDEST spans first (newest telemetry is the most useful).
    while (queue.length > maxQueue) {
      queue.shift();
      counters.dropped++;
    }
    if (queue.length >= FLUSH_THRESHOLD) flush().catch(() => {});
  }

  function recordRequest(entry) {
    // Belt-and-suspenders: recordRequest must never throw into the handler.
    try {
      entry = entry || {};
      const decision = entry.decision || {};
      const log = decision.log || {};
      const startMs = Number.isFinite(entry.startMs) ? entry.startMs : Date.now();
      const endMs = Number.isFinite(entry.endMs) ? entry.endMs : startMs;

      const tid = traceIdHex();
      const rootId = spanIdHex();

      // ----------------------------------------------------------------------------
      // CONTENT PRIVACY RULE (load-bearing).
      // The router already refuses to send secret-bearing prompts to a third-party
      // *model provider* (decide.js hasSensitiveContent + router.js privacy gate). A
      // third-party *observability* backend gets the SAME treatment: prompt-derived text
      // leaves this process only when the operator opted in (captureContent) AND the turn is
      // EXPLICITLY marked non-sensitive. hasSensitive must be === false (not merely "not true"):
      // an undefined/unknown value fails CLOSED — content is withheld. The digest embeds the
      // user's prompt, so capturing it IS prompt capture — hence the gate. The root span NEVER
      // carries prompt text.
      const allowContent = captureContent && decision.hasSensitive === false;
      // ----------------------------------------------------------------------------

      const rootAttrs = [];
      pushStr(rootAttrs, 'langwatch.span.type', 'span');
      pushStr(rootAttrs, 'langwatch.thread_id', log.key); // groups a conversation
      pushStr(rootAttrs, 'router.decision', log.decision);
      pushStr(rootAttrs, 'router.label', entry.label);
      pushStr(rootAttrs, 'router.provider', log.provider);
      pushStr(rootAttrs, 'router.original_model', log.original_model);
      pushStr(rootAttrs, 'router.routed_model', log.routed_model);
      pushBool(rootAttrs, 'router.rewrite', decision.rewrite);
      pushBool(rootAttrs, 'router.cache_hit', log.cache_hit);
      pushInt(rootAttrs, 'router.est_input_tokens', log.est_input_tokens);
      pushBool(rootAttrs, 'router.has_non_text', decision.hasNonText);
      pushBool(rootAttrs, 'router.has_sensitive', decision.hasSensitive);
      // Always emitted (even when false) so a viewer can tell content was deliberately withheld.
      rootAttrs.push(boolAttr('router.content_captured', allowContent));
      // Emitted ONLY when it happened: our rewrite drew a 400 and we re-served original bytes.
      if (entry.rewriteRejected === true) rootAttrs.push(boolAttr('router.rewrite_rejected', true));
      // Content-level pre-flight decline reasons (comma-joined) when the router declined to route.
      if (Array.isArray(entry.capabilityBlockers) && entry.capabilityBlockers.length) {
        pushStr(rootAttrs, 'router.capability_blockers', entry.capabilityBlockers.join(','));
      }
      pushInt(rootAttrs, 'http.response.status_code', entry.httpStatus);
      pushStr(rootAttrs, 'router.fallback_reason', normalizeFallbackReason(log.fallback_reason));

      const spans = [
        {
          traceId: tid,
          spanId: rootId,
          name: 'claude_router.request',
          kind: 2, // SERVER
          startTimeUnixNano: msToNano(startMs),
          endTimeUnixNano: msToNano(endMs),
          attributes: rootAttrs,
        },
      ];

      // Classifier child — ONLY when the classifier actually ran. Times are approximate:
      // classification happens first in the request, so we anchor it at the request start.
      if (entry.classifierMs != null) {
        const cAttrs = [];
        pushStr(cAttrs, 'langwatch.span.type', 'llm');
        pushStr(cAttrs, 'gen_ai.request.model', entry.classifierModel);
        pushStr(cAttrs, 'gen_ai.system', 'anthropic');
        if (allowContent) {
          // The ONLY place prompt-derived text may be emitted (see the gate above).
          pushStr(cAttrs, 'langwatch.input', JSON.stringify({ type: 'text', value: entry.digest }));
          pushStr(cAttrs, 'langwatch.output', JSON.stringify({ type: 'text', value: entry.label }));
        }
        spans.push({
          traceId: tid,
          spanId: spanIdHex(),
          parentSpanId: rootId,
          name: 'claude_router.classifier',
          kind: 3, // CLIENT
          startTimeUnixNano: msToNano(startMs),
          endTimeUnixNano: msToNano(startMs + Number(entry.classifierMs)),
          attributes: cAttrs,
        });
      }

      // Upstream child — ONLY when an upstream call happened. Anchored at the request end
      // (we have its duration, not its absolute start).
      if (entry.upstream) {
        const up = entry.upstream;
        const uAttrs = [];
        pushStr(uAttrs, 'langwatch.span.type', 'llm');
        pushStr(uAttrs, 'gen_ai.request.model', up.model);
        pushStr(uAttrs, 'gen_ai.system', up.provider);
        if (up.usage) {
          pushInt(uAttrs, 'gen_ai.usage.input_tokens', up.usage.input_tokens);
          pushInt(uAttrs, 'gen_ai.usage.output_tokens', up.usage.output_tokens);
        }
        const uMs = Number.isFinite(up.ms) ? up.ms : 0;
        spans.push({
          traceId: tid,
          spanId: spanIdHex(),
          parentSpanId: rootId,
          name: 'claude_router.upstream',
          kind: 3, // CLIENT
          startTimeUnixNano: msToNano(endMs - uMs),
          endTimeUnixNano: msToNano(endMs),
          attributes: uAttrs,
        });
      }

      enqueue(spans);
    } catch {
      // Telemetry must never break a request.
    }
  }

  const interval = setInterval(() => { flush().catch(() => {}); }, flushIntervalMs);
  if (interval.unref) interval.unref(); // never hold the process open for telemetry

  let shutdownPromise = null;
  function shutdown() {
    // Safe to call twice: memoize the final flush.
    if (shutdownPromise) return shutdownPromise;
    clearInterval(interval);
    shutdownPromise = flush(); // one final, bounded flush (EXPORT_TIMEOUT_MS)
    return shutdownPromise;
  }

  return { recordRequest, shutdown, stats };
}

module.exports = { createTelemetry, resolveTelemetryOptions, tracesUrl };
