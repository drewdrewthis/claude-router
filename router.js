#!/usr/bin/env node
'use strict';
// Local Anthropic Messages API proxy for Claude Code. Opt in per session via
//   ANTHROPIC_BASE_URL=http://127.0.0.1:3456
// It classifies a NEW conversation into a model tier, pins it session-sticky,
// rewrites body.model, and forwards to Anthropic. It NEVER breaks a request:
// any internal failure forwards the original bytes unmodified (fail-open).

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { Readable } = require('node:stream');
const { pipeline } = require('node:stream/promises');

const { decide } = require('./lib/decide');
const { classify } = require('./lib/classify');
const { createLogger } = require('./lib/log');
const { readCappedText, ERROR_BODY_MAX_BYTES } = require('./lib/http');
const {
  requestAnthropicToOpenai,
  responseOpenaiToAnthropic,
  streamOpenaiToAnthropic,
  singleShotSSE,
  toolNameMap,
} = require('./lib/translate');

const DEFAULTS = {
  mode: 'session',
  tiers: { light: 'claude-haiku-4-5', standard: 'claude-sonnet-5', heavy: 'claude-opus-4-8' },
  pinned: [],
  classifierModel: 'claude-haiku-4-5',
  classifierTimeoutMs: 3000,
  maxHaikuInputTokens: 150000,
  providerTimeoutMs: 30000,
  // Bounds time-to-response-headers on the Anthropic passthrough leg only (see
  // forward()). Deliberately just under UNDICI_HEADERS_TIMEOUT_MS so OUR bound is the
  // one that fires. Generous, because a non-stream completion withholds headers until
  // it is fully generated — do not confuse this with providerTimeoutMs.
  upstreamTimeoutMs: 240000,
  upstream: 'https://api.anthropic.com',
  port: 3456,
  providers: {
    anthropic: { type: 'anthropic' },
    nvidia: {
      type: 'openai',
      base_url: 'https://integrate.api.nvidia.com/v1',
      api_key_env: 'NVIDIA_API_KEY',
    },
  },
};

// Upstream hosts we will ever send live credentials to. Anything else is a
// fail-closed startup error: a redirected upstream would receive the caller's
// Anthropic auth headers. Loopback stays allowed so local dev/tests work.
const ALLOWED_UPSTREAM_HOSTS = ['api.anthropic.com', '127.0.0.1', 'localhost'];

// undici (the engine behind Node's global `fetch`) applies its own `headersTimeout`,
// default 300_000 ms, and there is no stdlib-only way to raise it — `fetch` would need
// a custom `dispatcher`, which means depending on `undici` directly. Verified on this
// runtime: a black-hole upstream rejects at ~300.7s with `UND_ERR_HEADERS_TIMEOUT`.
//
// Consequence: any `upstreamTimeoutMs` at or above this is a NO-OP, because undici
// aborts the fetch first. We therefore default below it and warn if an operator
// configures above it.
//
// undici's sibling `bodyTimeout` (also 300_000 ms) is an INTER-CHUNK idle timeout —
// the same guarantee as nginx `proxy_read_timeout` / envoy `stream_idle_timeout`. It
// already covers "upstream sent headers then stalled mid-SSE", so we deliberately do
// NOT hand-roll a body-idle timeout on top of it.
const UNDICI_HEADERS_TIMEOUT_MS = 300000;

// Hop-by-hop headers we must not relay. content-length is recomputed by fetch/node.
// On the response side we also drop content-encoding: global fetch transparently
// decompresses the body, so a stale content-encoding would corrupt the client.
const HOP_REQ = new Set([
  'host', 'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'proxy-connection',
]);
const HOP_RES = new Set([
  'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'content-encoding',
]);

// A generic Anthropic-style error envelope. We never echo internal error text to
// the client (that could leak upstream/config detail); the real error is logged.
function errorEnvelope(message) {
  return JSON.stringify({ type: 'error', error: { type: 'api_error', message } });
}

function fatal(msg) {
  console.error('[claude-router] FATAL: ' + msg);
  process.exit(1);
}

// Deep-merge a partial config over DEFAULTS. `tiers` merges key-by-key so a user
// who overrides only one tier keeps the other two.
function mergeConfig(base) {
  base = base || {};
  return {
    ...DEFAULTS,
    ...base,
    tiers: { ...DEFAULTS.tiers, ...(base.tiers || {}) },
    providers: { ...DEFAULTS.providers, ...(base.providers || {}) },
  };
}

function loadConfig(configPath) {
  const file = configPath || process.env.ROUTER_CONFIG || path.join(__dirname, 'config.json');
  try {
    return mergeConfig(JSON.parse(fs.readFileSync(file, 'utf8')));
  } catch {
    return mergeConfig({});
  }
}

// Fail-closed validation. Any violation exits the process — these are correctness
// and security controls, not something to degrade past.
function validateConfig(config) {
  for (const label of ['light', 'standard', 'heavy']) {
    const v = config.tiers && config.tiers[label];
    if (typeof v !== 'string' || v.trim() === '') {
      fatal(`config.tiers.${label} must resolve to a non-empty string (got ${JSON.stringify(v)}).`);
    }
  }

  let host;
  try {
    host = new URL(config.upstream).hostname;
  } catch {
    fatal(`invalid upstream URL: ${JSON.stringify(config.upstream)}`);
  }
  const allowed = new Set([...ALLOWED_UPSTREAM_HOSTS, ...(config.allowedUpstreamHosts || [])]);
  if (!allowed.has(host)) {
    fatal(
      `upstream host "${host}" is not allowlisted. Live credentials are only sent to ` +
        `${[...allowed].join(', ')}. Add it to config.allowedUpstreamHosts if this is intentional.`
    );
  }
  if (host !== 'api.anthropic.com') {
    console.error(`[claude-router] NOTICE: forwarding to non-default upstream ${config.upstream}`);
  }

  // A silently-inert config knob is exactly the class of bug this proxy exists to avoid.
  if ((config.upstreamTimeoutMs ?? 0) >= UNDICI_HEADERS_TIMEOUT_MS) {
    console.error(
      `[claude-router] WARNING: upstreamTimeoutMs=${config.upstreamTimeoutMs} is at or above ` +
        `undici's built-in ${UNDICI_HEADERS_TIMEOUT_MS}ms headersTimeout, so it will never fire — ` +
        `the fetch aborts first. Lower it below ${UNDICI_HEADERS_TIMEOUT_MS} to take effect.`
    );
  }

  // Provider base_urls are operator-configured (implicitly allowlisted) but must
  // still be a parseable URL and https (loopback exempt so local dev/tests work).
  for (const [name, p] of Object.entries(config.providers || {})) {
    if (!p || typeof p !== 'object') fatal(`provider "${name}" must be an object.`);
    if (p.type === 'openai') {
      let u;
      try {
        u = new URL(p.base_url);
      } catch {
        fatal(`provider "${name}": base_url ${JSON.stringify(p.base_url)} is not a valid URL.`);
      }
      const loopback = u.hostname === '127.0.0.1' || u.hostname === 'localhost';
      if (u.protocol !== 'https:' && !loopback) {
        fatal(`provider "${name}": base_url must be https (got ${u.protocol}//${u.hostname}).`);
      }
    }
  }
}

// Warn loudly for any provider referenced in tiers whose credential env is unset.
// (Requests to it will fail open to Anthropic — see the handler.)
function warnMissingProviderKeys(config) {
  const referenced = new Set();
  for (const v of Object.values(config.tiers || {})) {
    if (typeof v === 'string' && v.includes(',')) referenced.add(v.slice(0, v.indexOf(',')).trim());
  }
  for (const name of referenced) {
    const p = (config.providers || {})[name];
    if (!p) {
      console.error(`[claude-router] WARNING: tier references unknown provider "${name}".`);
      continue;
    }
    if (p.type === 'openai' && p.api_key_env && !process.env[p.api_key_env]) {
      console.error(
        `[claude-router] WARNING: provider "${name}" credential env ${p.api_key_env} is unset — ` +
          `requests routed to it will FAIL OPEN to Anthropic.`
      );
    }
  }
}

function resolvePort(opts, config) {
  let p;
  if (opts.port != null) p = opts.port;
  else if (process.env.ROUTER_PORT != null && process.env.ROUTER_PORT !== '') p = Number(process.env.ROUTER_PORT);
  else p = config.port;
  // 0 is the ephemeral-bind sentinel (used by tests); otherwise require 1-65535.
  if (!Number.isInteger(p) || p < 0 || p > 65535) {
    fatal(`invalid port ${JSON.stringify(p)} — must be an integer in 0..65535.`);
  }
  return p;
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}

// Stream the upstream response back byte-for-byte with backpressure + cleanup via
// pipeline (critical for SSE: first delta reaches the client before upstream
// finishes). May throw — the handler's try/catch owns error/abort recovery.
//
// TIMEOUT SHAPE (nginx's proxy_connect_timeout / proxy_read_timeout split, which envoy
// spells route-timeout vs stream_idle_timeout; forwardOpenai's stream branch already
// does the same thing): bound TIME-TO-RESPONSE-HEADERS, unbind the moment they arrive,
// and let an inter-chunk IDLE timeout guard the body. Never a total-duration timeout —
// an SSE completion legitimately streams for minutes, and aborting mid-body would
// truncate the client's answer.
//
// We hand-roll only the first half. The body-idle half is already provided by undici's
// `bodyTimeout` (see UNDICI_HEADERS_TIMEOUT_MS) — checked, present, so not reinvented.
//
// What this actually buys: before, `forward()` set no bound of its own and silently
// inherited undici's 300s `headersTimeout` — invisible, unconfigurable, and impossible
// to tighten. Now the bound is explicit, configurable, and defaults below undici's so
// it is the one that fires. A non-stream request withholds headers until the whole
// completion is generated, so this must stay generous: its job is to stop a HUNG
// upstream leaking the client socket, not to cap slow-but-healthy work.
async function forward(config, req, body, res) {
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) if (!HOP_REQ.has(k)) headers[k] = v;

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), config.upstreamTimeoutMs ?? 600000);
  let upstream;
  try {
    upstream = await fetch(config.upstream + req.url, {
      method: req.method,
      headers,
      body: req.method === 'GET' || req.method === 'HEAD' ? undefined : body,
      signal: ac.signal,
    });
  } finally {
    // Headers are in (or the fetch threw): unbind before any body streaming.
    clearTimeout(timer);
  }

  const outHeaders = {};
  for (const [k, v] of upstream.headers) if (!HOP_RES.has(k)) outHeaders[k] = v;
  res.writeHead(upstream.status, outHeaders);
  if (upstream.body) await pipeline(Readable.fromWeb(upstream.body), res);
  else res.end();
}

// Translate the Anthropic request onto an OpenAI-compatible provider and forward.
// SECURITY INVARIANT: the client's Anthropic credential headers are NEVER sent
// here — we send ONLY `Authorization: Bearer <provider env key>`.
//
// Returns a small result descriptor so the handler owns fail-open orchestration
// (this keeps forwardOpenai free of logger/config):
//   { fallback: '<reason>' }  failed BEFORE writing any bytes (fetch threw/aborted
//                             or !provRes.ok) — the handler serves via Anthropic.
//   { ok: true }              committed to responding (headers written) — no fallback.
async function forwardOpenai(plan, rawBody, res, timeoutMs) {
  const { provider, apiKey, model, name } = plan;
  const reportModel = `${name}/${model}`;

  let anthropicBody;
  try {
    anthropicBody = JSON.parse(rawBody.toString('utf8'));
  } catch {
    anthropicBody = {};
  }
  const stream = anthropicBody.stream === true;
  const openaiReq = requestAnthropicToOpenai(anthropicBody, model);
  // Tool names are mangled on the way out (OpenAI's 64-char/charset limit); this
  // rebuilds the reverse map so the client sees back the names it declared.
  const toolNames = toolNameMap(anthropicBody.tools);

  // Bound the provider request with an AbortController on `timeoutMs`. Once headers
  // arrive we branch on stream vs non-stream:
  //   STREAM: clear the timer immediately so a legitimately long streaming body is
  //           never aborted mid-flight (providers send headers before generating).
  //   NON-STREAM: keep the timer ARMED through the body read (`provRes.json()`) below,
  //           so a provider that sends 200 + headers then stalls its body still fails
  //           open within `timeoutMs` instead of hanging the client (no client bytes
  //           written yet). A pre-bytes failure returns a fallback descriptor.
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  let provRes;
  try {
    provRes = await fetch(provider.base_url.replace(/\/$/, '') + '/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(openaiReq),
      signal: ac.signal,
    });
  } catch (e) {
    clearTimeout(timer);
    console.error(`[claude-router] provider "${name}" fetch failed: ${String((e && e.message) || e).slice(0, 200)}`);
    return { fallback: `provider-fetch:${name}` };
  }
  // An ERROR body is never "the long streaming body", so it stays bounded in BOTH
  // dimensions regardless of `stream`:
  //   TIME  — the abort timer stays armed across the read (this is why the stream
  //           branch's clearTimeout now happens AFTER this block, not before it;
  //           previously a streaming request cleared the timer first and then read
  //           the error body with no deadline at all).
  //   BYTES — readCappedText stops at ERROR_BODY_MAX_BYTES and cancels the stream,
  //           so a provider that answers 500 with a gigabyte cannot OOM the router.
  if (!provRes.ok) {
    const detail = await readCappedText(provRes, ERROR_BODY_MAX_BYTES);
    clearTimeout(timer);
    console.error(`[claude-router] provider "${name}" error ${provRes.status}: ${detail.slice(0, 200)}`);
    return { fallback: `provider-http-${provRes.status}` };
  }

  if (stream) clearTimeout(timer); // stream: unbind before the (long) body streams

  if (stream) {
    // The client asked for streaming, so we always answer with SSE.
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    const ctype = provRes.headers.get('content-type') || '';
    if (!ctype.includes('text/event-stream')) {
      // Provider ignored stream:true and returned a single JSON body: buffer it and
      // re-envelope as a one-shot Anthropic stream (never an empty/hung stream).
      let json = null;
      try {
        json = await provRes.json();
      } catch {
        json = null;
      }
      for (const ev of singleShotSSE(json, reportModel, toolNames)) res.write(ev);
      res.end();
      return { ok: true };
    }
    const rawChunks = provRes.body ? Readable.fromWeb(provRes.body) : [];
    for await (const ev of streamOpenaiToAnthropic(rawChunks, reportModel, toolNames)) res.write(ev);
    res.end();
    return { ok: true };
  } else {
    // Non-stream: the timer is still armed, so a stalled body aborts `.json()` and we
    // fail open (no client bytes written yet). Clear it once the body resolves.
    let json;
    try {
      json = await provRes.json();
    } catch (e) {
      clearTimeout(timer);
      console.error(`[claude-router] provider "${name}" body read failed/timed out: ${String((e && e.message) || e).slice(0, 200)}`);
      return { fallback: `provider-body:${name}` };
    }
    clearTimeout(timer);
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify(responseOpenaiToAnthropic(json, reportModel, toolNames)));
    return { ok: true };
  }
}

// Resolve a routed decision into a concrete forwarding plan. Routes fail OPEN (rewritten
// to an Anthropic passthrough fallback, the reason recorded in the log) when the provider
// is unknown/misconfigured, its credential env is missing, the request carries non-text
// content a text-only provider cannot consume (the modality gate), OR the request text
// carries a high-confidence secret (the privacy gate). Mutates decision.log.
function planForward(config, decision) {
  const routed = decision.routed;
  if (!routed || routed.provider === 'anthropic') return null; // handled by Anthropic path

  const provider = (config.providers || {})[routed.provider];
  const apiKey = provider && provider.api_key_env ? process.env[provider.api_key_env] : null;

  if (!provider || provider.type !== 'openai') {
    failOpen(decision, `provider-unknown:${routed.provider}`);
    return null;
  }
  // MODALITY GATE: a non-text request (image/document/audio, or a tool_result embedding
  // one) must never reach a text-only model — translate.js faithfully emits an OpenAI
  // image_url block the model cannot use, so the provider 400s or silently answers blind.
  // Checked before the credential check: routing a non-text turn to a text model is wrong
  // even with a valid key. FIRST CUT: no openai provider is vision-capable (there is no
  // per-provider vision flag yet), so ANY non-text content forces the Anthropic path. A
  // follow-up adds a per-provider/-model vision-capability check to permit vision routes.
  if (decision.hasNonText) {
    failOpen(decision, 'modality:non-text-to-text-model');
    return null;
  }
  // PRIVACY GATE: a request whose text carries a high-confidence secret (API key, token,
  // private key, SSN -- see hasSensitiveContent) must never reach a third-party provider.
  // Fail closed to Anthropic (the account the client already trusts). Checked BEFORE the
  // credential check so it diverts even with a valid provider key. False positives only
  // forgo cheap routing (functionally safe); semantic PII is a future extension.
  if (decision.hasSensitive) {
    failOpen(decision, 'privacy:sensitive-content-to-thirdparty');
    return null;
  }
  if (!apiKey) {
    failOpen(decision, `provider-key-missing:${routed.provider}`);
    return null;
  }
  return { provider, apiKey, model: routed.model, name: routed.provider };
}

function failOpen(decision, reason) {
  decision.routed = null;
  decision.routedModel = null;
  decision.rewrite = false;
  decision.log.decision = 'fallback';
  decision.log.fallback_reason = reason;
  decision.log.provider = 'anthropic';
  decision.log.routed_model = decision.log.original_model;
}

function makeHandler(config, cache, logger, fallbackTracker) {
  return async function handler(req, res) {
    // Swallow client-side disconnects (ECONNRESET/EPIPE) so an aborted socket
    // never surfaces as an unhandled 'error' event and crashes the process.
    req.on('error', () => {});
    res.on('error', () => {});

    try {
      const rawBody = await readBody(req);
      const pathname = new URL(req.url, 'http://x').pathname;

      // Fast path (a): anything not exactly POST /v1/messages is a transparent,
      // unlogged passthrough (count_tokens, GETs, etc.).
      if (req.method !== 'POST' || pathname !== '/v1/messages') {
        await forward(config, req, rawBody, res);
        return;
      }

      let decision;
      try {
        decision = await decide({
          rawBody,
          config,
          classify: (digest) => classify({ digest, config, incomingHeaders: req.headers }),
          now: Date.now,
          cache,
          headers: req.headers, // session scoping only (x-claude-code-session-id); never logged
        });
      } catch (e) {
        decision = {
          routed: null,
          routedModel: null,
          rewrite: false,
          log: {
            key: null,
            original_model: null,
            routed_model: null,
            provider: 'anthropic',
            label: null,
            decision: 'fallback',
            fallback_reason: String((e && e.message) || e),
            est_input_tokens: Math.floor(rawBody.length / 4),
            cache_hit: false,
          },
        };
      }

      // Non-anthropic routes resolve to a forwarding plan (or fail open here).
      const openaiPlan = planForward(config, decision);

      let body = rawBody;
      if (!openaiPlan && decision.rewrite && decision.routedModel) {
        try {
          const obj = JSON.parse(rawBody.toString('utf8'));
          obj.model = decision.routedModel;
          body = Buffer.from(JSON.stringify(obj));
        } catch {
          body = rawBody; // fail-open: never corrupt the forwarded request
        }
      }

      // Await the log write so concurrent requests each land a complete line
      // before their response completes.
      try {
        await logger.write(decision.log);
      } catch {
        /* logging must never break a request */
      }
      fallbackTracker(decision.log.decision);

      if (openaiPlan) {
        const result = await forwardOpenai(openaiPlan, rawBody, res, config.providerTimeoutMs);
        // Runtime fail-open: the provider failed BEFORE any response bytes were
        // written (fetch threw/aborted or !ok). Serve via Anthropic with the ORIGINAL
        // bytes/model. body === rawBody here (the rewrite branch is !openaiPlan-guarded).
        if (result && result.fallback && !res.headersSent) {
          try {
            await logger.write({
              ...decision.log,
              decision: 'fallback',
              fallback_reason: result.fallback,
              provider: 'anthropic',
              routed_model: decision.log.original_model,
            });
          } catch {
            /* logging must never break a request */
          }
          fallbackTracker('fallback');
          await forward(config, req, rawBody, res);
        }
      } else {
        await forward(config, req, body, res);
      }
    } catch (e) {
      // Covers: aborted upload, client destroying the socket mid-SSE, upstream
      // fetch failure, and upstream body iterator errors. Never let it escape.
      console.error('[claude-router] request error:', (e && e.stack) || e);
      if (!res.headersSent) {
        res.writeHead(502, { 'content-type': 'application/json' });
        res.end(errorEnvelope('claude-router: upstream request failed'));
      } else {
        res.destroy();
      }
    }
  };
}

// Rolling counter: if the last 20 decisions were all fallbacks, the router is
// effectively a no-op. Warn loudly, once, until it recovers.
function makeFallbackTracker() {
  const recent = [];
  let warned = false;
  return function track(decision) {
    recent.push(decision === 'fallback');
    if (recent.length > 20) recent.shift();
    const allFallback = recent.length === 20 && recent.every(Boolean);
    if (allFallback && !warned) {
      console.error(
        '[claude-router] ERROR: last 20 decisions all fell back — routing is effectively disabled.'
      );
      warned = true;
    } else if (recent.length === 20 && !allFallback) {
      warned = false;
    }
  };
}

function readOauthToken() {
  try {
    const p = path.join(os.homedir(), '.claude', '.credentials.json');
    const j = JSON.parse(fs.readFileSync(p, 'utf8'));
    return j.claudeAiOauth && j.claudeAiOauth.accessToken;
  } catch {
    return null;
  }
}

// Boot ping: if a credential is available, fire one classifier call. A 4xx means
// the classifier will silently no-op for every request whose own creds also fail.
async function selfCheck(config) {
  const apiKey = process.env.ROUTER_ANTHROPIC_API_KEY;
  const token = apiKey || readOauthToken();
  if (!token) return;
  const incoming = apiKey
    ? {}
    : {
        authorization: `Bearer ${token}`,
        'anthropic-beta': 'oauth-2025-04-20',
        'anthropic-version': '2023-06-01',
      };
  try {
    await classify({ digest: 'ping\n[meta est_tokens=1 tools=0 thinking=false]', config, incomingHeaders: incoming });
  } catch (e) {
    if (/classifier-http-4/.test(String((e && e.message) || e))) {
      console.error(
        '[claude-router] WARNING: classifier self-check returned ' + e.message +
          ' — routing will SILENTLY NO-OP for any request whose own credentials also fail.'
      );
    }
  }
}

async function startServer(opts = {}) {
  let config = opts.config ? mergeConfig(opts.config) : loadConfig(opts.configPath);
  if (opts.upstream) config.upstream = opts.upstream;
  if (process.env.ROUTER_UPSTREAM) config.upstream = process.env.ROUTER_UPSTREAM;

  validateConfig(config); // fail-closed: exits on bad tiers/upstream/providers
  warnMissingProviderKeys(config);

  const logFile =
    opts.logFile || process.env.ROUTER_LOG_FILE || path.join(__dirname, 'decisions.jsonl');
  const port = resolvePort(opts, config);

  const cache = new Map();
  const logger = createLogger(logFile);
  const server = http.createServer(makeHandler(config, cache, logger, makeFallbackTracker()));

  await new Promise((r) => server.listen(port, '127.0.0.1', r));
  return {
    server,
    config,
    port: server.address().port,
    logFile,
    close: () => new Promise((r) => server.close(r)),
  };
}

module.exports = {
  startServer,
  loadConfig,
  mergeConfig,
  validateConfig,
  warnMissingProviderKeys,
  DEFAULTS,
};

if (require.main === module) {
  startServer({})
    .then(({ port, config }) => {
      console.error(`[claude-router] listening on http://127.0.0.1:${port} -> ${config.upstream}`);
      selfCheck(config);
    })
    .catch((e) => {
      console.error('[claude-router] failed to start:', e);
      process.exit(1);
    });
}
