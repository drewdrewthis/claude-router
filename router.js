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

const DEFAULTS = {
  mode: 'session',
  tiers: { light: 'claude-haiku-4-5', standard: 'claude-sonnet-5', heavy: 'claude-opus-4-8' },
  pinned: [],
  classifierModel: 'claude-haiku-4-5',
  classifierTimeoutMs: 3000,
  maxHaikuInputTokens: 150000,
  upstream: 'https://api.anthropic.com',
  port: 3456,
};

// Upstream hosts we will ever send live credentials to. Anything else is a
// fail-closed startup error: a redirected upstream would receive the caller's
// Anthropic auth headers. Loopback stays allowed so local dev/tests work.
const ALLOWED_UPSTREAM_HOSTS = ['api.anthropic.com', '127.0.0.1', 'localhost'];

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
  return { ...DEFAULTS, ...base, tiers: { ...DEFAULTS.tiers, ...(base.tiers || {}) } };
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
async function forward(config, req, body, res) {
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) if (!HOP_REQ.has(k)) headers[k] = v;
  const upstream = await fetch(config.upstream + req.url, {
    method: req.method,
    headers,
    body: req.method === 'GET' || req.method === 'HEAD' ? undefined : body,
  });
  const outHeaders = {};
  for (const [k, v] of upstream.headers) if (!HOP_RES.has(k)) outHeaders[k] = v;
  res.writeHead(upstream.status, outHeaders);
  if (upstream.body) await pipeline(Readable.fromWeb(upstream.body), res);
  else res.end();
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
        });
      } catch (e) {
        decision = {
          routedModel: null,
          rewrite: false,
          log: {
            key: null,
            original_model: null,
            routed_model: null,
            label: null,
            decision: 'fallback',
            fallback_reason: String((e && e.message) || e),
            est_input_tokens: Math.floor(rawBody.length / 4),
            cache_hit: false,
          },
        };
      }

      let body = rawBody;
      if (decision.rewrite && decision.routedModel) {
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

      await forward(config, req, body, res);
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

  validateConfig(config); // fail-closed: exits on bad tiers/upstream

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

module.exports = { startServer, loadConfig, mergeConfig, validateConfig, DEFAULTS };

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
