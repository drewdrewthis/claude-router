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

// Hop-by-hop headers we must not relay. content-length is recomputed by fetch/node.
// On the response side we also drop content-encoding: global fetch transparently
// decompresses the body, so a stale content-encoding would corrupt the client.
const HOP_REQ = new Set([
  'host', 'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'proxy-connection',
]);
const HOP_RES = new Set([
  'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'content-encoding',
]);

function loadConfig(configPath) {
  const file = configPath || process.env.ROUTER_CONFIG || path.join(__dirname, 'config.json');
  try {
    return { ...DEFAULTS, ...JSON.parse(fs.readFileSync(file, 'utf8')) };
  } catch {
    return { ...DEFAULTS };
  }
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}

// Stream the upstream response back byte-for-byte, flushing each chunk as it
// arrives (critical for SSE: first delta must reach the client before the
// upstream finishes). Status + headers relayed verbatim minus hop-by-hop.
async function forward(config, req, body, res) {
  let upstream;
  try {
    const headers = {};
    for (const [k, v] of Object.entries(req.headers)) if (!HOP_REQ.has(k)) headers[k] = v;
    upstream = await fetch(config.upstream + req.url, {
      method: req.method,
      headers,
      body: req.method === 'GET' || req.method === 'HEAD' ? undefined : body,
    });
  } catch (e) {
    if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' });
    res.end('claude-router upstream error: ' + String((e && e.message) || e));
    return;
  }
  const outHeaders = {};
  for (const [k, v] of upstream.headers) if (!HOP_RES.has(k)) outHeaders[k] = v;
  res.writeHead(upstream.status, outHeaders);
  if (upstream.body) {
    for await (const chunk of upstream.body) res.write(chunk);
  }
  res.end();
}

function makeHandler(config, cache, logger, fallbackTracker) {
  return async function handler(req, res) {
    const rawBody = await readBody(req);
    const pathname = new URL(req.url, 'http://x').pathname;

    // Fast path (a): anything that is not exactly POST /v1/messages is a
    // transparent, unlogged passthrough (count_tokens, GETs, etc.).
    if (req.method !== 'POST' || pathname !== '/v1/messages') {
      return forward(config, req, rawBody, res);
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

    return forward(config, req, body, res);
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
  const config = opts.config || loadConfig(opts.configPath);
  if (opts.upstream) config.upstream = opts.upstream;
  if (process.env.ROUTER_UPSTREAM) config.upstream = process.env.ROUTER_UPSTREAM;

  const logFile =
    opts.logFile || process.env.ROUTER_LOG_FILE || path.join(__dirname, 'decisions.jsonl');
  const port =
    opts.port != null ? opts.port : process.env.ROUTER_PORT ? Number(process.env.ROUTER_PORT) : config.port;

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

module.exports = { startServer, loadConfig, DEFAULTS };

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
