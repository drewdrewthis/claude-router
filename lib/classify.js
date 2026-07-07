'use strict';
// Classifier call: one claude-haiku-4-5 completion that labels a request
// light | standard | heavy. Hard-bounded by an AbortController timeout so the
// caller's fail-open budget is respected. Anything unexpected throws -> fail-open.

const CLASSIFIER_SYSTEM =
  'You route coding-agent requests to a model tier. Reply with exactly one word: ' +
  'light, standard, or heavy. light = trivial/short single-step asks; ' +
  'standard = normal coding tasks; heavy = complex multi-file reasoning, ' +
  'architecture, long-horizon work.';

// If ROUTER_ANTHROPIC_API_KEY is set, use it as an x-api-key credential.
// Otherwise reuse the incoming request's credential headers verbatim (works for
// both API-key and OAuth/Bearer Claude Code sessions).
function authHeaders(incoming) {
  incoming = incoming || {};
  if (process.env.ROUTER_ANTHROPIC_API_KEY) {
    return {
      'x-api-key': process.env.ROUTER_ANTHROPIC_API_KEY,
      'anthropic-version': incoming['anthropic-version'] || '2023-06-01',
    };
  }
  const out = {};
  for (const h of ['authorization', 'x-api-key', 'anthropic-version', 'anthropic-beta']) {
    if (incoming[h] != null) out[h] = incoming[h];
  }
  return out;
}

async function classify({ digest, config, incomingHeaders }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.classifierTimeoutMs ?? 3000);
  try {
    const res = await fetch(`${config.upstream}/v1/messages`, {
      method: 'POST',
      signal: controller.signal,
      headers: { 'content-type': 'application/json', ...authHeaders(incomingHeaders) },
      body: JSON.stringify({
        model: config.classifierModel,
        max_tokens: 8,
        system: CLASSIFIER_SYSTEM,
        messages: [{ role: 'user', content: digest }],
      }),
    });
    if (!res.ok) throw new Error('classifier-http-' + res.status);
    const json = await res.json();
    const text = ((json.content && json.content[0] && json.content[0].text) || '')
      .trim()
      .toLowerCase();
    const m = text.match(/light|standard|heavy/);
    if (!m) throw new Error('classifier-unparseable');
    return m[0];
  } finally {
    clearTimeout(timer);
  }
}

module.exports = { classify, authHeaders, CLASSIFIER_SYSTEM };
