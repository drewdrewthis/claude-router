# claude-router

A local Anthropic Messages API proxy for Claude Code. It classifies a **new**
conversation into a model tier (`light` / `standard` / `heavy`) with one cheap
`claude-haiku-4-5` call, pins that choice for the whole session, rewrites
`body.model`, and forwards to `https://api.anthropic.com`.

It **never breaks a request**: any internal failure (parse error, classifier
timeout, unexpected label) forwards the original bytes unmodified — **fail-open**.

## Why

- **Rate-limit headroom (the real win on a Max subscription).** Trivial asks get
  routed to Haiku, normal work to Sonnet, and only genuinely hard work to Opus.
  You spend your Opus quota where it matters instead of on "rename this variable".
- **Session-sticky by design.** Anthropic's prompt cache is *model-scoped*.
  Switching models mid-conversation cold-restarts that cache and **increases** net
  token spend. So we classify once per conversation and pin it. A conversation we
  can't recognize (proxy restarted mid-session, `messages.length > 2`, unseen key)
  is passed through untouched rather than re-routed.

## Run

```sh
node services/claude-router/router.js
# listening on http://127.0.0.1:3456 -> https://api.anthropic.com
```

Env overrides: `ROUTER_PORT`, `ROUTER_UPSTREAM`, `ROUTER_CONFIG`, `ROUTER_LOG_FILE`,
`ROUTER_ANTHROPIC_API_KEY`.

## Opt a session in

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:3456 claude
```

## Disable

Just unset the env var — the proxy is **opt-in per session**. Nothing routes
through it unless a session points `ANTHROPIC_BASE_URL` at it, so it is never a
fleet single-point-of-failure.

## Credentials for the classifier call

- If `ROUTER_ANTHROPIC_API_KEY` is set, the classifier call uses it as `x-api-key`.
- Otherwise it **reuses the incoming request's own credential headers verbatim**
  (`authorization`, `x-api-key`, `anthropic-version`, `anthropic-beta`). This works
  for OAuth Claude Code sessions (`Bearer` + `anthropic-beta: oauth-2025-04-20`).

On boot, if a credential is available, the router fires one classifier ping. A
`4xx` prints a loud stderr warning — routing will silently no-op for any request
whose own credentials also fail.

## Config (`config.json`, override path via `ROUTER_CONFIG`)

| key | default | meaning |
|-----|---------|---------|
| `mode` | `"session"` | `session` = classify once, cache sticky; `request` = classify every request |
| `tiers` | haiku / sonnet-5 / opus-4-8 | label → model map (read live, not hardcoded) |
| `pinned` | `[]` | model-name prefixes to pass through untouched |
| `classifierModel` | `claude-haiku-4-5` | model used for the classifier call |
| `classifierTimeoutMs` | `3000` | hard timeout on the classifier call (covers cold TLS handshake + OAuth Haiku latency; harmless in session mode since it fires once per session) |
| `maxHaikuInputTokens` | `150000` | above this (est. bytes/4), the `light` tier is forbidden and upgraded to `standard` |
| `upstream` | `https://api.anthropic.com` | forward target |
| `allowedUpstreamHosts` | `[]` | extra hostnames allowed as `upstream` (see below) |
| `port` | `3456` | listen port |

> **Upstream allowlist (fail-closed).** On startup the router validates
> `new URL(upstream).hostname` against `api.anthropic.com`, `127.0.0.1`,
> `localhost`, plus anything in `allowedUpstreamHosts`. A non-allowlisted host
> **exits the process** — it does not fail open — because a redirected upstream
> would receive your live Anthropic credentials. A non-default upstream logs a
> one-line stderr notice. Invalid `tiers` (any of light/standard/heavy missing or
> empty) and an out-of-range `ROUTER_PORT` are likewise fatal at startup.

> **`mode: "request"` warning.** Per-request classification busts Anthropic's
> model-scoped prompt cache (every model switch cold-restarts it) and usually
> costs **more**, not less. Keep `session` unless you know why you want otherwise.

## Multi-provider routing (offload tiers to other providers)

The real payoff: route cheaper tiers onto a **different provider** — e.g. NVIDIA's
free OpenAI-compatible endpoints — while the client keeps speaking Anthropic. The
router translates the request to `chat.completions`, forwards it, and translates
the response (including streaming SSE) back to Anthropic events.

A **tier value** is either `"provider,model"` or a bare model name (which implies
the `anthropic` provider — full backward compatibility). Providers are declared
under `providers`:

```json
"providers": {
  "anthropic": { "type": "anthropic" },
  "nvidia": { "type": "openai", "base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY" }
}
```

### NVIDIA free-tier quickstart

1. Get an NVIDIA API key and export it: `export NVIDIA_API_KEY=nvapi-…`
2. Point a tier at an NVIDIA model, e.g. in `config.json`:
   ```json
   "tiers": {
     "light": "nvidia,meta/llama-3.3-70b-instruct",
     "standard": "claude-sonnet-5",
     "heavy": "claude-opus-4-8"
   }
   ```
3. Run the router. Light requests now go to NVIDIA (free), the rest to Anthropic.

If `NVIDIA_API_KEY` is unset, the router **warns loudly at startup** and any
request routed to NVIDIA **fails open to an Anthropic passthrough** (logged as
`fallback` with `fallback_reason: "provider-key-missing:nvidia"`) — it never
breaks the request.

### Security invariant (client credentials never cross providers)

When forwarding to a non-Anthropic provider, the router sends **only**
`Authorization: Bearer <env[api_key_env]>`. The client's Anthropic credential
headers (`authorization`, `x-api-key`, `anthropic-*`) are **never** forwarded to a
non-Anthropic provider — a redirected upstream must not be able to harvest them.
Provider `base_url`s are validated at startup as parseable **https** URLs
(loopback exempt for local dev), fail-closed.

## Decision log (`decisions.jsonl`, override via `ROUTER_LOG_FILE`)

One JSON object per line. **Never contains prompt text, message content, or
headers** — only routing metadata:

```json
{"ts":"…","key":"<12-char hash>","original_model":"…","routed_model":"…","provider":"nvidia","label":"heavy","decision":"routed","est_input_tokens":1234,"classifier_ms":210,"cache_hit":false}
```

`decision` is one of: `routed`, `cache-hit`, `already-light`, `pinned`,
`midflight-passthrough`, `malformed-passthrough`, `fallback` (adds
`fallback_reason`, e.g. `provider-key-missing:<name>`). `provider` names the
provider the request was routed to (`anthropic` for all passthrough/fallback).

## Known limitations

- `message_start.model` in the streamed response reports the **routed** model, not
  what the client asked for. Tools that key off it will see the rewritten model.
- The classifier call **consumes your subscription/API quota** (one tiny Haiku
  completion per new conversation).
- `mode: "request"` defeats the prompt cache — see the warning above.
- Token estimate is `bytes/4`, a coarse heuristic used only for the Haiku
  context guard, not for billing.

### Cross-provider translation limitations

- **`thinking` is dropped.** Extended-thinking / reasoning requests are silently
  stripped when translating to a non-Anthropic provider (no equivalent field).
- **Prompt caching does not apply cross-provider.** Anthropic's model-scoped cache
  is meaningless once a tier lands on another provider; session-stickiness still
  avoids thrashing within a session.
- **`count_tokens` is Anthropic-only.** `/v1/messages/count_tokens` always
  passes through to Anthropic; for a session pinned to a non-Anthropic provider it
  is an approximation, not the provider's own tokenizer.
- **Tool fidelity on small models.** `tool_use` / `tool_result` are translated to
  OpenAI `tool_calls` / `role:"tool"` messages, but small open models may emit
  malformed tool-call JSON; unparseable arguments degrade to `input: {}` (logged
  to stderr) rather than failing the response.
- Response `message_start.model` reports the routed `provider/model` string.
