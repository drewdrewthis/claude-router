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
| `providerTimeoutMs` | `30000` | hard timeout on the OpenAI-compatible provider fetch. For a **streaming** request it bounds only time-to-response-headers, so a long streaming body is never clipped. For a **non-stream** request it *also* bounds the response-body read, so a provider that sends headers then stalls its body still **fails open to Anthropic**. On timeout or provider error (before any client bytes stream) the request fails open |
| `upstreamTimeoutMs` | `240000` | hard timeout on the **Anthropic passthrough** fetch. Bounds only time-to-response-headers, then unbinds — a long SSE body is never clipped. On expiry the client gets a clean `502`. **Ceiling:** undici (Node's `fetch` engine) enforces its own `headersTimeout` of `300000` ms and stdlib-only code cannot raise it, so any value ≥ `300000` never fires and is warned about at startup. The default sits below it deliberately. The body-idle half of the guarantee is undici's `bodyTimeout` (also `300000` ms, inter-chunk — the nginx `proxy_read_timeout` analogue), so the router does not hand-roll one |
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

- **`thinking` blocks are dropped when routing to a non-Anthropic provider, in both
  directions.** On the request leg neither the `thinking` parameter nor any
  `thinking` / `redacted_thinking` blocks in assistant history are forwarded. On the
  response leg a reasoning model's `reasoning_content` (DeepSeek-R1, QwQ — including
  over NVIDIA NIM) is **not** re-emitted as an Anthropic `thinking` block. That block
  carries a `signature` only Anthropic can mint; a fabricated one is echoed back by the
  client on the next turn, and Anthropic rejects a modified thinking block with
  `400 invalid_request_error` — which would break the very next turn that fails open.
  Losing the reasoning text is strictly safer than poisoning the conversation.
  (LiteLLM and claude-code-router both synthesize `signature: ""`/`undefined` here;
  we deliberately do not.)
- **Long tool names are mangled on the wire and restored on the way back.** OpenAI
  enforces `^[a-zA-Z0-9_-]{1,64}$` on function names, while Claude Code routinely emits
  `mcp__server__some_long_tool`. Names over the limit (or with illegal characters) are
  sanitized, truncated, and given an 8-hex-char content-hash suffix — applied
  consistently to `tools[]`, `tool_choice`, and historical `tool_use` blocks — then
  mapped back to the name the client declared on the response leg (non-stream and
  streaming). A name that is already legal is passed through byte-identical.
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
- **`stop_reason` follows the CONTENT, not the provider's `finish_reason`.** A provider
  that emits `tool_calls` alongside `finish_reason: "stop"` (NVIDIA does exactly this —
  see `docs/live-nvidia-toolcall-proof.txt` `[TOOL-4]`) would otherwise map to
  `end_turn`, and the client would end the turn instead of executing the tool. If any
  `tool_use` block was emitted, `stop_reason` is `tool_use`. `max_tokens` still wins,
  because a truncated tool call is a truncation first.
- **Cache-token accounting is translated, not passed through.** OpenAI's
  `prompt_tokens` INCLUDES cache hits; Anthropic's `input_tokens` excludes tokens both
  read from *and* used to create a cache. The router reports
  `input_tokens = max(prompt_tokens - cache_read - cache_creation, 0)` and surfaces
  `cache_read_input_tokens` / `cache_creation_input_tokens` when (and only when) the
  provider reports them.
- Response `message_start.model` reports the routed `provider/model` string.
- **Runtime fail-open to Anthropic.** If a provider fetch throws, times out
  (`providerTimeoutMs` — which bounds time-to-response-headers for a streaming request
  and also the response-body read for a non-stream request, so long streams are never
  clipped yet a stalled non-stream body still aborts), or returns a non-2xx **before
  any response bytes are streamed**, the request is transparently re-served through
  Anthropic with the original body/model. A second decision-log line records it
  (`decision:"fallback"` with `fallback_reason:"provider-http-<status>"`,
  `"provider-fetch:<name>"`, or `"provider-body:<name>"`, `provider:"anthropic"`). Once
  streaming has begun there is nothing to fall back to — the stream is cleanly
  terminated instead.
- **Non-text `tool_result` content is placeholdered, not dropped.** Image/other
  non-text blocks inside a `tool_result` have no OpenAI tool-message equivalent, so
  they are replaced by the literal `[non-text tool result omitted]` (text blocks in
  the same result are preserved).
- **`tool_choice: "none"` is honored.** Anthropic `none` (the string or
  `{ "type": "none" }`) maps to OpenAI `tool_choice: "none"`.
- **Non-SSE provider stream responses are re-enveloped.** If a provider ignores
  `stream:true` and returns a single JSON body, it is buffered and re-emitted as a
  single-shot Anthropic event stream, so a streaming client still receives a valid,
  non-empty stream (an unparseable body yields a valid, empty terminated stream).

## Evaluation

An offline routing eval lives in [`eval/`](eval/): it drives the shipped `decide()` / `classify()` product code over a small labeled suite (light / standard / heavy plus modality and privacy cases) at the **decision level only** — no provider is ever contacted — authenticating the classifier with the local Max OAuth token. Run it with `node eval/stage1-routing.js`.

**Status:** the routing brain is validated offline (~82% tier accuracy, heavy-tier detection reliable, deterministic at `temperature: 0`). Free-model *adequacy* and end-to-end answer quality are **not** yet tested. A **fail-closed secret gate** now diverts any request whose text carries a high-confidence credential (API key, token, PEM private key, SSN) to Anthropic before it can reach a third-party provider — the same fail-closed pattern as the modality gate, checked *before* the provider-key check so it holds even with a valid provider key. **Semantic PII** (names, addresses, free-form secrets) is not yet detected and can still route to a free provider on complexity alone.
