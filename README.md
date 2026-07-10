# claude-router

A local Anthropic Messages API proxy for Claude Code. It classifies a **new**
conversation into a model tier (`light` / `standard` / `heavy`) with one cheap
`claude-haiku-4-5` call, pins that choice for the whole session, rewrites
`body.model`, and forwards to `https://api.anthropic.com`.

It **never breaks a request**: any internal failure (parse error, classifier
timeout, unexpected label) forwards the original bytes unmodified — **fail-open**.

> **Reality check — the light tier yields little to no savings for a *stock* Claude Code
> session today.** Claude Code tailors every request to its **configured** model: betas,
> `output_config.effort`, adaptive `thinking`, and a mid-conversation `role:"system"` message.
> `claude-haiku-4-5` rejects several outright (measured, live API: `role 'system' is not
> supported on this model`; `output_config.effort` → `400`; `adaptive thinking is not
> supported`). The router strips the **params** it can safely reconcile, but it **refuses to
> mutate conversation content** — so a turn carrying a `role:"system"` message is **declined
> pre-flight** (`decision:"capability-mismatch"`) and forwarded on the original model. Routing
> is only sound **between models of comparable capability**; for real savings, tier across
> models that all support the features your client emits. See *Model-param reconciliation & the
> capability gate* below.

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
| `modelCapabilities` | `{ "claude-haiku-4-5": { "effort": false, "adaptiveThinking": false } }` | per-model param support (**measured**). A rewrite target's unsupported params are stripped before forwarding; an **unlisted model is assumed capable** (nothing stripped) and relies on the rewrite-rejected retry |

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

## Model-param reconciliation & the capability gate

**Principle: strip params, never content — and when content is incompatible, decline to
route.** A re-target can hit two kinds of incompatibility, handled by three layers:

**0. Capability gate (pre-flight, content-level).** Some incompatibilities live in conversation
**content**, which must not be mutated (it carries the caller's real instructions). Measured,
live API, `claude-haiku-4-5`:

| content the target rejects | result |
|---|---|
| a mid-conversation `role:"system"` message (`mid-conversation-system` beta) | `role 'system' is not supported on this model` → **declined** |

If `capabilityBlockers` finds such content on a rewrite target, the router does **not** rewrite:
it forwards the caller's original bytes untouched and logs `decision:"capability-mismatch"` with
`capability_blockers` (telemetry root span: `router.capability_blockers`). This is a deliberate
no-route — **not** a fallback, one upstream round-trip, zero waste — and it is not counted
against the fallback health tracker. An unlisted model makes no claims (nothing is declined).

The remaining two layers keep a rewrite from breaking the turn on **model-specific params** the
new target rejects (e.g. `claude-fable-5`'s `output_config.effort` / adaptive thinking on
`claude-haiku-4-5`):

1. **Sanitize (proactive).** Before forwarding a rewritten body, params the target cannot
   accept are stripped, per a **measured** capability table (`config.modelCapabilities`, keyed
   by exact model id). Measured today (live API):

   | param | `claude-haiku-4-5` |
   |---|---|
   | `output_config.effort` | rejected → stripped (drop `output_config` if it empties) |
   | `thinking: {type:'adaptive'}` | rejected → dropped (never downgraded to a budget the caller never chose) |
   | `thinking: {type:'enabled', budget_tokens}` | accepted → preserved |
   | `context_management` `clear_thinking_*` edit | requires thinking → dropped when thinking is removed/absent; if that empties `edits`, `context_management` is dropped too |

   **Dependency cascade.** Stripping a field cascades to whatever *depends* on it: removing
   `thinking` also removes any `context_management` edit whose contract requires thinking (the
   `clear_thinking_*` family, matched by family so a future dated variant is covered) —
   otherwise the body is self-contradictory ("clear the thinking blocks" + "there is no
   thinking") and Anthropic 400s.

   **An unlisted model carries no capability claims — nothing is stripped for it.** Other
   models were rate-limited and could not be measured; guessing would break requests they
   accept. Layer 2 is their net.

2. **Rewrite-rejected retry (reactive).** If a **rewritten** request still 400s (a param we
   had no measured basis to strip), the router retries **once** with the caller's **original
   bytes** (original model + params) and serves that response — the router mutated the request,
   so the router, not the caller, owns the 400. A **non-rewritten** request that 400s is
   **never** retried: a genuine caller 400 reaches the caller unchanged. The event is logged
   (`decision:"fallback"`, `fallback_reason:"rewrite-rejected-400"`) and the session is pinned
   so subsequent turns pass through unmodified (`rewrite-rejected-passthrough`) instead of
   paying the 400+retry double round-trip every turn. Telemetry marks the root span
   `router.rewrite_rejected=true`.

## Telemetry (LangWatch)

The router can emit an **OTLP trace per request** to [LangWatch](https://langwatch.ai) so
you can see, per conversation, which tier each turn was routed to and why. It is
**fire-and-forget**: telemetry never blocks, slows, or breaks a proxied request — every
exporter error is swallowed and nothing is awaited in the request path. **Zero runtime
dependencies** (OTLP/HTTP JSON over the built-in `fetch`).

### Quickstart

```sh
bin/claude-repl
```

Starts the router, waits for it to bind, and launches `claude` pointed at it. The key is
resolved from `LANGWATCH_API_KEY`, or failing that the `Authorization: Bearer …` token in
`~/.claude/settings.json`'s `OTEL_EXPORTER_OTLP_HEADERS`. No key found → it prints a warning
and runs **without** telemetry (routing still works). Pass `--capture-content` to opt into
prompt capture (see below).

> **Ingestion latency.** Traces are typically queryable within **seconds** (measured ~6s)
> under `service.name = claude-router`. Note that **a `200` from the collector does not mean
> the trace is queryable** — the OTLP POST can return `200` with `rejectedSpans: 0` yet the
> span still be minutes from ingestion. If a trace does not appear, check
> <https://status.langwatch.ai>: a degraded LangWatch **Processor** delays ingestion by many
> minutes, and that failure mode looks identical to "instrumentation broken" — rule out a
> Processor degradation before debugging your setup.

### Config & env

`config.json`:

```json
"telemetry": { "enabled": true, "captureContent": false }
```

| var | meaning |
|-----|---------|
| `LANGWATCH_API_KEY` | LangWatch ingestion key (`sk-lw-…`). Its presence enables telemetry. |
| `OTEL_EXPORTER_OTLP_HEADERS` | fallback key source — parsed for `Authorization=Bearer <tok>`. |
| `LANGWATCH_ENDPOINT` / `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP base URL (default `https://app.langwatch.ai/api/otel`); the exporter POSTs to `<base>/v1/traces`. |
| `ROUTER_TELEMETRY=0` | hard off-switch (overrides config). |
| `ROUTER_TELEMETRY_CAPTURE_CONTENT=1` | opt into prompt capture (same as `config.telemetry.captureContent`). |

Telemetry is **disabled** whenever no key resolves, `ROUTER_TELEMETRY=0`, or
`config.telemetry.enabled === false`.

### What a trace contains

One trace per request: a root `claude_router.request` span (the routing decision — tier,
provider, original/routed model, est. input tokens, cache hit, fallback reason, HTTP status)
with child `claude_router.classifier` and `claude_router.upstream` LLM spans when those
calls actually happened. The **root span never carries prompt text.**

### Content capture is off by default (and gated even when on)

By default **no prompt text leaves the process** — only routing metadata. `--capture-content`
(or `captureContent: true`) attaches the classifier's input digest and label to the
classifier span so you can inspect *what* was being routed. Even then, a turn whose text
trips the **secret gate** (API key, token, private key, SSN — the same fail-closed detector
that keeps secrets off third-party *model* providers) has its content **withheld**: that
protection extends to a third-party *observability* backend too. `router.content_captured`
on the root span records, per request, whether content was actually attached — so a
deliberately withheld turn is visible as such.

## Decision log (`decisions.jsonl`, override via `ROUTER_LOG_FILE`)

One JSON object per line. **Never contains prompt text, message content, or
headers** — only routing metadata:

```json
{"ts":"…","key":"<12-char hash>","original_model":"…","routed_model":"…","provider":"nvidia","label":"heavy","decision":"routed","est_input_tokens":1234,"classifier_ms":210,"cache_hit":false}
```

`decision` is one of: `routed`, `cache-hit`, `already-light`, `pinned`,
`midflight-passthrough`, `malformed-passthrough`, `rewrite-rejected-passthrough`
(session pinned to passthrough after a rewrite drew a 400), `capability-mismatch` (declined
pre-flight because the target cannot serve the request's content unchanged; adds
`capability_blockers`), `fallback` (adds `fallback_reason`, e.g. `provider-key-missing:<name>`
or `rewrite-rejected-400`). `provider` names the
provider the request was routed to (`anthropic` for all passthrough/fallback).

## Known limitations

- **The light tier yields little to no savings for a *stock* Claude Code session today** —
  Claude Code tailors each request to its configured model (betas, `output_config.effort`,
  adaptive `thinking`, a mid-conversation `role:"system"` message), and `claude-haiku-4-5`
  rejects several, so those turns are declined pre-flight (`capability-mismatch`) and pass
  through on the original model. Real savings require tiering across models of **comparable
  capability**. See *Model-param reconciliation & the capability gate*.
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

**Status:** the routing brain is validated offline (~82% tier accuracy, heavy-tier detection reliable, deterministic at `temperature: 0`). Free-model *adequacy* and end-to-end answer quality are **not** yet tested. A **fail-closed secret gate** now diverts any request whose text carries a high-confidence credential (API key, token, PEM private key, SSN) to Anthropic before it can reach a third-party provider — the same fail-closed pattern as the modality gate, checked *before* the provider-key check so it holds even with a valid provider key. The scan covers **everything the translator forwards**: the system prompt, message text, `tool_result` text, and — critically — `tool_use.input` (a `Bash` command or `Write`/`Edit` payload is serialized onto the provider wire and is where a real secret is most likely to hide). **Semantic PII** (names, addresses, free-form secrets) is not yet detected and can still route to a free provider on complexity alone.
