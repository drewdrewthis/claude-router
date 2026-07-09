'use strict';
// Pure translation between the Anthropic Messages API and the OpenAI
// chat.completions API. No I/O — every function is unit-testable in isolation.
// The router uses these to route a tier onto an OpenAI-compatible provider
// (e.g. NVIDIA's free endpoints) while the client keeps speaking Anthropic.

const crypto = require('node:crypto');

function genId(prefix) {
  return prefix + '_' + crypto.randomBytes(12).toString('hex');
}

// OpenAI constrains function names to ^[a-zA-Z0-9_-]{1,64}$. Anthropic does not:
// Claude Code routinely emits names like `mcp__<server>__<tool>` that blow past 64
// chars, and an over-long/illegal name makes an OpenAI-compatible provider reject the
// whole request with a 400.
const OPENAI_TOOL_NAME_MAX = 64;
const OPENAI_TOOL_NAME_OK = /^[a-zA-Z0-9_-]{1,64}$/;

// Anthropic tool name -> a legal OpenAI function name.
//
// Pattern borrowed from LiteLLM's `_sanitize_openai_function_tool_name` +
// `openai_tool_name_mapping.py` (PR BerriAI/litellm#27114, still open at time of
// writing): regex-replace illegal chars with `_`, truncate to 64, and keep a
// forward/reverse map so `tool_choice` is rewritten on the request leg and the
// original name is restored on the response leg. (claude-code-router / @musistudio/llms
// does NOT sanitize at all — it forwards `mcp__server__tool` verbatim and relies on the
// provider tolerating it.)
//
// ONE DELIBERATE DEVIATION: LiteLLM disambiguates truncation collisions with a numeric
// suffix (`_1`, `_2`) allocated against the whole tool set, which forces the mapping to
// live in per-request state (a ContextVar). We use an 8-hex-char content hash instead,
// which makes this a PURE FUNCTION OF THE NAME ALONE — so the request leg and the
// response leg each rederive the identical mapping from `body.tools`, and no mutable
// state has to be threaded through the router. Reversal is still by map (a hash is not
// invertible); the hash only supplies uniqueness.
//
// A name that is already legal maps to ITSELF, so the common case is untouched.
// 55 + 1 + 8 == 64 exactly.
//
// Residual: a legal name could in principle equal another name's mangled form. That
// needs a chosen 8-hex-char sha256 prefix collision, so it is not defended against.
function toOpenaiToolName(name) {
  if (typeof name !== 'string' || name === '') return name;
  if (OPENAI_TOOL_NAME_OK.test(name)) return name;
  const digest = crypto.createHash('sha256').update(name).digest('hex').slice(0, 8);
  const safe = name.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, OPENAI_TOOL_NAME_MAX - 9);
  return `${safe}_${digest}`;
}

// Reverse map (openai name -> original anthropic name) for the response leg. Built
// from the request's `tools`, because a client can only ever match a tool_use block
// by the name IT declared — echoing back a mangled name silently breaks tool calls.
// Names that round-trip unchanged are still included; lookup misses fall back to the
// provider-supplied name.
function toolNameMap(tools) {
  const rev = new Map();
  for (const t of Array.isArray(tools) ? tools : []) {
    if (!t || typeof t.name !== 'string' || t.name === '') continue;
    rev.set(toOpenaiToolName(t.name), t.name);
  }
  return rev;
}

function restoreToolName(rev, name) {
  if (!rev || typeof name !== 'string') return name;
  return rev.has(name) ? rev.get(name) : name;
}

// stop-reason table (OpenAI finish_reason -> Anthropic stop_reason).
function mapFinish(reason) {
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    default:
      return 'end_turn';
  }
}

function blocksToText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((b) => (b && typeof b.text === 'string' ? b.text : '')).join('');
  }
  return '';
}

// Like blocksToText, but for a tool_result's content: a non-text block (image, etc.)
// becomes an explicit placeholder instead of collapsing to '' — documenting the loss
// rather than silently dropping it. Kept separate from blocksToText so the system-
// prompt path (which also uses blocksToText) never leaks this placeholder.
function toolResultToText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((b) => (b && typeof b.text === 'string' ? b.text : '[non-text tool result omitted]'))
      .join('');
  }
  return '';
}

// Collapse OpenAI content: all-text -> string; any image -> parts array; empty -> null.
function partsToContent(parts) {
  if (!parts.length) return null;
  if (parts.every((p) => p.type === 'text')) return parts.map((p) => p.text).join('');
  return parts;
}

function translateOneMessage(m, out) {
  const role = m.role;
  if (typeof m.content === 'string') {
    out.push({ role, content: m.content });
    return;
  }
  const blocks = Array.isArray(m.content) ? m.content : [];

  // tool_result blocks become standalone {role:"tool"} messages, emitted BEFORE
  // any sibling text so ordering matches the OpenAI expectation.
  for (const b of blocks) {
    if (b && b.type === 'tool_result') {
      // Lossy: non-text tool_result blocks (images, etc.) have no OpenAI tool-message
      // equivalent, so toolResultToText substitutes a placeholder instead of dropping.
      out.push({ role: 'tool', tool_call_id: b.tool_use_id, content: toolResultToText(b.content) });
    }
  }

  const parts = [];
  const toolCalls = [];
  for (const b of blocks) {
    if (!b) continue;
    if (b.type === 'text') {
      parts.push({ type: 'text', text: b.text });
    } else if (b.type === 'image' && b.source) {
      parts.push({
        type: 'image_url',
        image_url: { url: `data:${b.source.media_type};base64,${b.source.data}` },
      });
    } else if (b.type === 'tool_use') {
      toolCalls.push({
        id: b.id,
        type: 'function',
        // Historical tool_use blocks carry the tool's name too, so they need the same
        // mangle as `tools[]` or the provider sees two different names for one tool.
        function: {
          name: toOpenaiToolName(b.name),
          arguments: JSON.stringify(b.input == null ? {} : b.input),
        },
      });
    } else if (b.type === 'thinking' || b.type === 'redacted_thinking') {
      // Deliberately dropped, not translated. An Anthropic thinking block is only
      // meaningful to Anthropic (it is signed, and `redacted_thinking` is opaque
      // ciphertext); OpenAI has no equivalent. Beyond "no equivalent": replaying
      // reasoning to an OpenAI-compatible reasoning backend re-injects rendered
      // chain-of-thought and drives phrase-repetition loops that compound over long
      // tool chains (BerriAI/litellm#31279 measured 9/15 turns affected with replay,
      // 0/15 without). Called out explicitly so the omission reads as a decision
      // rather than a gap in the if-chain.
      continue;
    }
  }

  if (role === 'assistant') {
    const content = partsToContent(parts);
    if (content === null && toolCalls.length === 0) return; // nothing to say
    const msg = { role: 'assistant', content };
    if (toolCalls.length) msg.tool_calls = toolCalls;
    out.push(msg);
  } else {
    if (parts.length) out.push({ role: 'user', content: partsToContent(parts) });
  }
}

function mapToolChoice(tc) {
  if (tc == null) return undefined;
  if (typeof tc === 'string') {
    if (tc === 'auto') return 'auto';
    if (tc === 'any') return 'required';
    if (tc === 'none') return 'none'; // OpenAI accepts tool_choice:'none'
    return 'auto';
  }
  if (tc.type === 'auto') return 'auto';
  if (tc.type === 'any') return 'required';
  if (tc.type === 'none') return 'none';
  // A forced tool must be named with the SAME mangled name we put in `tools[]`.
  if (tc.type === 'tool' && tc.name) {
    return { type: 'function', function: { name: toOpenaiToolName(tc.name) } };
  }
  return 'auto';
}

// Anthropic Messages request -> OpenAI chat.completions request.
// Anthropic-only fields (thinking, metadata, cache_control anywhere, betas) are
// dropped silently by virtue of not being copied.
function requestAnthropicToOpenai(body, model) {
  const out = { model, messages: [] };

  const sysText = blocksToText(body.system);
  if (sysText) out.messages.push({ role: 'system', content: sysText });

  for (const m of Array.isArray(body.messages) ? body.messages : []) {
    translateOneMessage(m, out.messages);
  }

  if (body.max_tokens != null) out.max_tokens = body.max_tokens;
  if (body.stop_sequences != null) out.stop = body.stop_sequences;
  if (body.temperature != null) out.temperature = body.temperature;
  if (body.top_p != null) out.top_p = body.top_p;

  if (Array.isArray(body.tools) && body.tools.length) {
    out.tools = body.tools.map((t) => ({
      type: 'function',
      function: {
        name: toOpenaiToolName(t.name),
        description: t.description,
        parameters: t.input_schema,
      },
    }));
  }
  const tc = mapToolChoice(body.tool_choice);
  if (tc !== undefined) out.tool_choice = tc;

  if (body.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// Fresh (non-cached) input tokens.
//
// The two APIs disagree about what the headline input count MEANS:
//   OpenAI    `usage.prompt_tokens` is the FULL prompt, cache hits INCLUDED, with the
//             cached subset broken out as `prompt_tokens_details.cached_tokens`.
//   Anthropic `usage.input_tokens` is "the number of input tokens which were NOT read
//             from or used to create a cache", i.e. cache reads AND cache writes are
//             both excluded, and total = input + cache_read + cache_creation.
//
// So a faithful translation subtracts BOTH counters, not just the read. (Subtracting
// only the read over-reports fresh input on any turn that writes to cache.) This is
// LiteLLM's `_translate_openai_usage_to_anthropic_usage_delta`:
//     input = max(prompt_tokens - cache_read - cache_creation, 0)
// Absent all cache fields, `prompt_tokens` passes through unchanged.
function inputTokensFromUsage(usage) {
  if (!usage) return 0;
  const prompt = usage.prompt_tokens ?? 0;
  return Math.max(0, prompt - cacheReadTokens(usage) - cacheCreationTokens(usage));
}

// Cache-read tokens: `prompt_tokens_details.cached_tokens` (OpenAI's spelling) with
// `cache_read_input_tokens` (the Anthropic-style alias some gateways emit) as fallback.
// An explicit typeof test, not `||`, so a reported 0 is honoured rather than skipped.
function cacheReadTokens(usage) {
  const u = usage || {};
  const details = u.prompt_tokens_details || {};
  if (typeof details.cached_tokens === 'number') return details.cached_tokens;
  if (typeof u.cache_read_input_tokens === 'number') return u.cache_read_input_tokens;
  return 0;
}

// Cache-WRITE tokens. Only the Anthropic-style spelling exists — OpenAI has no
// cache-creation counter — so this is 0 for a plain OpenAI-compatible provider.
function cacheCreationTokens(usage) {
  const u = usage || {};
  return typeof u.cache_creation_input_tokens === 'number' ? u.cache_creation_input_tokens : 0;
}

// OpenAI usage -> Anthropic usage. `input_tokens` excludes both cache counters (see
// inputTokensFromUsage); the counters are then surfaced separately so a client's cost
// accounting can see them at all — previously they were computed, used for the
// subtraction, and thrown away.
//
// The cache_* keys are emitted ONLY when the provider actually reported them: most
// OpenAI-compatible providers have no prompt cache, and synthesizing `0` there would
// assert a cache-miss we never observed.
function usageToAnthropic(usage) {
  const u = usage || {};
  const out = {
    input_tokens: inputTokensFromUsage(u),
    output_tokens: u.completion_tokens ?? 0,
  };
  const read = cacheReadTokens(u);
  if (read > 0) out.cache_read_input_tokens = read;
  const created = cacheCreationTokens(u);
  if (created > 0) out.cache_creation_input_tokens = created;
  return out;
}

// A provider that emits tool_calls but reports finish_reason:"stop" (NVIDIA does
// exactly this — see docs/live-nvidia-toolcall-proof.txt [TOOL-4]) would map to
// stop_reason:"end_turn", and an Anthropic client then ENDS THE TURN instead of
// executing the tool. Content wins over finish_reason: if we emitted tool_use blocks,
// the stop reason is tool_use. `max_tokens` is preserved, because a truncated tool
// call is a truncation first.
function stopReasonFor(finishReason, hasToolUse) {
  const mapped = mapFinish(finishReason);
  if (hasToolUse && mapped === 'end_turn') return 'tool_use';
  return mapped;
}

// OpenAI chat.completions (non-stream) response -> Anthropic Messages response.
// `reportModel` is the routed "provider/model" string (NOT what the client asked).
// `toolNames` is the reverse name map from toolNameMap(requestBody.tools).
function responseOpenaiToAnthropic(resp, reportModel, toolNames) {
  const choice = (resp.choices && resp.choices[0]) || {};
  const msg = choice.message || {};
  const content = [];

  // `msg.reasoning_content` (DeepSeek-R1, QwQ, and other reasoning models served over
  // OpenAI-compatible endpoints, incl. NVIDIA NIM) is deliberately DROPPED rather than
  // re-emitted as an Anthropic `thinking` block.
  //
  // This is where we KNOWINGLY DIVERGE from the mature proxies: LiteLLM emits
  // `signature: ""` and @musistudio/llms emits `signature: undefined`. But a thinking
  // block's signature is what Anthropic decrypts to verify the block came from Claude,
  // and Anthropic rejects a modified one with `400 invalid_request_error` ("`thinking`
  // or `redacted_thinking` blocks in the latest assistant message cannot be modified").
  // The client echoes the assistant turn back on the NEXT request — so the first turn
  // that fails open to Anthropic carries our forged block straight into that 400.
  // Losing the reasoning text is strictly safer than poisoning the conversation.
  if (typeof msg.content === 'string' && msg.content.length) {
    content.push({ type: 'text', text: msg.content });
  }
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      let input = {};
      try {
        input = JSON.parse((tc.function && tc.function.arguments) || '{}');
      } catch {
        input = {};
        console.error('[claude-router] tool_call arguments were not valid JSON; using {}');
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: restoreToolName(toolNames, tc.function && tc.function.name),
        input,
      });
    }
  }
  if (!content.length) content.push({ type: 'text', text: '' });

  const hasToolUse = content.some((b) => b.type === 'tool_use');
  return {
    id: resp.id || genId('msg'),
    type: 'message',
    role: 'assistant',
    model: reportModel,
    content,
    stop_reason: stopReasonFor(choice.finish_reason, hasToolUse),
    stop_sequence: null,
    usage: usageToAnthropic(resp.usage),
  };
}

function sse(type, data) {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

// OpenAI SSE chunk stream -> Anthropic SSE event stream, as an async generator.
// `rawChunks` is any async iterable of string|Buffer SSE bytes. Malformed chunks
// are skipped; if the source dies mid-way we still emit a clean close (never hang).
//
// Each OpenAI `tool_calls[].index` maps to a STABLE Anthropic content-block index via
// `toolIndexToBlock`. That is what OpenAI's index is FOR: a provider may interleave
// deltas across parallel tool calls (tool[0] args -> tool[1] args -> tool[0] args
// again), and each fragment must reassemble into the block for THAT index. A sibling
// tool index appearing must NOT close a still-streaming block — so we never close
// blocks eagerly; every opened block is closed once, at stream end, in ascending
// index order. Text keeps its own dedicated block. (Reference: @musistudio/llms
// toolCallIndexToContentBlockIndex, LiteLLM.)
async function* streamOpenaiToAnthropic(rawChunks, reportModel, toolNames) {
  let buffer = '';
  let started = false;
  const messageId = genId('msg');
  let nextBlockIndex = 0; // monotonically allocates Anthropic content-block indices
  let textBlockIndex = null; // the block all text deltas route to, once opened
  const toolIndexToBlock = new Map(); // OAI tool `index` -> Anthropic block index
  const openBlocks = []; // every block opened, in allocation (== ascending) order
  let lastToolKey = null; // fallback target for a tool delta that omits `index`
  let finishReason = null; // raw OpenAI finish_reason; mapped once, at close
  let usage = null;

  function* ensureStart() {
    if (started) return;
    started = true;
    yield sse('message_start', {
      type: 'message_start',
      message: {
        id: messageId,
        type: 'message',
        role: 'assistant',
        model: reportModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  }
  function* ensureTextBlock() {
    if (textBlockIndex !== null) return;
    textBlockIndex = nextBlockIndex++;
    openBlocks.push(textBlockIndex);
    yield sse('content_block_start', {
      type: 'content_block_start',
      index: textBlockIndex,
      content_block: { type: 'text', text: '' },
    });
  }
  // Returns the Anthropic block index for this OAI tool key, opening the block (using
  // the id/name from the FIRST delta at that key — OpenAI sends id+name only on the
  // first delta per index, args-only after) the first time the key is seen.
  function* ensureToolBlock(key, id, name) {
    if (toolIndexToBlock.has(key)) return toolIndexToBlock.get(key);
    const bi = nextBlockIndex++;
    toolIndexToBlock.set(key, bi);
    openBlocks.push(bi);
    yield sse('content_block_start', {
      type: 'content_block_start',
      index: bi,
      content_block: {
        type: 'tool_use',
        id: id || genId('toolu'),
        // Hand the client back the name IT declared, not our mangled wire name.
        name: restoreToolName(toolNames, name) || '',
        input: {},
      },
    });
    return bi;
  }

  try {
    for await (const raw of rawChunks) {
      buffer += typeof raw === 'string' ? raw : Buffer.from(raw).toString('utf8');
      let nl;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '' || payload === '[DONE]') continue;

        let chunk;
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue; // malformed chunk: skip, keep the stream alive
        }

        yield* ensureStart();
        if (chunk.usage) usage = chunk.usage;

        const choice = (chunk.choices && chunk.choices[0]) || {};
        const delta = choice.delta || {};

        // `delta.reasoning_content` is intentionally NOT consumed — see the rationale
        // on responseOpenaiToAnthropic. Emitting it as text would splice raw chain-of-
        // thought into the visible answer; emitting it as a `thinking` block would need
        // a signature we cannot mint.
        if (typeof delta.content === 'string' && delta.content.length) {
          yield* ensureTextBlock();
          yield sse('content_block_delta', {
            type: 'content_block_delta',
            index: textBlockIndex,
            delta: { type: 'text_delta', text: delta.content },
          });
        }

        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index;
            // OpenAI always sends `index`; if a provider omits it, continue the
            // last-seen tool (or a single synthetic key) rather than corrupt state.
            const key = idx != null ? idx : lastToolKey != null ? lastToolKey : '__tool0__';
            lastToolKey = key;
            const bi = yield* ensureToolBlock(key, tc.id, tc.function && tc.function.name);
            const args = tc.function && tc.function.arguments;
            if (typeof args === 'string' && args.length) {
              yield sse('content_block_delta', {
                type: 'content_block_delta',
                index: bi, // route THIS delta to the block mapped for its OAI index
                delta: { type: 'input_json_delta', partial_json: args },
              });
            }
          }
        }

        if (choice.finish_reason) finishReason = choice.finish_reason;
      }
    }
  } catch {
    // Source stream died mid-way — fall through to a graceful close below.
  }

  // Always emit a valid, terminated Anthropic stream. Close EVERY opened block, once,
  // in ascending block-index order, before the terminal message_delta.
  yield* ensureStart();
  for (const bi of openBlocks.slice().sort((a, b) => a - b)) {
    yield sse('content_block_stop', { type: 'content_block_stop', index: bi });
  }
  yield sse('message_delta', {
    type: 'message_delta',
    // Same content-wins-over-finish_reason rule as the non-stream path: if we opened
    // any tool_use block, the turn ended to call a tool, whatever the provider said.
    delta: { stop_reason: stopReasonFor(finishReason, toolIndexToBlock.size > 0), stop_sequence: null },
    // input_tokens is only known once OpenAI's final usage chunk arrives (it is not
    // available at message_start); Anthropic clients accept it here in message_delta.
    usage: usageToAnthropic(usage),
  });
  yield sse('message_stop', { type: 'message_stop' });
}

// A provider that HONORED stream:true but replied with a single non-SSE JSON body
// would otherwise translate to an empty Anthropic stream. Re-envelope that one
// completion as a valid single-shot Anthropic event stream so the streaming client
// still gets a well-formed, non-empty stream. `openaiJson` may be null (unparseable
// body) — we then emit a valid, terminated EMPTY stream rather than hang.
function* singleShotSSE(openaiJson, reportModel, toolNames) {
  let msg = null;
  try {
    if (openaiJson) msg = responseOpenaiToAnthropic(openaiJson, reportModel, toolNames);
  } catch {
    msg = null; // defensive: never throw out of the stream path
  }
  const messageId = (msg && msg.id) || genId('msg');
  const blocks = msg && Array.isArray(msg.content) ? msg.content : [];
  const stopReason = (msg && msg.stop_reason) || 'end_turn';
  const usage = (msg && msg.usage) || { input_tokens: 0, output_tokens: 0 };

  yield sse('message_start', {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      model: reportModel,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  for (let index = 0; index < blocks.length; index++) {
    const block = blocks[index];
    if (block && block.type === 'text') {
      yield sse('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      });
      yield sse('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: block.text || '' },
      });
      yield sse('content_block_stop', { type: 'content_block_stop', index });
    } else if (block && block.type === 'tool_use') {
      yield sse('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: block.id || genId('toolu'), name: block.name || '', input: {} },
      });
      yield sse('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input == null ? {} : block.input) },
      });
      yield sse('content_block_stop', { type: 'content_block_stop', index });
    }
  }

  yield sse('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    // Already Anthropic-shaped (usageToAnthropic ran inside responseOpenaiToAnthropic),
    // so carry any cache_* counters through rather than flattening to two fields.
    usage: { input_tokens: 0, output_tokens: 0, ...usage },
  });
  yield sse('message_stop', { type: 'message_stop' });
}

module.exports = {
  requestAnthropicToOpenai,
  responseOpenaiToAnthropic,
  streamOpenaiToAnthropic,
  singleShotSSE,
  toolResultToText,
  toOpenaiToolName,
  toolNameMap,
  usageToAnthropic,
  stopReasonFor,
  mapFinish,
  mapToolChoice,
  genId,
  OPENAI_TOOL_NAME_MAX,
};
