'use strict';
// Pure translation between the Anthropic Messages API and the OpenAI
// chat.completions API. No I/O — every function is unit-testable in isolation.
// The router uses these to route a tier onto an OpenAI-compatible provider
// (e.g. NVIDIA's free endpoints) while the client keeps speaking Anthropic.

const crypto = require('node:crypto');

function genId(prefix) {
  return prefix + '_' + crypto.randomBytes(12).toString('hex');
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
        function: { name: b.name, arguments: JSON.stringify(b.input == null ? {} : b.input) },
      });
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
  if (tc.type === 'tool' && tc.name) return { type: 'function', function: { name: tc.name } };
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
      function: { name: t.name, description: t.description, parameters: t.input_schema },
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

// Fresh (non-cached) input tokens. OpenAI (and LiteLLM-normalized responses) report
// `prompt_tokens` as the FULL prompt INCLUDING any cache-read tokens; subtract the
// cache-read portion so a cache hit is not re-counted as fresh input (this mirrors
// Anthropic's own `input_tokens`, which excludes cache reads). The read count comes
// from `prompt_tokens_details.cached_tokens` (OpenAI) or `cache_read_input_tokens`
// (Anthropic-style alias). `cache_creation_input_tokens` is NOT subtracted — those are
// genuinely fresh input being written to cache. Absent all cache fields, `prompt_tokens`
// passes through unchanged.
function inputTokensFromUsage(usage) {
  if (!usage) return 0;
  const prompt = usage.prompt_tokens ?? 0;
  const details = usage.prompt_tokens_details || {};
  const cachedRead =
    (typeof details.cached_tokens === 'number' && details.cached_tokens) ||
    (typeof usage.cache_read_input_tokens === 'number' && usage.cache_read_input_tokens) ||
    0;
  return Math.max(0, prompt - cachedRead);
}

// OpenAI chat.completions (non-stream) response -> Anthropic Messages response.
// `reportModel` is the routed "provider/model" string (NOT what the client asked).
function responseOpenaiToAnthropic(resp, reportModel) {
  const choice = (resp.choices && resp.choices[0]) || {};
  const msg = choice.message || {};
  const content = [];

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
      content.push({ type: 'tool_use', id: tc.id, name: tc.function && tc.function.name, input });
    }
  }
  if (!content.length) content.push({ type: 'text', text: '' });

  const usage = resp.usage || {};
  return {
    id: resp.id || genId('msg'),
    type: 'message',
    role: 'assistant',
    model: reportModel,
    content,
    stop_reason: mapFinish(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: inputTokensFromUsage(usage),
      output_tokens: usage.completion_tokens ?? 0,
    },
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
async function* streamOpenaiToAnthropic(rawChunks, reportModel) {
  let buffer = '';
  let started = false;
  const messageId = genId('msg');
  let nextBlockIndex = 0; // monotonically allocates Anthropic content-block indices
  let textBlockIndex = null; // the block all text deltas route to, once opened
  const toolIndexToBlock = new Map(); // OAI tool `index` -> Anthropic block index
  const openBlocks = []; // every block opened, in allocation (== ascending) order
  let lastToolKey = null; // fallback target for a tool delta that omits `index`
  let stopReason = 'end_turn';
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
      content_block: { type: 'tool_use', id: id || genId('toolu'), name: name || '', input: {} },
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

        if (choice.finish_reason) stopReason = mapFinish(choice.finish_reason);
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
    delta: { stop_reason: stopReason, stop_sequence: null },
    // input_tokens is only known once OpenAI's final usage chunk arrives (it is not
    // available at message_start); Anthropic clients accept it here in message_delta.
    usage: {
      input_tokens: usage ? inputTokensFromUsage(usage) : 0,
      output_tokens: usage ? usage.completion_tokens ?? 0 : 0,
    },
  });
  yield sse('message_stop', { type: 'message_stop' });
}

// A provider that HONORED stream:true but replied with a single non-SSE JSON body
// would otherwise translate to an empty Anthropic stream. Re-envelope that one
// completion as a valid single-shot Anthropic event stream so the streaming client
// still gets a well-formed, non-empty stream. `openaiJson` may be null (unparseable
// body) — we then emit a valid, terminated EMPTY stream rather than hang.
function* singleShotSSE(openaiJson, reportModel) {
  let msg = null;
  try {
    if (openaiJson) msg = responseOpenaiToAnthropic(openaiJson, reportModel);
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
    usage: { input_tokens: usage.input_tokens ?? 0, output_tokens: usage.output_tokens ?? 0 },
  });
  yield sse('message_stop', { type: 'message_stop' });
}

module.exports = {
  requestAnthropicToOpenai,
  responseOpenaiToAnthropic,
  streamOpenaiToAnthropic,
  singleShotSSE,
  toolResultToText,
  mapFinish,
  mapToolChoice,
  genId,
};
