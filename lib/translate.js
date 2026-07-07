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
      out.push({ role: 'tool', tool_call_id: b.tool_use_id, content: blocksToText(b.content) });
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
    return 'auto';
  }
  if (tc.type === 'auto') return 'auto';
  if (tc.type === 'any') return 'required';
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
      input_tokens: usage.prompt_tokens ?? 0,
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
async function* streamOpenaiToAnthropic(rawChunks, reportModel) {
  let buffer = '';
  let started = false;
  const messageId = genId('msg');
  let blockIndex = -1;
  let openType = null; // 'text' | 'tool' | null
  let currentToolOaiIndex = null;
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
  function* closeBlock() {
    if (openType !== null) {
      yield sse('content_block_stop', { type: 'content_block_stop', index: blockIndex });
      openType = null;
    }
  }
  function* startText() {
    yield* closeBlock();
    blockIndex++;
    openType = 'text';
    yield sse('content_block_start', {
      type: 'content_block_start',
      index: blockIndex,
      content_block: { type: 'text', text: '' },
    });
  }
  function* startTool(id, name) {
    yield* closeBlock();
    blockIndex++;
    openType = 'tool';
    yield sse('content_block_start', {
      type: 'content_block_start',
      index: blockIndex,
      content_block: { type: 'tool_use', id: id || genId('toolu'), name: name || '', input: {} },
    });
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
          if (openType !== 'text') yield* startText();
          yield sse('content_block_delta', {
            type: 'content_block_delta',
            index: blockIndex,
            delta: { type: 'text_delta', text: delta.content },
          });
        }

        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index;
            const isNew =
              tc.id != null || openType !== 'tool' || (idx != null && idx !== currentToolOaiIndex);
            if (isNew) {
              yield* startTool(tc.id, tc.function && tc.function.name);
              if (idx != null) currentToolOaiIndex = idx;
            }
            const args = tc.function && tc.function.arguments;
            if (typeof args === 'string' && args.length) {
              yield sse('content_block_delta', {
                type: 'content_block_delta',
                index: blockIndex,
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

  // Always emit a valid, terminated Anthropic stream.
  yield* ensureStart();
  yield* closeBlock();
  yield sse('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: usage ? usage.completion_tokens ?? 0 : 0 },
  });
  yield sse('message_stop', { type: 'message_stop' });
}

module.exports = {
  requestAnthropicToOpenai,
  responseOpenaiToAnthropic,
  streamOpenaiToAnthropic,
  mapFinish,
  mapToolChoice,
  genId,
};
