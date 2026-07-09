'use strict';
const test = require('node:test');
const assert = require('node:assert');
const {
  requestAnthropicToOpenai,
  responseOpenaiToAnthropic,
  streamOpenaiToAnthropic,
  singleShotSSE,
  toolResultToText,
  mapFinish,
  mapToolChoice,
} = require('../lib/translate');

// ---------- request mapping ----------

test('request: system + text + tool_use/tool_result round trip, anthropic-only fields dropped', () => {
  const anthropic = {
    model: 'claude-3-5-sonnet',
    system: 'you are helpful',
    max_tokens: 100,
    stop_sequences: ['STOP'],
    stream: false,
    messages: [
      { role: 'user', content: 'hi' },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'let me check' },
          { type: 'tool_use', id: 'tu1', name: 'calc', input: { a: 1 } },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'tu1', content: '42' },
          { type: 'text', text: 'thanks' },
        ],
      },
    ],
    tools: [{ name: 'calc', description: 'does math', input_schema: { type: 'object' } }],
    tool_choice: { type: 'auto' },
    thinking: { type: 'enabled', budget_tokens: 1000 },
    metadata: { user_id: 'u1' },
  };
  const out = requestAnthropicToOpenai(anthropic, 'target-model');
  assert.deepStrictEqual(out, {
    model: 'target-model',
    messages: [
      { role: 'system', content: 'you are helpful' },
      { role: 'user', content: 'hi' },
      {
        role: 'assistant',
        content: 'let me check',
        tool_calls: [{ id: 'tu1', type: 'function', function: { name: 'calc', arguments: '{"a":1}' } }],
      },
      { role: 'tool', tool_call_id: 'tu1', content: '42' }, // emitted before sibling text
      { role: 'user', content: 'thanks' },
    ],
    max_tokens: 100,
    stop: ['STOP'],
    tools: [{ type: 'function', function: { name: 'calc', description: 'does math', parameters: { type: 'object' } } }],
    tool_choice: 'auto',
  });
  assert.ok(!('thinking' in out) && !('metadata' in out) && !('stream' in out));
});

test('request: image block becomes an image_url data URI content part', () => {
  const out = requestAnthropicToOpenai(
    {
      model: 'm',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAA' } },
            { type: 'text', text: 'what is this' },
          ],
        },
      ],
    },
    'm'
  );
  assert.deepStrictEqual(out.messages, [
    {
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
        { type: 'text', text: 'what is this' },
      ],
    },
  ]);
});

test('request: tool_choice variants', () => {
  assert.strictEqual(mapToolChoice('auto'), 'auto');
  assert.strictEqual(mapToolChoice({ type: 'auto' }), 'auto');
  assert.strictEqual(mapToolChoice({ type: 'any' }), 'required');
  assert.deepStrictEqual(mapToolChoice({ type: 'tool', name: 'search' }), {
    type: 'function',
    function: { name: 'search' },
  });
});

test('request: stream sets stream + stream_options.include_usage', () => {
  const out = requestAnthropicToOpenai(
    { model: 'm', stream: true, messages: [{ role: 'user', content: 'go' }] },
    'm'
  );
  assert.strictEqual(out.stream, true);
  assert.deepStrictEqual(out.stream_options, { include_usage: true });
});

// ---------- response mapping ----------

test('response: text + tool_calls -> anthropic content blocks, finish_reason + usage mapped', () => {
  const out = responseOpenaiToAnthropic(
    {
      id: 'cmpl-1',
      choices: [
        {
          finish_reason: 'tool_calls',
          message: {
            role: 'assistant',
            content: 'here',
            tool_calls: [
              { id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"q":"x"}' } },
            ],
          },
        },
      ],
      usage: { prompt_tokens: 12, completion_tokens: 5 },
    },
    'nvidia/meta-llama'
  );
  assert.strictEqual(out.type, 'message');
  assert.strictEqual(out.role, 'assistant');
  assert.strictEqual(out.model, 'nvidia/meta-llama');
  assert.strictEqual(out.id, 'cmpl-1');
  assert.deepStrictEqual(out.content, [
    { type: 'text', text: 'here' },
    { type: 'tool_use', id: 'call_1', name: 'lookup', input: { q: 'x' } },
  ]);
  assert.strictEqual(out.stop_reason, 'tool_use');
  assert.deepStrictEqual(out.usage, { input_tokens: 12, output_tokens: 5 });
});

test('response: finish_reason table', () => {
  assert.strictEqual(mapFinish('stop'), 'end_turn');
  assert.strictEqual(mapFinish('length'), 'max_tokens');
  assert.strictEqual(mapFinish('tool_calls'), 'tool_use');
  assert.strictEqual(mapFinish('something-else'), 'end_turn');
});

test('response: unparseable tool arguments degrade to input {}', () => {
  const out = responseOpenaiToAnthropic(
    {
      choices: [
        {
          finish_reason: 'tool_calls',
          message: { tool_calls: [{ id: 'c1', function: { name: 'f', arguments: '{bad json' } }] },
        },
      ],
    },
    'p/m'
  );
  const toolBlock = out.content.find((b) => b.type === 'tool_use');
  assert.deepStrictEqual(toolBlock.input, {});
});

// ---------- streaming translation ----------

async function* fromLines(lines) {
  for (const l of lines) yield l;
}
async function collect(gen) {
  const out = [];
  for await (const s of gen) out.push(s);
  return out;
}
function eventTypes(events) {
  return events.map((e) => e.match(/event: (\w+)/)[1]);
}
function dataOf(event) {
  return JSON.parse(event.match(/data: (.*)\n\n$/s)[1]);
}

test('stream: text deltas -> message_start, block start/deltas/stop, message_delta, message_stop', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { role: 'assistant', content: '' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { content: 'Hel' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { content: 'lo' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`,
    'data: [DONE]\n\n',
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  assert.deepStrictEqual(eventTypes(events), [
    'message_start',
    'content_block_start',
    'content_block_delta',
    'content_block_delta',
    'content_block_stop',
    'message_delta',
    'message_stop',
  ]);
  assert.strictEqual(dataOf(events[0]).message.model, 'p/m');
  assert.strictEqual(dataOf(events[2]).delta.text, 'Hel');
  assert.strictEqual(dataOf(events[5]).delta.stop_reason, 'end_turn');
});

test('stream: tool_call fragments -> tool_use block with input_json_delta partials', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { role: 'assistant' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"loc' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: 'ation":"NYC"}' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })}\n\n`,
    'data: [DONE]\n\n',
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  assert.deepStrictEqual(eventTypes(events), [
    'message_start',
    'content_block_start',
    'content_block_delta',
    'content_block_delta',
    'content_block_stop',
    'message_delta',
    'message_stop',
  ]);
  const startBlock = dataOf(events[1]).content_block;
  assert.strictEqual(startBlock.type, 'tool_use');
  assert.strictEqual(startBlock.id, 'call_1');
  assert.strictEqual(startBlock.name, 'get_weather');
  assert.strictEqual(dataOf(events[2]).delta.partial_json, '{"loc');
  assert.strictEqual(dataOf(events[3]).delta.partial_json, 'ation":"NYC"}');
  assert.strictEqual(dataOf(events[5]).delta.stop_reason, 'tool_use');
});

test('stream: malformed chunk is skipped, stream stays alive', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { content: 'a' } }] })}\n\n`,
    'data: {this is not valid json\n\n',
    `data: ${JSON.stringify({ choices: [{ delta: { content: 'b' } }] })}\n\n`,
    'data: [DONE]\n\n',
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  const types = eventTypes(events);
  assert.strictEqual(types[0], 'message_start');
  assert.strictEqual(types[types.length - 1], 'message_stop');
  const deltas = events.filter((e) => /content_block_delta/.test(e)).map((e) => dataOf(e).delta.text);
  assert.deepStrictEqual(deltas, ['a', 'b']); // malformed dropped, both real deltas survive
});

test('stream: source dying mid-way still terminates the anthropic stream', async () => {
  async function* dies() {
    yield `data: ${JSON.stringify({ choices: [{ delta: { content: 'partial' } }] })}\n\n`;
    throw new Error('upstream socket died');
  }
  const events = await collect(streamOpenaiToAnthropic(dies(), 'p/m'));
  const types = eventTypes(events);
  assert.strictEqual(types[types.length - 2], 'message_delta');
  assert.strictEqual(types[types.length - 1], 'message_stop');
  assert.strictEqual(dataOf(events[events.length - 2]).delta.stop_reason, 'end_turn');
});

test('stream: empty source still emits a valid, terminated envelope', async () => {
  const events = await collect(streamOpenaiToAnthropic(fromLines([]), 'p/m'));
  assert.deepStrictEqual(eventTypes(events), ['message_start', 'message_delta', 'message_stop']);
});

// ---------- hardening: tool_choice none, streaming input_tokens, tool_result
//            placeholder, single-shot re-envelope ----------

test('request: tool_choice none maps to OpenAI none (string and object forms)', () => {
  assert.strictEqual(mapToolChoice('none'), 'none');
  assert.strictEqual(mapToolChoice({ type: 'none' }), 'none');
  // existing cases stay intact
  assert.strictEqual(mapToolChoice('auto'), 'auto');
  assert.strictEqual(mapToolChoice('any'), 'required');
  assert.strictEqual(mapToolChoice({ type: 'any' }), 'required');
  assert.deepStrictEqual(mapToolChoice({ type: 'tool', name: 'search' }), {
    type: 'function',
    function: { name: 'search' },
  });
});

test('stream: final usage chunk populates message_delta input_tokens', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { role: 'assistant', content: 'hi' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 17, completion_tokens: 4 } })}\n\n`,
    'data: [DONE]\n\n',
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  const delta = events.find((e) => /event: message_delta/.test(e));
  const usage = dataOf(delta).usage;
  assert.strictEqual(usage.input_tokens, 17);
  assert.strictEqual(usage.output_tokens, 4);
});

test('request: non-text tool_result block becomes a placeholder, not empty string', () => {
  const out = requestAnthropicToOpenai(
    {
      model: 'm',
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'tu1',
              content: [
                { type: 'text', text: 'see image:' },
                { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAA' } },
              ],
            },
          ],
        },
      ],
    },
    'm'
  );
  const toolMsg = out.messages.find((x) => x.role === 'tool');
  assert.ok(toolMsg, 'tool_result should map to a role:tool message');
  assert.notStrictEqual(toolMsg.content, '');
  assert.match(toolMsg.content, /see image:/);
  assert.match(toolMsg.content, /\[non-text tool result omitted\]/);
});

test('toolResultToText: string passes through, non-text array block is placeholdered', () => {
  assert.strictEqual(toolResultToText('42'), '42');
  assert.strictEqual(
    toolResultToText([{ type: 'image', source: {} }]),
    '[non-text tool result omitted]'
  );
});

test('singleShotSSE: wraps a non-SSE chat.completions JSON as a single-shot Anthropic stream', async () => {
  const json = {
    id: 'cmpl-z',
    choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'buffered reply' } }],
    usage: { prompt_tokens: 7, completion_tokens: 2 },
  };
  const events = await collect(singleShotSSE(json, 'p/m'));
  const types = eventTypes(events);
  assert.strictEqual(types[0], 'message_start');
  assert.strictEqual(types[types.length - 1], 'message_stop');
  assert.strictEqual(dataOf(events[0]).message.model, 'p/m');

  const textDelta = events.find((e) => /"text_delta"/.test(e));
  assert.ok(textDelta, 'expected a text_delta event');
  assert.strictEqual(dataOf(textDelta).delta.text, 'buffered reply');

  const md = events.find((e) => /event: message_delta/.test(e));
  assert.strictEqual(dataOf(md).delta.stop_reason, 'end_turn');
  assert.strictEqual(dataOf(md).usage.input_tokens, 7);
  assert.strictEqual(dataOf(md).usage.output_tokens, 2);
});

test('singleShotSSE: text + tool_call JSON yields text_delta and input_json_delta blocks', async () => {
  const json = {
    id: 'cmpl-t',
    choices: [
      {
        finish_reason: 'tool_calls',
        message: {
          role: 'assistant',
          content: 'calling',
          tool_calls: [{ id: 'call_9', type: 'function', function: { name: 'search', arguments: '{"q":"x"}' } }],
        },
      },
    ],
    usage: { prompt_tokens: 3, completion_tokens: 6 },
  };
  const events = await collect(singleShotSSE(json, 'p/m'));
  const types = eventTypes(events);
  assert.strictEqual(types[0], 'message_start');
  assert.strictEqual(types[types.length - 1], 'message_stop');
  const textDelta = events.find((e) => /"text_delta"/.test(e));
  assert.strictEqual(dataOf(textDelta).delta.text, 'calling');
  const jsonDelta = events.find((e) => /"input_json_delta"/.test(e));
  assert.ok(jsonDelta, 'expected an input_json_delta for the tool call');
  assert.strictEqual(dataOf(jsonDelta).delta.partial_json, '{"q":"x"}');
  const md = events.find((e) => /event: message_delta/.test(e));
  assert.strictEqual(dataOf(md).delta.stop_reason, 'tool_use');
});

test('singleShotSSE: null/unparseable input still emits a valid, terminated empty stream', async () => {
  const events = await collect(singleShotSSE(null, 'p/m'));
  assert.deepStrictEqual(eventTypes(events), ['message_start', 'message_delta', 'message_stop']);
});

// ---------- hardening round 2: interleaved parallel tool calls, cache-token accounting ----------

test('stream: interleaved parallel tool calls map to two stable blocks (no third block, correct reassembly)', async () => {
  // Provider interleaves deltas across two parallel tool calls: tool[0] partial ->
  // tool[1] partial -> tool[0] rest -> tool[1] rest. A single-scalar tracker would
  // re-open a THIRD block for tool 0 and corrupt the JSON. Each OAI index must map to
  // a STABLE Anthropic block index instead.
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { role: 'assistant' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_a', type: 'function', function: { name: 'alpha', arguments: '{"a"' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 1, id: 'call_b', type: 'function', function: { name: 'beta', arguments: '{"b"' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ':1}' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 1, function: { arguments: ':2}' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })}\n\n`,
    'data: [DONE]\n\n',
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));

  // EXACTLY two tool_use blocks opened — no third re-open of tool 0.
  const starts = events.filter((e) => /event: content_block_start/.test(e)).map(dataOf);
  assert.strictEqual(starts.length, 2, 'exactly two content blocks opened');
  assert.ok(starts.every((s) => s.content_block.type === 'tool_use'), 'both blocks are tool_use');

  // Each OAI index owns one block; id/name come from that index's FIRST delta.
  const startByIndex = new Map(starts.map((s) => [s.index, s.content_block]));
  assert.deepStrictEqual([...startByIndex.keys()].sort((a, b) => a - b), [0, 1]);
  assert.strictEqual(startByIndex.get(0).id, 'call_a');
  assert.strictEqual(startByIndex.get(0).name, 'alpha');
  assert.strictEqual(startByIndex.get(1).id, 'call_b');
  assert.strictEqual(startByIndex.get(1).name, 'beta');

  // Interleaved input_json_delta fragments reassemble to the correct JSON per block.
  const argsByIndex = {};
  for (const e of events) {
    if (!/event: content_block_delta/.test(e)) continue;
    const d = dataOf(e);
    if (d.delta.type !== 'input_json_delta') continue;
    argsByIndex[d.index] = (argsByIndex[d.index] || '') + d.delta.partial_json;
  }
  assert.deepStrictEqual(JSON.parse(argsByIndex[0]), { a: 1 });
  assert.deepStrictEqual(JSON.parse(argsByIndex[1]), { b: 2 });

  // Exactly one stop per opened block, in ascending index order, before message_delta.
  const stops = events.filter((e) => /event: content_block_stop/.test(e)).map((e) => dataOf(e).index);
  assert.deepStrictEqual(stops, [0, 1]);

  // No block index beyond 1 ever appears (no corruption / phantom third block).
  const allIndices = events
    .filter((e) => /event: content_block_(start|delta|stop)/.test(e))
    .map((e) => dataOf(e).index);
  assert.ok(
    allIndices.every((i) => i === 0 || i === 1),
    `all content-block indices must be 0 or 1, saw ${JSON.stringify(allIndices)}`
  );

  const md = events.find((e) => /event: message_delta/.test(e));
  assert.strictEqual(dataOf(md).delta.stop_reason, 'tool_use');
});

test('response: cached prompt tokens are subtracted from input_tokens (not double-counted)', () => {
  // OpenAI-style: prompt_tokens is the FULL prompt; cached_tokens is the cache-read
  // subset. Fresh input = prompt_tokens - cached_tokens.
  const a = responseOpenaiToAnthropic(
    {
      choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'hi' } }],
      usage: { prompt_tokens: 100, completion_tokens: 10, prompt_tokens_details: { cached_tokens: 40 } },
    },
    'p/m'
  );
  assert.strictEqual(a.usage.input_tokens, 60);
  assert.strictEqual(a.usage.output_tokens, 10);

  // Anthropic-style alias: cache_read_input_tokens.
  const b = responseOpenaiToAnthropic(
    { choices: [{ message: { content: 'x' } }], usage: { prompt_tokens: 30, completion_tokens: 1, cache_read_input_tokens: 12 } },
    'p/m'
  );
  assert.strictEqual(b.usage.input_tokens, 18);

  // Absent cache fields: unchanged passthrough (prompt_tokens as-is).
  const c = responseOpenaiToAnthropic(
    { choices: [{ message: { content: 'x' } }], usage: { prompt_tokens: 7, completion_tokens: 2 } },
    'p/m'
  );
  assert.strictEqual(c.usage.input_tokens, 7);
});

// ---------- tool-name limit (OpenAI: ^[a-zA-Z0-9_-]{1,64}$) ----------

const {
  toOpenaiToolName,
  toolNameMap,
  usageToAnthropic,
  stopReasonFor,
  OPENAI_TOOL_NAME_MAX,
} = require('../lib/translate');

// A real Claude Code MCP tool name shape, 71 chars — over OpenAI's 64-char ceiling.
const LONG_MCP = 'mcp__langwatch_platform__list_simulation_runs_for_a_given_scenario_set';

test('toolName: a legal short name is left EXACTLY alone (the common case is untouched)', () => {
  for (const n of ['Bash', 'get_weather', 'a-b_c9', 'x'.repeat(64)]) {
    assert.strictEqual(toOpenaiToolName(n), n);
  }
});

test('toolName: an over-long name is truncated to <=64 chars and stays charset-legal', () => {
  assert.ok(LONG_MCP.length > OPENAI_TOOL_NAME_MAX, 'fixture must actually exceed the limit');
  const m = toOpenaiToolName(LONG_MCP);
  assert.ok(m.length <= OPENAI_TOOL_NAME_MAX, `mangled name too long: ${m.length}`);
  assert.match(m, /^[a-zA-Z0-9_-]{1,64}$/);
});

test('toolName: illegal characters are replaced, and the mangle is deterministic', () => {
  const a = toOpenaiToolName('bad name.with:chars!');
  assert.match(a, /^[a-zA-Z0-9_-]{1,64}$/);
  assert.strictEqual(a, toOpenaiToolName('bad name.with:chars!'));
});

test('toolName: two long names sharing a 55-char prefix do NOT collide after truncation', () => {
  const p = 'mcp__server__' + 'x'.repeat(50);
  const a = toOpenaiToolName(p + '_alpha');
  const b = toOpenaiToolName(p + '_beta');
  assert.notStrictEqual(a, b, 'truncation without a content hash would collide here');
  assert.ok(a.length <= 64 && b.length <= 64);
});

test('toolName: reverse map restores the original name; a miss falls back to the wire name', () => {
  const rev = toolNameMap([{ name: LONG_MCP }, { name: 'Bash' }]);
  assert.strictEqual(rev.get(toOpenaiToolName(LONG_MCP)), LONG_MCP);
  assert.strictEqual(rev.get('Bash'), 'Bash');
  assert.strictEqual(rev.get('never_declared'), undefined);
});

test('request: long tool name is mangled in tools[], tool_choice, AND historical tool_use', () => {
  const out = requestAnthropicToOpenai(
    {
      model: 'm',
      tools: [{ name: LONG_MCP, description: 'd', input_schema: { type: 'object' } }],
      tool_choice: { type: 'tool', name: LONG_MCP },
      messages: [
        { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: LONG_MCP, input: { a: 1 } }] },
      ],
    },
    'm'
  );
  const wire = toOpenaiToolName(LONG_MCP);
  assert.strictEqual(out.tools[0].function.name, wire);
  assert.strictEqual(out.tool_choice.function.name, wire);
  // All three sites must agree, or the provider sees two names for one tool.
  assert.strictEqual(out.messages[0].tool_calls[0].function.name, wire);
  assert.ok(wire.length <= 64);
});

test('response: a mangled tool name is restored to the name the CLIENT declared', () => {
  const rev = toolNameMap([{ name: LONG_MCP }]);
  const out = responseOpenaiToAnthropic(
    {
      choices: [
        {
          finish_reason: 'tool_calls',
          message: {
            tool_calls: [
              { id: 'c1', function: { name: toOpenaiToolName(LONG_MCP), arguments: '{}' } },
            ],
          },
        },
      ],
    },
    'p/m',
    rev
  );
  assert.strictEqual(out.content[0].name, LONG_MCP);
});

test('stream: a mangled tool name is restored in content_block_start', async () => {
  const rev = toolNameMap([{ name: LONG_MCP }]);
  const wire = toOpenaiToolName(LONG_MCP);
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: wire, arguments: '{}' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })}\n\n`,
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m', rev));
  const start = events.find((e) => /event: content_block_start/.test(e));
  assert.strictEqual(dataOf(start).content_block.name, LONG_MCP);
});

// ---------- stop_reason: content wins over finish_reason ----------

test('stopReasonFor: tool_use blocks upgrade end_turn, but never mask max_tokens', () => {
  assert.strictEqual(stopReasonFor('stop', true), 'tool_use');
  assert.strictEqual(stopReasonFor('stop', false), 'end_turn');
  assert.strictEqual(stopReasonFor('tool_calls', true), 'tool_use');
  assert.strictEqual(stopReasonFor('length', true), 'max_tokens');
});

test('response: provider emits tool_calls with finish_reason "stop" -> stop_reason is tool_use', () => {
  // Regression: NVIDIA does exactly this (docs/live-nvidia-toolcall-proof.txt [TOOL-4]).
  // Mapping it to end_turn makes the client end the turn instead of running the tool.
  const out = responseOpenaiToAnthropic(
    {
      choices: [
        {
          finish_reason: 'stop',
          message: { tool_calls: [{ id: 'c1', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] },
        },
      ],
    },
    'nvidia/meta-llama'
  );
  assert.strictEqual(out.content[0].type, 'tool_use');
  assert.strictEqual(out.stop_reason, 'tool_use');
});

test('stream: tool_call deltas with finish_reason "stop" -> message_delta stop_reason is tool_use', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'f', arguments: '{}' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`,
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  const md = events.find((e) => /event: message_delta/.test(e));
  assert.strictEqual(dataOf(md).delta.stop_reason, 'tool_use');
});

test('stream: a text-only stream still reports end_turn (no false tool_use upgrade)', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { content: 'hi' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`,
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  const md = events.find((e) => /event: message_delta/.test(e));
  assert.strictEqual(dataOf(md).delta.stop_reason, 'end_turn');
});

// ---------- thinking / reasoning blocks ----------

test('request: thinking and redacted_thinking blocks are dropped, siblings survive', () => {
  const out = requestAnthropicToOpenai(
    {
      model: 'm',
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'secret chain of thought', signature: 'sig-abc' },
            { type: 'redacted_thinking', data: 'opaque-ciphertext' },
            { type: 'text', text: 'visible answer' },
          ],
        },
      ],
    },
    'm'
  );
  const wire = JSON.stringify(out);
  assert.ok(!wire.includes('secret chain of thought'), 'thinking text must not reach the provider');
  assert.ok(!wire.includes('opaque-ciphertext'), 'redacted_thinking must not reach the provider');
  assert.ok(!wire.includes('sig-abc'), 'thinking signature must not reach the provider');
  assert.deepStrictEqual(out.messages[0], { role: 'assistant', content: 'visible answer' });
});

test('request: an assistant turn of ONLY thinking blocks emits no message at all', () => {
  const out = requestAnthropicToOpenai(
    { model: 'm', messages: [{ role: 'assistant', content: [{ type: 'thinking', thinking: 't' }] }] },
    'm'
  );
  assert.deepStrictEqual(out.messages, []);
});

test('request: thinking + tool_use -> the tool_call survives, the thinking does not', () => {
  const out = requestAnthropicToOpenai(
    {
      model: 'm',
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'ponder' },
            { type: 'tool_use', id: 't1', name: 'f', input: {} },
          ],
        },
      ],
    },
    'm'
  );
  assert.strictEqual(out.messages[0].tool_calls[0].function.name, 'f');
  assert.ok(!JSON.stringify(out).includes('ponder'));
});

test('response: reasoning_content is dropped, never re-emitted as an unsigned thinking block', () => {
  // We cannot mint Anthropic's `signature`; a fabricated thinking block would be echoed
  // back by the client and rejected the first time a turn fails open to Anthropic.
  const out = responseOpenaiToAnthropic(
    {
      choices: [
        { finish_reason: 'stop', message: { content: 'the answer', reasoning_content: 'let me think...' } },
      ],
    },
    'p/m'
  );
  assert.deepStrictEqual(out.content, [{ type: 'text', text: 'the answer' }]);
  assert.ok(!out.content.some((b) => b.type === 'thinking'));
  assert.ok(!JSON.stringify(out).includes('let me think'));
});

test('stream: delta.reasoning_content is dropped, visible text still streams', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: 'hidden thoughts' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { content: 'visible' } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`,
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  const joined = events.join('');
  assert.ok(!joined.includes('hidden thoughts'), 'reasoning must not leak into the client stream');
  assert.ok(joined.includes('visible'));
  assert.strictEqual(eventTypes(events).filter((t) => t === 'content_block_start').length, 1);
});

// ---------- cache-token accounting ----------

test('usage: cache_read/cache_creation are surfaced when the provider reports them', () => {
  const u = usageToAnthropic({
    prompt_tokens: 100,
    completion_tokens: 10,
    prompt_tokens_details: { cached_tokens: 40 },
    cache_creation_input_tokens: 7,
  });
  // Anthropic's input_tokens excludes tokens "read from OR used to create a cache",
  // so BOTH counters come off the OpenAI prompt_tokens total: 100 - 40 - 7 = 53.
  assert.deepStrictEqual(u, {
    input_tokens: 53,
    output_tokens: 10,
    cache_read_input_tokens: 40,
    cache_creation_input_tokens: 7,
  });
  // And the three parts reconstruct the provider's full prompt count.
  assert.strictEqual(u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens, 100);
});

test('usage: a reported cached_tokens of 0 is honoured, not skipped by falsiness', () => {
  const u = usageToAnthropic({
    prompt_tokens: 9,
    completion_tokens: 1,
    prompt_tokens_details: { cached_tokens: 0 },
    cache_read_input_tokens: 5,
  });
  assert.strictEqual(u.input_tokens, 9, 'cached_tokens:0 must win over the alias, not fall through to it');
  assert.ok(!('cache_read_input_tokens' in u));
});

test('usage: cache_* keys are ABSENT when the provider reports no cache (no invented zeros)', () => {
  const u = usageToAnthropic({ prompt_tokens: 7, completion_tokens: 2 });
  assert.deepStrictEqual(u, { input_tokens: 7, output_tokens: 2 });
  assert.ok(!('cache_read_input_tokens' in u));
  assert.ok(!('cache_creation_input_tokens' in u));
});

test('usage: missing usage collapses to zeros rather than throwing', () => {
  assert.deepStrictEqual(usageToAnthropic(undefined), { input_tokens: 0, output_tokens: 0 });
  assert.deepStrictEqual(usageToAnthropic(null), { input_tokens: 0, output_tokens: 0 });
});

test('response: cache counters reach the Anthropic envelope', () => {
  const out = responseOpenaiToAnthropic(
    {
      choices: [{ message: { content: 'x' } }],
      usage: { prompt_tokens: 50, completion_tokens: 3, prompt_tokens_details: { cached_tokens: 20 } },
    },
    'p/m'
  );
  assert.strictEqual(out.usage.input_tokens, 30);
  assert.strictEqual(out.usage.cache_read_input_tokens, 20);
});

test('stream: final usage chunk carries cache counters into message_delta', async () => {
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { content: 'x' } }] })}\n\n`,
    `data: ${JSON.stringify({
      choices: [{ delta: {}, finish_reason: 'stop' }],
      usage: { prompt_tokens: 80, completion_tokens: 4, prompt_tokens_details: { cached_tokens: 30 } },
    })}\n\n`,
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  const md = dataOf(events.find((e) => /event: message_delta/.test(e)));
  assert.strictEqual(md.usage.input_tokens, 50);
  assert.strictEqual(md.usage.cache_read_input_tokens, 30);
  assert.strictEqual(md.usage.output_tokens, 4);
});

// ---------- streaming tool-index (landed in 0b2e3bd) — residual-edge verification ----------

test('stream: a provider that OMITS tool_calls[].index keeps one block, not one per delta', async () => {
  // The `lastToolKey` fallback exists for providers that skip `index`. If it regressed,
  // each args fragment would open a fresh content block and the tool JSON would shatter.
  const chunks = [
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ id: 'c1', function: { name: 'f', arguments: '{"a"' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ function: { arguments: ':1}' } }] } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })}\n\n`,
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  const starts = events.filter((e) => /event: content_block_start/.test(e));
  const stops = events.filter((e) => /event: content_block_stop/.test(e));
  assert.strictEqual(starts.length, 1, 'one tool => exactly one content block');
  assert.strictEqual(stops.length, 1, 'every opened block is closed exactly once');
  const partials = events
    .filter((e) => /input_json_delta/.test(e))
    .map((e) => dataOf(e).delta.partial_json)
    .join('');
  assert.strictEqual(partials, '{"a":1}', 'fragments must reassemble in order');
});

test('stream: three interleaved tools each keep a stable block, all closed in ascending order', async () => {
  const d = (i, args, id, name) => ({
    choices: [{ delta: { tool_calls: [{ index: i, ...(id ? { id, function: { name, arguments: args } } : { function: { arguments: args } }) }] } }],
  });
  const chunks = [
    `data: ${JSON.stringify(d(0, '{"x"', 'c0', 'f0'))}\n\n`,
    `data: ${JSON.stringify(d(1, '{"y"', 'c1', 'f1'))}\n\n`,
    `data: ${JSON.stringify(d(2, '{"z"', 'c2', 'f2'))}\n\n`,
    `data: ${JSON.stringify(d(1, ':2}'))}\n\n`,
    `data: ${JSON.stringify(d(0, ':1}'))}\n\n`,
    `data: ${JSON.stringify(d(2, ':3}'))}\n\n`,
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })}\n\n`,
  ];
  const events = await collect(streamOpenaiToAnthropic(fromLines(chunks), 'p/m'));
  assert.strictEqual(events.filter((e) => /content_block_start/.test(e)).length, 3);

  const stopIdx = events.filter((e) => /content_block_stop/.test(e)).map((e) => dataOf(e).index);
  assert.deepStrictEqual(stopIdx, [0, 1, 2], 'blocks close once each, ascending');

  // Reassemble per block index: interleaving must not cross-contaminate.
  const byBlock = {};
  for (const e of events.filter((x) => /input_json_delta/.test(x))) {
    const o = dataOf(e);
    byBlock[o.index] = (byBlock[o.index] || '') + o.delta.partial_json;
  }
  assert.deepStrictEqual(byBlock, { 0: '{"x":1}', 1: '{"y":2}', 2: '{"z":3}' });
});
