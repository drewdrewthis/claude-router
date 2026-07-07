'use strict';
const test = require('node:test');
const assert = require('node:assert');
const {
  requestAnthropicToOpenai,
  responseOpenaiToAnthropic,
  streamOpenaiToAnthropic,
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
