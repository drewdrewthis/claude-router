'use strict';
const test = require('node:test');
const assert = require('node:assert');

const { readCappedText, ERROR_BODY_MAX_BYTES } = require('../lib/http');

// A Response whose body NEVER ends. An unbounded reader (`res.text()`) hangs here
// forever; readCappedText must return after the cap. This is the regression test for
// "provider answers 500 with an endless body and OOMs/hangs the router".
function endlessResponse(chunkSize = 4096) {
  let cancelled = false;
  const body = new ReadableStream({
    pull(controller) {
      if (cancelled) return;
      controller.enqueue(new Uint8Array(chunkSize).fill(0x61)); // 'a'
    },
    cancel() {
      cancelled = true;
    },
  });
  return { response: { body }, wasCancelled: () => cancelled };
}

test('readCappedText: an endless body returns at the cap instead of reading forever', async () => {
  const { response, wasCancelled } = endlessResponse();
  const text = await readCappedText(response, 1024);
  // Overshoot is bounded by one chunk, and the RESULT is sliced to the cap exactly.
  assert.strictEqual(text.length, 1024);
  assert.ok(/^a+$/.test(text));
  assert.strictEqual(wasCancelled(), true, 'the remainder of the body must be cancelled, not drained');
});

test('readCappedText: a short body is returned whole, under the cap', async () => {
  const body = new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode('{"error":"nope"}'));
      c.close();
    },
  });
  assert.strictEqual(await readCappedText({ body }, 8192), '{"error":"nope"}');
});

test('readCappedText: a body that errors mid-read yields the bytes already buffered', async () => {
  let sent = false;
  const body = new ReadableStream({
    pull(c) {
      if (!sent) {
        sent = true;
        c.enqueue(new TextEncoder().encode('partial'));
        return;
      }
      c.error(new Error('socket died'));
    },
  });
  assert.strictEqual(await readCappedText({ body }, 8192), 'partial');
});

test('readCappedText: a bodyless response is the empty string, not a throw', async () => {
  assert.strictEqual(await readCappedText({ body: null }, 8192), '');
  assert.strictEqual(await readCappedText(null, 8192), '');
});

test('readCappedText: default cap is the nginx-style 8 KiB proxy_buffer_size', async () => {
  assert.strictEqual(ERROR_BODY_MAX_BYTES, 8192);
  const { response } = endlessResponse();
  assert.strictEqual((await readCappedText(response)).length, 8192);
});
