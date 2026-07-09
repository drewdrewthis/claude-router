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

test('readCappedText: an all-empty-chunk stream self-terminates (no external abort needed)', async () => {
  // A stream that yields only zero-length chunks would never advance `total`. The
  // function must terminate on its own — a future caller may not arm an abort timer.
  let pulls = 0;
  const body = new ReadableStream({
    pull(c) {
      pulls++;
      if (pulls > 100000) { c.close(); return; } // safety net; should never be hit
      c.enqueue(new Uint8Array(0));
    },
  });
  const text = await readCappedText({ body }, 8192);
  assert.strictEqual(text, '');
  assert.ok(pulls < 1000, `should stop after a small bounded number of empty reads, did ${pulls}`);
});

test('readCappedText: empty chunks interleaved with real bytes still return the real bytes', async () => {
  const enc = new TextEncoder();
  const parts = [new Uint8Array(0), enc.encode('re'), new Uint8Array(0), enc.encode('al'), new Uint8Array(0)];
  let i = 0;
  const body = new ReadableStream({
    pull(c) { if (i < parts.length) c.enqueue(parts[i++]); else c.close(); },
  });
  assert.strictEqual(await readCappedText({ body }, 8192), 'real');
});
