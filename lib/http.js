'use strict';
// Transport helpers shared by the forwarding paths. No translation logic lives here.

// Cap for an upstream/provider ERROR body. Error envelopes are small; anything
// larger is either a misconfigured endpoint (an HTML error page, a tarball) or an
// attack. nginx bounds the equivalent read with `proxy_buffer_size` (default 4k/8k);
// 8 KiB is that default, and we only ever log the first 200 chars of it anyway.
const ERROR_BODY_MAX_BYTES = 8192;

// Read at most `maxBytes` of a fetch() Response body as UTF-8, then CANCEL the
// stream so the remainder is never pulled into memory.
//
// Build-vs-reuse check: `Response.text()`, `Response.json()` and `node:stream/consumers`
// all buffer the ENTIRE body with no size ceiling — checked, absent. `Readable.take(n)`
// limits the number of CHUNKS, not bytes — checked, wrong tool. LiteLLM/httpx read the
// whole error body, so there is nothing to borrow there either. The reusable primitive
// is the standard web-stream `body.getReader()` + `reader.cancel()` pair, which is what
// this wraps.
//
// Overshoot is bounded by one chunk: we stop requesting after the cap is reached and
// slice the result, so peak memory is `maxBytes + one upstream chunk`.
//
// Never throws: a body that errors mid-read yields whatever was already buffered.
//
// Self-terminating: a stream that yields only zero-length chunks would otherwise never
// advance `total` and never hit the cap, so consecutive empty reads are counted and
// bounded. Callers must NOT rely on an external abort for termination.
const MAX_EMPTY_READS = 64;

async function readCappedText(res, maxBytes = ERROR_BODY_MAX_BYTES) {
  if (!res || !res.body) return '';
  const reader = res.body.getReader();
  const chunks = [];
  let total = 0;
  let empties = 0;
  try {
    while (total < maxBytes) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.length === 0) {
        if (++empties >= MAX_EMPTY_READS) break; // pathological all-empty stream
        continue; // do not accumulate empty chunks
      }
      empties = 0;
      chunks.push(value);
      total += value.length;
    }
  } catch {
    // Aborted or broken body: fall through and return what we have.
  } finally {
    // Discard the remainder without draining it.
    try {
      await reader.cancel();
    } catch {
      /* already closed */
    }
  }
  return Buffer.concat(chunks).subarray(0, maxBytes).toString('utf8');
}

module.exports = { readCappedText, ERROR_BODY_MAX_BYTES };
