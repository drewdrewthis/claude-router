'use strict';
// JSONL decision logger. One fs.appendFile per line => atomic under concurrency
// on POSIX (small writes with the 'a' flag are not interleaved).
// PII/secret hygiene: only the enumerated decision fields are ever written here.
// Never pass message content, prompt text, or headers into an entry.

const fs = require('node:fs/promises');

function createLogger(file) {
  return {
    file,
    async write(entry) {
      const line = JSON.stringify({ ts: new Date().toISOString(), ...entry }) + '\n';
      await fs.appendFile(file, line);
    },
  };
}

module.exports = { createLogger };
