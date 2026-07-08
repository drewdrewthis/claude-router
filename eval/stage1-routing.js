'use strict';
// Stage-1 routing-accuracy eval for the claude-router proxy.
//
// Measures the router's DECISION only (never forwards to any provider): does the
// shipped claude-haiku-4-5 classifier assign sensible light|standard|heavy tier
// labels over a labeled task suite, how much premium->cheap load-shift a cheap-first
// tier map would produce, and whether the modality/privacy cases behave.
//
// Uses SHIPPED product code verbatim — no routing logic is reimplemented:
//   decide()      services/claude-router/lib/decide.js  (canonical decision pipeline)
//   classify()    services/claude-router/lib/classify.js (prompt + parse + timeout)
//   mergeConfig() services/claude-router/router.js       (config helper; no server start)
//
// Each task is classified EXACTLY ONCE: two configs share one content-keyed cache
// Map, so the 2nd (cheap-first) run is a cache-hit that reuses the label.

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

const { decide } = require('../lib/decide');
const { classify } = require('../lib/classify');
const { mergeConfig } = require('../router');

const RESULTS_PATH = path.join(__dirname, 'results', 'stage1-routing.json');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- Auth: real Claude Max OAuth token as the incoming credential. Held in a
// local var only — NEVER printed, echoed, or written anywhere. ---
function readOauthToken() {
  const p = path.join(os.homedir(), '.claude', '.credentials.json');
  const j = JSON.parse(fs.readFileSync(p, 'utf8'));
  const t = j.claudeAiOauth && j.claudeAiOauth.accessToken;
  if (!t) throw new Error('no OAuth accessToken in ~/.claude/.credentials.json');
  return t;
}

// --- The labeled suite (spec-provided; do not invent). ---
const SUITE = [
  // LIGHT
  { id: 'L1', group: 'light', expected: 'light', content: 'What does the -p flag do in `claude -p`?' },
  { id: 'L2', group: 'light', expected: 'light', content: 'Rename the variable `foo` to `bar` in: const foo = 1; console.log(foo);' },
  { id: 'L3', group: 'light', expected: 'light', content: 'What is 2 + 2?' },
  { id: 'L4', group: 'light', expected: 'light', content: "What's the git command to show the current branch?" },
  { id: 'L5', group: 'light', expected: 'light', content: "Convert this string to uppercase in JS: 'hello'" },
  { id: 'L6', group: 'light', expected: 'light', content: 'Fix the typo in this comment: // retrun the sum' },
  { id: 'L7', group: 'light', expected: 'light', content: 'What does `Array.prototype.flat()` do?' },
  { id: 'L8', group: 'light', expected: 'light', content: 'Add a trailing semicolon to this line: const x = 5' },
  // STANDARD
  { id: 'S1', group: 'standard', expected: 'standard', content: 'Write a debounce function in TypeScript with a configurable delay in ms.' },
  { id: 'S2', group: 'standard', expected: 'standard', content: "Add input validation to this Express handler so it 400s when `email` is missing: app.post('/u',(req,res)=>{ save(req.body); res.send('ok') })" },
  { id: 'S3', group: 'standard', expected: 'standard', content: "This Jest test throws 'undefined is not a function' calling sum(). Here's the code: function sum(a,b){return a+b} and the test expects sum to be imported. Fix the import/export." },
  { id: 'S4', group: 'standard', expected: 'standard', content: "Convert this to async/await: fs.readFile('a.txt','utf8',(e,d)=>{ if(e) throw e; console.log(d) })" },
  { id: 'S5', group: 'standard', expected: 'standard', content: 'Write a SQL query to get the top 10 customers by total order value from orders(customer_id, amount).' },
  { id: 'S6', group: 'standard', expected: 'standard', content: "Add limit/offset pagination to this endpoint: app.get('/items',(req,res)=>res.json(db.items))" },
  { id: 'S7', group: 'standard', expected: 'standard', content: 'Write a Jest unit test for: function calculateTax(amount, rate){ return amount * rate }' },
  { id: 'S8', group: 'standard', expected: 'standard', content: 'Refactor this 30-line function to extract the validation into a helper (assume typical field checks).' },
  // HEAVY
  { id: 'H1', group: 'heavy', expected: 'heavy', content: 'Design a caching layer for our REST API: propose the invalidation strategy, storage, and how it interacts with our existing Postgres + Redis setup.' },
  { id: 'H2', group: 'heavy', expected: 'heavy', content: 'Refactor our auth system to support SSO via SAML while keeping existing password login working across web and mobile clients.' },
  { id: 'H3', group: 'heavy', expected: 'heavy', content: 'Under load, two workers sometimes process the same job. Given the queue consumer, DB schema, and lock code, root-cause the race condition and propose a fix.' },
  { id: 'H4', group: 'heavy', expected: 'heavy', content: 'Plan a migration from our monolith to microservices: identify service boundaries, the data-ownership split, and a phased rollout.' },
  { id: 'H5', group: 'heavy', expected: 'heavy', content: 'Architect an event-sourcing system for our orders domain, including snapshotting and replay.' },
  { id: 'H6', group: 'heavy', expected: 'heavy', content: 'Design the schema and API for a multi-tenant billing system with usage-based pricing and proration.' },
  // MODALITY (expected: hasNonText === true; tier label incidental)
  { id: 'M1', group: 'modality', expected: null, content: [
    { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==' } },
    { type: 'text', text: 'What is shown in this screenshot?' },
  ] },
  { id: 'M2', group: 'modality', expected: null, content: [
    { type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: 'JVBERi0xLjQK' } },
    { type: 'text', text: 'Summarize this contract.' },
  ] },
  // PRIVACY-SENSITIVE (ideal: never leave premium/Anthropic; this suite shows the gap)
  { id: 'P1', group: 'privacy', expected: null, content: "Here's a dump of our customer records with names, emails and SSNs (e.g. 123-45-6789). Write a script to de-duplicate them by email." },
  { id: 'P2', group: 'privacy', expected: null, content: 'Review this config: it contains our production database password and Stripe secret key sk_live_FAKE_EXAMPLE_NOT_A_REAL_KEY. Is the structure right?' },
];

function makeBody(content) {
  return { model: 'claude-opus-4-8', max_tokens: 1024, messages: [{ role: 'user', content }] };
}
function routeStr(routed) {
  return routed ? `${routed.provider}/${routed.model}` : null;
}

// nearest-rank percentile over an ascending-sorted numeric array
function percentile(sorted, p) {
  if (!sorted.length) return null;
  const rank = Math.ceil((p / 100) * sorted.length);
  const idx = Math.min(Math.max(rank - 1, 0), sorted.length - 1);
  return sorted[idx];
}

async function main() {
  const token = readOauthToken();
  // authHeaders() reuses these incoming OAuth headers because ROUTER_ANTHROPIC_API_KEY is unset.
  delete process.env.ROUTER_ANTHROPIC_API_KEY;
  const incomingHeaders = {
    authorization: `Bearer ${token}`,
    'anthropic-beta': 'oauth-2025-04-20',
    'anthropic-version': '2023-06-01',
  };

  const configA = mergeConfig({}); // shipped: all-Anthropic tiers
  const configB = mergeConfig({
    tiers: {
      light: 'nvidia,meta/llama-3.1-70b-instruct',
      standard: 'nvidia,meta/llama-3.1-70b-instruct',
      heavy: 'claude-opus-4-8',
    },
  });

  // upstream + classifierModel are identical across A/B, so one classify closure serves both.
  const classifyFn = (digest) => classify({ digest, config: configA, incomingHeaders });
  const cache = new Map(); // shared -> config B run is all cache-hits

  const runDecide = (rawBody, config) =>
    decide({ rawBody, config, classify: classifyFn, now: Date.now, cache });

  const rows = {};

  // ---- Phase A (shipped config): the ONLY phase that actually classifies. ----
  process.stderr.write('[stage1] phase A (classify, all-Anthropic tiers)\n');
  for (const task of SUITE) {
    const rawBody = Buffer.from(JSON.stringify(makeBody(task.content)));
    let dec = await runDecide(rawBody, configA);
    if (dec.log.decision === 'fallback' && dec.log.label == null) {
      process.stderr.write(`[stage1] ${task.id} fallback (${dec.log.fallback_reason || 'unknown'}); retry in 2s\n`);
      await sleep(2000);
      dec = await runDecide(rawBody, configA);
    }
    const errored = dec.log.decision === 'fallback' && dec.log.label == null;
    rows[task.id] = {
      id: task.id,
      group: task.group,
      expected: task.expected,
      predicted: dec.log.label,
      decisionA: dec.log.decision,
      fallback_reason: dec.log.fallback_reason || null,
      classifier_ms: dec.log.classifier_ms != null ? dec.log.classifier_ms : null,
      routeA: routeStr(dec.routed),
      providerA: dec.routed ? dec.routed.provider : null,
      hasNonTextA: dec.hasNonText,
      error: errored,
    };
    process.stderr.write(`[stage1] A ${task.id} (${task.group}) -> ${dec.log.label} ${dec.log.classifier_ms != null ? dec.log.classifier_ms + 'ms' : ''}\n`);
    await sleep(150);
  }

  // ---- Phase B (cheap-first tiers): cache-hits, no new classifier calls. ----
  process.stderr.write('[stage1] phase B (cache-hit, cheap-first tiers)\n');
  for (const task of SUITE) {
    const rawBody = Buffer.from(JSON.stringify(makeBody(task.content)));
    const dec = await runDecide(rawBody, configB);
    const r = rows[task.id];
    r.routeB = routeStr(dec.routed);
    r.providerB = dec.routed ? dec.routed.provider : null;
    r.decisionB = dec.log.decision;
    r.hasNonTextB = dec.hasNonText;
    // Backfill if phase A had errored but B managed to classify (cache miss).
    if (r.error && dec.log.label != null) {
      r.predicted = dec.log.label;
      r.error = false;
      if (r.classifier_ms == null && dec.log.classifier_ms != null) r.classifier_ms = dec.log.classifier_ms;
    }
    if (dec.log.decision !== 'cache-hit') await sleep(150);
  }

  // ---------------- metrics ----------------
  const TIERS = ['light', 'standard', 'heavy'];
  const lsh = SUITE.filter((t) => TIERS.includes(t.group)).map((t) => rows[t.id]);

  // 1. accuracy + confusion matrix (expected x predicted), errors bucketed
  const cm = {};
  for (const e of TIERS) cm[e] = { light: 0, standard: 0, heavy: 0, error: 0 };
  let correct = 0;
  for (const r of lsh) {
    r.match = r.predicted === r.expected;
    if (r.match) correct++;
    const col = r.predicted && TIERS.includes(r.predicted) ? r.predicted : 'error';
    cm[r.expected][col]++;
  }
  const accuracy = { correct, total: lsh.length, pct: +((100 * correct) / lsh.length).toFixed(1) };

  // 2. per-tier precision / recall
  const perTier = {};
  for (const t of TIERS) {
    const tp = cm[t][t];
    const expectedCount = lsh.filter((r) => r.expected === t).length;
    const predictedCount = lsh.filter((r) => r.predicted === t).length;
    perTier[t] = {
      tp,
      expected: expectedCount,
      predicted: predictedCount,
      precision: predictedCount ? +(tp / predictedCount).toFixed(3) : null,
      recall: expectedCount ? +(tp / expectedCount).toFixed(3) : null,
    };
  }

  // 3. classifier latency p50/p95 over every real classifier call (phase A)
  const lat = Object.values(rows)
    .map((r) => r.classifier_ms)
    .filter((v) => typeof v === 'number')
    .sort((a, b) => a - b);
  const latency = {
    n: lat.length,
    p50: percentile(lat, 50),
    p95: percentile(lat, 95),
    min: lat.length ? lat[0] : null,
    max: lat.length ? lat[lat.length - 1] : null,
  };

  // 4. config-B load-shift over the 22 L/S/H tasks
  const toFree = lsh.filter((r) => r.providerB === 'nvidia').length;
  const premium = lsh.filter((r) => r.providerB === 'anthropic').length;
  const loadShift = {
    total: lsh.length,
    to_free_nvidia: toFree,
    pct_free: +((100 * toFree) / lsh.length).toFixed(1),
    premium_opus: premium,
    pct_premium: +((100 * premium) / lsh.length).toFixed(1),
  };

  // 5. modality safety: hasNonText === true under BOTH configs for M1/M2
  const modality = {};
  let modalityPass = true;
  for (const id of ['M1', 'M2']) {
    const r = rows[id];
    const ok = r.hasNonTextA === true && r.hasNonTextB === true;
    modality[id] = { predicted: r.predicted, hasNonTextA: r.hasNonTextA, hasNonTextB: r.hasNonTextB, pass: ok };
    if (!ok) modalityPass = false;
  }

  // 6. privacy gap: P1/P2 decision-level provider under config B
  const privacy = {};
  for (const id of ['P1', 'P2']) {
    const r = rows[id];
    privacy[id] = { predicted: r.predicted, providerB: r.providerB, routeB: r.routeB };
  }

  const out = {
    generated_at: new Date().toISOString(),
    note: 'Decision-only eval: decide() is exercised; no provider is ever contacted. planForward()/NVIDIA are NOT invoked.',
    env: {
      node: process.version,
      ROUTER_ANTHROPIC_API_KEY_set: process.env.ROUTER_ANTHROPIC_API_KEY != null,
      NVIDIA_API_KEY_set: process.env.NVIDIA_API_KEY != null,
    },
    configs: { A_tiers: configA.tiers, B_tiers: configB.tiers, classifierModel: configA.classifierModel, upstream: configA.upstream },
    metrics: {
      configA_accuracy: accuracy,
      confusion_matrix: cm,
      per_tier: perTier,
      classifier_latency_ms: latency,
      configB_load_shift: loadShift,
      modality_safety: { ...modality, pass: modalityPass, gate_ref: 'router.js:313', proven_by: 'test/proxy.test.js:646' },
      privacy_gap: privacy,
    },
    tasks: SUITE.map((t) => {
      const r = rows[t.id];
      return {
        id: r.id, group: r.group, expected: r.expected, predicted: r.predicted,
        match: r.match != null ? r.match : null,
        routeA: r.routeA, routeB: r.routeB, hasNonText: r.hasNonTextA,
        decisionA: r.decisionA, decisionB: r.decisionB,
        classifier_ms: r.classifier_ms, error: r.error, fallback_reason: r.fallback_reason,
      };
    }),
  };

  fs.mkdirSync(path.dirname(RESULTS_PATH), { recursive: true });
  fs.writeFileSync(RESULTS_PATH, JSON.stringify(out, null, 2));

  // ---------------- readable summary ----------------
  const L = (s) => process.stdout.write(s + '\n');
  L('\n================ STAGE-1 ROUTING EVAL ================');
  L(`Config A accuracy (L/S/H, n=${accuracy.total}): ${accuracy.correct}/${accuracy.total} = ${accuracy.pct}%`);
  L('\nConfusion matrix (rows=expected, cols=predicted):');
  L('  expected\\pred   light  standard  heavy  error');
  for (const e of TIERS) {
    L(`  ${e.padEnd(13)} ${String(cm[e].light).padStart(5)} ${String(cm[e].standard).padStart(9)} ${String(cm[e].heavy).padStart(6)} ${String(cm[e].error).padStart(6)}`);
  }
  L('\nPer-tier precision / recall:');
  for (const t of TIERS) {
    const p = perTier[t];
    L(`  ${t.padEnd(9)} precision=${p.precision} recall=${p.recall}  (tp=${p.tp}, predicted=${p.predicted}, expected=${p.expected})`);
  }
  L(`\nClassifier latency (n=${latency.n}): p50=${latency.p50}ms  p95=${latency.p95}ms  (min=${latency.min}, max=${latency.max})`);
  L(`\nConfig B load-shift: ${loadShift.to_free_nvidia}/${loadShift.total} = ${loadShift.pct_free}% -> free (nvidia); ${loadShift.premium_opus}/${loadShift.total} = ${loadShift.pct_premium}% stay premium (opus)`);
  L(`\nModality safety: M1 hasNonText A/B=${modality.M1.hasNonTextA}/${modality.M1.hasNonTextB}, M2 A/B=${modality.M2.hasNonTextA}/${modality.M2.hasNonTextB} -> ${modalityPass ? 'PASS' : 'FAIL'}`);
  L(`\nPrivacy gap (Config B decision-level provider):`);
  for (const id of ['P1', 'P2']) L(`  ${id}: predicted=${privacy[id].predicted}  provider=${privacy[id].providerB}  route=${privacy[id].routeB}`);
  L('\nMismatches (predicted != expected):');
  const misses = lsh.filter((r) => !r.match);
  if (!misses.length) L('  (none)');
  for (const r of misses) L(`  ${r.id} (${r.group}): expected ${r.expected}, got ${r.predicted}`);
  L('\nPer-task table:');
  L('  id   expected  predicted  match  routeA                          routeB                                   nonText');
  for (const t of SUITE) {
    const r = rows[t.id];
    L(`  ${r.id.padEnd(4)} ${String(r.expected).padEnd(9)} ${String(r.predicted).padEnd(10)} ${String(r.match).padEnd(6)} ${String(r.routeA).padEnd(31)} ${String(r.routeB).padEnd(40)} ${r.hasNonTextA}`);
  }
  L(`\nResults JSON: ${RESULTS_PATH}`);
  L('=====================================================');
}

main().catch((e) => {
  console.error('[stage1] FATAL:', (e && e.stack) || e);
  process.exit(1);
});
