# Mission: harden claude-router PR#4 — make it usable (drewdrewthis/claude-router)

Branch `classifier-hardening-and-eval-harness` (PR#4, currently DRAFT, no green CI), worktree root here.
This is drewdrewthis/claude-router. **NEVER merge — Drew merges.** Drive PR#4 to green CI + undrafted + review-ready.

## HARD CONSTRAINT — REUSE-FIRST (Drew: don't reinvent the wheel)
For EACH fix below: FIRST check how **LiteLLM / mature proxies** solve it and BORROW the pattern. Hand-roll ONLY
where Max-OAuth / security / stdlib forces it. The **CORE Anthropic↔OpenAI translation STAYS hand-rolled** per
dec.2026-07-07-claude-router-keep-hand-rolled-translation — do NOT adopt CCR/LiteLLM for the translation itself;
only the edges/robustness borrow. (cf failure-mode hand-rolled-a-solved-problem.)

## Fixes
Core robustness bugs:
1. **Unbounded streaming error-body read** — cap/stream the error body (don't read an unbounded body into memory).
2. **Passthrough `forward()` has no timeout** — add a sensible timeout (borrow the proxy pattern).
3. **Cache-key cross-session collision** — scope the cache key so sessions can't collide.
Round-2 edges:
4. Map streaming **tool-index bug** (streaming tool-call index handling).
5. **64-char tool-name limit** (enforce/handle the provider limit).
6. **thinking-blocks** handling.
7. **cache-token accounting**.

## Definition of done
1. Each fix done reuse-first, with a one-line note in the PR body citing the borrowed pattern (or why hand-rolled).
2. Tests for each fix (the branch has an eval harness — use/extend it); green CI.
3. `git rebase origin/main` if behind.
4. Undraft PR#4; commit + push; run `/review` at the new HEAD; resolve review threads; set assignee; request reviewer.
5. **Do NOT `gh pr merge`** — Drew merges.

## Notes
- No `.env` was copied. If a fix needs live OAuth/creds to test (vs unit-level), FLAG precisely what's needed —
  don't guess or fabricate creds.
- Do NOT commit `.claude-context/`. **NEVER merge.**
