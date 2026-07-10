'use strict';
// Reconcile model-specific request params with a rewritten target model.
//
// decide() rewrites body.model, but Claude Code attaches params only SOME models accept.
// Sent to a model that rejects them, Anthropic 400s — a failure the router's own mutation
// caused, not the caller's. sanitizeForModel strips the params a target cannot accept, per a
// MEASURED capability table (config.modelCapabilities), so a rewrite does not break the turn.
//
// MEASURED (live API): claude-haiku-4-5 400s on output_config.effort ("does not support the
// effort parameter") and on thinking.type==='adaptive' ("adaptive thinking is not supported");
// it ACCEPTS thinking.type==='enabled' with budget_tokens. Other models were rate-limited and
// could NOT be measured — so they carry NO capability claims here (see the unknown-model rule);
// the fail-open 400 retry in router.js is the net for them.

// Context-management edit families whose contract REQUIRES thinking to be present (enabled or
// adaptive). MEASURED (live API): a clear_thinking_20251015 edit with thinking removed/absent
// 400s -> "clear_thinking_20251015 strategy requires thinking to be enabled or adaptive". Match
// the FAMILY, not the dated string, so a future clear_thinking_2026xxxx is handled too. This is
// the general rule: removing a field must cascade to whatever depends on it — a future
// thinking-dependent param gets a new entry here.
const DEPENDS_ON_THINKING = [/^clear_thinking(_|$)/];

// sanitizeForModel(body, targetModel, capabilities) -> { body, stripped }
// `body` is a NEW object when anything was stripped (the caller's parsed body is NEVER
// mutated); `stripped` lists the dotted paths removed, for logging. `capabilities` is
// config.modelCapabilities — a map keyed by exact model id.
function sanitizeForModel(body, targetModel, capabilities) {
  const stripped = [];
  const cap = capabilities && capabilities[targetModel];
  // Unknown model => NO capability claims, strip NOTHING. We only remove a param when we have
  // MEASURED that this exact target rejects it; guessing would break requests a model accepts.
  if (!cap || typeof cap !== 'object') return { body, stripped };

  let out;
  try {
    out = JSON.parse(JSON.stringify(body)); // fresh copy so deleting keys can't mutate the caller
  } catch {
    return { body, stripped }; // non-serializable (shouldn't happen for a parsed body) -> no-op
  }

  // effort:false -> remove output_config.effort; drop output_config entirely if that empties it.
  if (cap.effort === false && out.output_config && typeof out.output_config === 'object') {
    if ('effort' in out.output_config) {
      delete out.output_config.effort;
      stripped.push('output_config.effort');
      if (Object.keys(out.output_config).length === 0) delete out.output_config;
    }
  }

  // adaptiveThinking:false -> drop an adaptive thinking block ENTIRELY. We do NOT downgrade it
  // to {type:'enabled', budget_tokens}: that invents a budget the caller never chose and
  // changes semantics; a light-tier target does not need thinking.
  if (cap.adaptiveThinking === false && out.thinking && out.thinking.type === 'adaptive') {
    delete out.thinking;
    stripped.push('thinking');
  }

  // Dependency cascade: a stripped field's DEPENDENTS must go too. A no-adaptive-thinking target
  // cannot carry context-management edits whose contract requires thinking -- leaving one makes
  // the body self-contradictory ("clear the thinking blocks" + "there is no thinking") and
  // Anthropic 400s. Strip such edits when the OUTGOING body has no thinking (we removed adaptive
  // thinking, or it was absent/null). A target that KEEPS thinking (adaptiveThinking !== false,
  // or an enabled+budget_tokens block we preserved -> out.thinking truthy) is left untouched,
  // since clear_thinking is valid whenever thinking is enabled or adaptive.
  if (
    cap.adaptiveThinking === false &&
    !out.thinking &&
    out.context_management &&
    Array.isArray(out.context_management.edits)
  ) {
    const cm = out.context_management;
    const kept = cm.edits.filter((edit) => {
      const type = edit && typeof edit.type === 'string' ? edit.type : '';
      if (DEPENDS_ON_THINKING.some((re) => re.test(type))) {
        stripped.push(`context_management.edits[${type}]`);
        return false;
      }
      return true;
    });
    if (kept.length !== cm.edits.length) {
      if (kept.length === 0) {
        delete out.context_management; // no edits left -> drop the whole object
        stripped.push('context_management');
      } else {
        cm.edits = kept;
      }
    }
  }

  // Nothing matched => hand back the caller's object unchanged (no spurious copy).
  return stripped.length ? { body: out, stripped } : { body, stripped };
}

// capabilityBlockers(body, targetModel, capabilities) -> string[]
// Content-level reasons the target model cannot serve this request UNCHANGED. Unlike
// sanitizeForModel (which strips PARAMS, always safe), these live in conversation CONTENT and
// must NOT be mutated — merging or dropping a message changes what the model is told — so the
// router DECLINES to route rather than rewrite. Unknown model => [] (no claims; the 400 retry
// remains the net). Pure, no mutation, no I/O.
function capabilityBlockers(body, targetModel, capabilities) {
  const blockers = [];
  const cap = capabilities && capabilities[targetModel];
  if (!cap || typeof cap !== 'object') return blockers;

  // MEASURED (live API): claude-haiku-4-5 400s on a role:'system' message — "role 'system' is
  // not supported on this model". Claude Code emits a mid-conversation role:'system' turn (the
  // mid-conversation-system beta; content is the agent-types reminder) that we cannot merge or
  // drop without changing the instructions the model receives.
  if (cap.systemRole === false && body && Array.isArray(body.messages)) {
    if (body.messages.some((m) => m && m.role === 'system')) blockers.push('system-role-message');
  }
  return blockers;
}

module.exports = { sanitizeForModel, capabilityBlockers };
