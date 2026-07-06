# 0001. Orchestrate Grant Relevance Scoring Across Multiple LLM Providers

Date: 2026-03-04
Status: Accepted

## Context

The initial commit (`4cb739b`, 2026-02-27) shipped the AI relevance-scoring
stage hard-coded to a single provider: Anthropic Claude, via
`ANTHROPIC_API_KEY` and a direct `import anthropic` call to a pinned model.

Commit `4c44054` / `4cef75a` ("Refactor AI scoring and security handling",
2026-03-04) replaced this with a provider registry (`_PROVIDERS` in
`src/matcher.py`) supporting four providers:

- **Anthropic Claude** (`ANTHROPIC_API_KEY`, default)
- **OpenAI** (`OPENAI_API_KEY`)
- **Groq** (`GROQ_API_KEY`) — implemented by reusing the OpenAI-compatible
  chat client pointed at `https://api.groq.com/openai/v1`
- **Ollama** (`OLLAMA_BASE_URL`) — local inference, no key required,
  also via the OpenAI-compatible client pointed at `http://localhost:11434/v1`

This was refined further in `60f7bdf` / `7bdfcdc` (2026-03-30), which
expanded `.env.example` and the README with an "AI Provider Support" table,
and added unit test coverage.

Relevant facts found in the repo that motivate this decision:

- The app is a single-tenant, no-auth, no-database, read-only tool
  (`docs/architecture.md`, `docs/security-controls.md`) intended for
  nonprofits — an audience with limited or no dedicated AI budget.
- `docs/compliance-considerations.md` §3.2 and §4.3 explicitly recommends
  switching to Ollama (local) for FedRAMP-sensitive or data-sensitivity-
  conscious deployments, since "Anthropic, OpenAI, and Groq do not currently
  hold FedRAMP authorization for their commercial APIs" and Ollama ensures
  "no profile data ever leaves the operator's environment."
- The README documents Groq as "fast inference, free tier available" —
  a cost-conscious option for the same reason.
- AI scoring (Stage 2) is optional and additive on top of always-on keyword
  scoring (Stage 1); the app is fully functional with zero providers
  configured (see ADR 0002).
- Provider selection is a **static, per-deployment configuration choice**
  (auto-detected from whichever env var key is present, in fixed priority
  Anthropic → OpenAI → Groq → Ollama, or forced via `AI_PROVIDER`) — it is
  not a runtime failover mechanism. `ai_score()` catches `ImportError`,
  `ValueError`, and generic exceptions per call and returns
  `(0.0, "<sanitized message>")`; `score_grants()` then blends this as
  `combined = 0.60 * keyword_score + 0.40 * 0.0`. If the one configured
  provider errors or is down, that grant's AI contribution silently drops to
  zero for that call — there is no automatic retry against a different
  provider.
- Models are pinned to specific dated versions per provider (e.g.
  `claude-haiku-4-5-20251001`, `gpt-4o-mini-2024-07-18`,
  `llama-3.1-8b-instant`, `llama3.2`) specifically so that provider-side
  alias updates cannot silently change output and break the anchored-regex
  response parser (`matcher.py` comment; `docs/compliance-considerations.md`
  §5.3).

## Decision

Support four interchangeable LLM providers behind a single internal
interface (`_score_anthropic` / `_score_openai`, keyed off a `_PROVIDERS`
registry of `provider -> (scorer_fn, install_package, pinned_model)`),
selected via environment-variable auto-detection or an explicit
`AI_PROVIDER` override, instead of hard-coupling the app to one vendor.

Groq and Ollama are implemented by reusing the OpenAI-compatible
chat-completions client with a custom `base_url`, rather than writing a
bespoke integration for each — the marginal cost of adding a new
OpenAI-compatible provider is a registry entry, not a new client.

AI scoring remains strictly optional and additive on top of keyword
scoring, so provider choice (or absence of any provider) never blocks core
functionality.

## Consequences

- Nonprofits can pick whichever provider matches their existing
  account/budget (Anthropic default, OpenAI, Groq free tier), or run fully
  offline/private via Ollama with no profile data leaving their machine —
  directly serving the FedRAMP/data-sensitivity guidance already documented
  in `docs/compliance-considerations.md`.
- Adding another OpenAI-compatible provider in the future is mostly a
  registry entry plus a pinned model, not a new SDK integration.
- Every provider path must independently satisfy the same prompt-injection
  sanitization, response-parsing, error-redaction, and SSRF
  (`_validate_base_url`) requirements — more surface to secure and test than
  a single-provider design. `docs/security-controls.md` §5 notes the SSRF
  guard specifically still lacks dedicated unit tests as of last review.
- Pinning models avoids silent behavioral drift from vendor alias changes,
  at the cost of manual upkeep (and re-testing of the anchored-regex parser)
  whenever a pinned model is deprecated upstream.
- There is **no automatic cross-provider failover**. An operator whose sole
  configured provider has an outage gets silently discounted (keyword-only)
  scores for affected grants rather than a fallback to a second provider.
  This is a deliberate simplicity trade-off, not an oversight, but it should
  be understood as a limitation rather than "orchestration with failover."

**Evidence gap:** the repo does not record *why* these specific four
providers were chosen over others (no issue, design note, or commit message
states a selection rationale beyond what's inferred from the compliance and
README text above). The cost/local-offline/compliance motivations above are
grounded in the docs; the original technical decision-making process for
picking exactly these four is not otherwise documented.
