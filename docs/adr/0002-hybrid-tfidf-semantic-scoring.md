# 0002. Hybrid Keyword-Overlap + LLM-Judgment Scoring

Date: 2026-02-27
Status: Accepted

## Context

From the very first commit (`4cb739b`, 2026-02-27), `src/matcher.py` already
implements two scoring stages blended into one result — this was not an
incremental evolution away from a single-method approach that this repo's
history shows failing; it is the earliest recorded architecture. The initial
commit message itself describes "a two-stage scoring engine (keyword +
Claude AI)."

**Stage 1 — keyword scoring** (`keyword_score`, always runs, no API key
needed): term-frequency overlap between the nonprofit profile's tokens and
each grant's title (weighted 2x) and description, with a hand-curated
`_PRIORITY_TERMS` set (`reentry`, `housing`, `workforce`, `hipaa`, `veteran`,
etc.) that doubles the weight of nonprofit-sector domain terms, plus a
category-match bonus and a +0.05 eligibility bonus when the grant accepts
nonprofits.

**Stage 2 — AI scoring** (`ai_score`, optional): sends the full profile
summary and grant description to a configured LLM provider (see ADR 0001)
and asks it to return `SCORE: 0-10` plus a `RATIONALE:` sentence — a
judgment call over full natural-language context, not a keyword match.

Combined as `combined = 0.60 * keyword_score + 0.40 * ai_score` when AI
scoring is enabled, or `keyword_score` alone otherwise — documented
identically in the module docstring, `docs/architecture.md` §3, and
`docs/api-reference.md`.

Two important corrections to how this decision is often described (and to
how this repo's own docstring describes it), found by reading the actual
implementation rather than the comments:

- **"TF-IDF" is a misnomer already present in the code's own docstring.**
  The module header calls Stage 1 "TF-IDF-style weighted overlap," but
  `_term_freq()` computes only term frequency within a single grant's
  title/description — there is no inverse-document-frequency term computed
  against any corpus of grants anywhere in the code. It is TF plus a
  hand-curated priority-term boost list, not TF-IDF. This ADR does not
  repeat that overstatement as fact.
- **Stage 2 is not embedding/vector-similarity search.** A repo-wide search
  for `embedding`, `cosine`, and `vector` as implementation terms returns no
  hits in `src/` or `app/`. There is no embeddings model, vector store, or
  similarity computation anywhere. "Semantic" scoring here means an LLM
  reading full context and returning a judgment + rationale, not a nearest-
  neighbor search over vector representations. This is a materially
  different (heavier per-call, costlier, but more explainable via the
  rationale text) architecture than a typical embedding-based semantic
  layer, and anyone extending this expecting a vector index will not find
  one.

Other supporting evidence:

- `app/app.py` surfaces `keyword_score`, `ai_score`, and `combined_score`
  individually in an expandable "Score breakdown" panel — the two signals
  are treated as complementary evidence for a human to weigh, not just an
  internal blending detail hidden from the user.
- The keyword-score normalization was corrected in commit `e9dd221`
  (2026-03-30) to reserve headroom (`min(0.95, ...)`) before applying the
  +0.05 eligibility bonus, replacing an earlier version where the bonus
  could push scores in a less controlled way. This is scoring-calibration
  maintenance, not a change to the two-stage architecture itself.
- AI-call parallelization (`ThreadPoolExecutor(max_workers=min(5,
  len(grants)))`, added in commit `d520554`/`ddbcc1c`, 2026-03-30) and
  precomputing profile tokens once per run (`_get_profile_tokens`) were
  both added specifically to keep the added latency of Stage 2 manageable
  when scoring a full result set.

## Decision

Score every grant with two independent signals rather than one:

1. A fast, deterministic, always-available keyword/term-frequency overlap
   score that requires no external API, costs nothing, and never fails —
   this is the load-bearing default and the only stage that runs when no AI
   provider is configured.
2. An optional LLM-based relevance judgment that reads full natural-language
   context (mission statement, full grant description) and returns both a
   numeric score and a human-readable rationale, to catch conceptual or
   programmatic fit that raw term overlap misses (e.g., a grant that never
   uses the word "reentry" but clearly funds transitional housing for
   justice-involved adults).

Weight the blend 60% keyword / 40% AI when AI scoring is active, so keyword
matching remains the majority signal and AI scoring is a refinement layer
rather than a replacement.

## Consequences

- The tool is fully usable with zero API keys or cost (keyword-only mode) —
  preserved because keyword scoring is architecturally primary, not a
  fallback bolted onto an AI-first design.
- Keyword scoring reliably catches exact/near-exact eligibility-type
  language ("nonprofit," "faith-based") that appears verbatim in Grants.gov
  data, a case where an LLM's paraphrastic judgment adds little and costs
  latency/money.
- The optional LLM stage adds per-grant network latency and, for paid
  providers, marginal per-call cost — mitigated by parallelizing calls and
  by keeping the stage fully optional.
- The blended 60/40 weighting and the `_PRIORITY_TERMS` list are manually
  chosen constants. **No evaluation dataset, benchmark, or A/B test exists
  in this repo to justify 60/40 over any other split**, or to validate that
  the priority-term list is well-calibrated against the real Grants.gov
  corpus. This is a genuine, unaddressed gap — tuning is currently
  judgment-based, not measured.
- The docstring's "TF-IDF" framing should be treated as inaccurate
  shorthand rather than a technical claim; if this project's scoring
  approach is described externally (portfolio write-ups, interviews), it
  should be described as "TF-weighted keyword overlap," not "TF-IDF."
- Because Stage 2 is LLM-judgment-based rather than embeddings-based, this
  system has no vector index and no offline-precomputable similarity
  scores — every AI-scored grant requires a live model call. A future
  embeddings-based semantic layer would be a distinct, additive change, not
  something already partially present under a different name.
