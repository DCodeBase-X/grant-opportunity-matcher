"""
matcher.py
==========
Two-stage grant matching and scoring engine.

Stage 1 — Keyword scoring (always runs, no API key needed)
    TF-IDF-style weighted overlap between the nonprofit profile
    keywords and each grant's title + description.

Stage 2 — AI scoring (optional, requires ANTHROPIC_API_KEY)
    Uses Claude to produce a nuanced relevance score and a
    human-readable rationale for each match.

Final score = 0.6 × keyword_score + 0.4 × ai_score (if available)
              or simply keyword_score if AI scoring is skipped.
"""

import os
import math
import re
from typing import Optional

from .grants_api import GrantOpportunity
from .profile    import NonprofitProfile, profile_summary

# ── Keyword scoring ───────────────────────────────────────────────────────────

# Stopwords to exclude from keyword matching
_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "are", "from",
    "has", "have", "will", "its", "not", "but", "can", "all",
    "any", "may", "one", "two", "new", "use", "also", "each",
    "their", "they", "been", "more", "other", "than", "into",
    "was", "our", "who", "how", "such", "when", "what", "which",
}

# High-value terms that double their weight when matched
_PRIORITY_TERMS = {
    "reentry", "re-entry", "incarceration", "justice", "housing",
    "transitional", "workforce", "vocational", "youth", "nonprofit",
    "hipaa", "mental health", "substance", "veteran", "homeless",
    "community", "501c3", "faith", "family", "training", "services",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [t for t in text.split() if t not in _STOPWORDS and len(t) > 2]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Compute normalized term frequency."""
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def keyword_score(grant: GrantOpportunity, profile: NonprofitProfile) -> float:
    """
    Compute a keyword relevance score in [0, 1].

    Combines title matching (weighted 2×) with description matching,
    boosting priority domain terms.
    """
    profile_tokens = set(_tokenize(
        profile.mission
        + " " + " ".join(profile.focus_areas)
        + " " + " ".join(profile.populations)
        + " " + " ".join(profile.programs)
        + " " + " ".join(profile.keywords_extra)
    ))

    # Grant text: title is more signal-dense, weight it higher
    title_tokens = _tokenize(grant.title)
    desc_tokens  = _tokenize(grant.description)

    title_tf = _term_freq(title_tokens)
    desc_tf  = _term_freq(desc_tokens)

    score = 0.0
    for term in profile_tokens:
        t_weight = 2.0 if term in _PRIORITY_TERMS else 1.0
        # Title match (weight 2×)
        if term in title_tf:
            score += t_weight * 2.0 * title_tf[term]
        # Description match
        if term in desc_tf:
            score += t_weight * desc_tf[term]

    # Category bonus
    grant_cats_lower = " ".join(grant.categories).lower()
    for area in profile.focus_areas:
        if area.lower() in grant_cats_lower:
            score += 0.3

    # Eligibility bonus — does the grant accept nonprofits?
    elig_text = " ".join(grant.eligible_types).lower()
    if any(w in elig_text for w in ["nonprofit", "non-profit", "faith", "community"]):
        score += 0.2

    # Normalize to [0, 1] using a logistic-like squash
    return round(min(1.0, score / (score + 2.0) * 2.5), 4)


# ── AI scoring (Claude) ───────────────────────────────────────────────────────

def ai_score(
    grant: GrantOpportunity,
    profile: NonprofitProfile,
) -> tuple[float, str]:
    """
    Use Claude to score grant relevance and produce a rationale.

    Returns:
        (score_0_to_1, rationale_string)
        Returns (0.0, "AI scoring unavailable") if no API key is set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return 0.0, "Set ANTHROPIC_API_KEY in .env to enable AI scoring."

    try:
        import anthropic  # lazy import — only needed for AI scoring

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are a nonprofit grant specialist. Score how well the following grant opportunity matches this nonprofit organization's profile.

NONPROFIT PROFILE:
{profile_summary(profile)}

GRANT OPPORTUNITY:
Title: {grant.title}
Agency: {grant.agency}
Award Range: {grant.award_floor_fmt} – {grant.award_ceiling_fmt}
Deadline: {grant.deadline_display}
Categories: {', '.join(grant.categories) or 'Not specified'}
Eligible Applicants: {', '.join(grant.eligible_types) or 'Not specified'}
Description:
{grant.description[:1200]}

Respond with ONLY the following format — no extra text:
SCORE: [number from 0 to 10]
RATIONALE: [2-3 sentences explaining the match quality, including the strongest alignment points and any eligibility concerns]"""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text.strip()

        # Parse score
        score_line = next(
            (l for l in response_text.splitlines() if l.startswith("SCORE:")), ""
        )
        rationale_line = next(
            (l for l in response_text.splitlines() if l.startswith("RATIONALE:")), ""
        )

        raw_score  = float(score_line.replace("SCORE:", "").strip())
        normalized = round(min(1.0, max(0.0, raw_score / 10.0)), 4)
        rationale  = rationale_line.replace("RATIONALE:", "").strip()

        return normalized, rationale

    except ImportError:
        return 0.0, "Install 'anthropic' package to enable AI scoring."
    except Exception as e:
        return 0.0, f"AI scoring error: {e}"


# ── Main matching pipeline ────────────────────────────────────────────────────

def score_grants(
    grants:       list[GrantOpportunity],
    profile:      NonprofitProfile,
    use_ai:       bool = False,
    min_score:    float = 0.10,
) -> list[GrantOpportunity]:
    """
    Score and rank a list of grants against a nonprofit profile.

    Args:
        grants:    Grant opportunities from grants_api.search_grants().
        profile:   The nonprofit's profile.
        use_ai:    Whether to run Claude-powered AI scoring (requires API key).
        min_score: Drop grants below this combined score threshold.

    Returns:
        Sorted list of GrantOpportunity objects, highest score first,
        with .relevance_score and .score_breakdown populated.
    """
    results = []

    for grant in grants:
        kw_score = keyword_score(grant, profile)

        breakdown = {"keyword_score": kw_score}
        ai_weight = 0.0
        rationale = ""

        if use_ai:
            ai_val, rationale = ai_score(grant, profile)
            breakdown["ai_score"] = ai_val
            breakdown["ai_rationale"] = rationale
            ai_weight = ai_val

        # Combined weighted score
        if use_ai and breakdown.get("ai_score", 0) > 0:
            combined = round(0.60 * kw_score + 0.40 * ai_weight, 4)
        else:
            combined = kw_score

        breakdown["combined_score"] = combined
        grant.relevance_score = combined
        grant.score_breakdown  = breakdown

        if combined >= min_score:
            results.append(grant)

    results.sort(key=lambda g: g.relevance_score, reverse=True)
    return results
