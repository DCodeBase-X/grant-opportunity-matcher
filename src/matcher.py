"""
matcher.py
==========
Two-stage grant matching and scoring engine.

Stage 1 — Keyword scoring (always runs, no API key needed)
    TF-IDF-style weighted overlap between the nonprofit profile
    keywords and each grant's title + description.

Stage 2 — AI scoring (optional, requires an AI provider API key)
    Uses a configurable AI provider to produce a nuanced relevance
    score and a human-readable rationale for each match.

    Supported providers (auto-detected from env, or forced via AI_PROVIDER):
        ANTHROPIC_API_KEY  → Anthropic Claude  (default)
        OPENAI_API_KEY     → OpenAI
        GROQ_API_KEY       → Groq  (OpenAI-compatible)
        OLLAMA_BASE_URL    → Ollama (local, no key needed)

    Optional overrides:
        AI_PROVIDER     → "anthropic" | "openai" | "groq" | "ollama"
        AI_MODEL        → override the pinned default model
        OPENAI_BASE_URL → custom base URL for any OpenAI-compatible endpoint

Final score = 0.6 × keyword_score + 0.4 × ai_score  (if AI available)
              or simply keyword_score if AI scoring is skipped.

Security notes
--------------
- All external string fields are sanitized before prompt interpolation
  to defend against prompt-injection attacks embedded in grant data.
- Model output is parsed with anchored regex, HTML-escaped, and length-capped
  before it reaches any caller / UI layer.
- API key material is stripped from exception messages before they are returned.
- base_url values are validated for scheme and SSRF-risky private IP ranges.
- AI_PROVIDER env value is validated against a strict whitelist before use.
"""

import html
import ipaddress
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from urllib.parse import urlparse

from .grants_api import GrantOpportunity
from .profile import NonprofitProfile, profile_summary

_log = logging.getLogger(__name__)



# [ 1 KEYWORD SCORING ]

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
    "hipaa", "mental", "health", "substance", "veteran", "homeless",
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


def _get_profile_tokens(profile: NonprofitProfile) -> set[str]:
    """Pre-compute profile tokens once so they aren't rebuilt for every grant."""
    return set(_tokenize(
        profile.mission
        + " " + " ".join(profile.focus_areas)
        + " " + " ".join(profile.populations)
        + " " + " ".join(profile.programs)
        + " " + " ".join(profile.keywords_extra)
    ))


def keyword_score(
    grant: GrantOpportunity,
    profile: NonprofitProfile,
    profile_tokens: set[str] | None = None,
) -> float:
    """
    Compute a keyword relevance score in [0, 1].

    Combines title matching (weighted 2×) with description matching,
    boosting priority domain terms.
    """
    if profile_tokens is None:
        profile_tokens = _get_profile_tokens(profile)

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
        if re.search(r"\b" + re.escape(area.lower()) + r"\b", grant_cats_lower):
            score += 0.3

    # Normalize keyword score to [0, 0.95], reserving headroom for eligibility bonus
    normalized = min(0.95, score / (score + 2.0) * 2.5)

    # Eligibility bonus — does the grant accept nonprofits?
    elig_text = " ".join(grant.eligible_types).lower()
    if any(w in elig_text for w in ["nonprofit", "non-profit", "faith", "community"]):
        normalized += 0.05

    return round(min(1.0, normalized), 4)


#  [ 2 SECURITY HELPERS ]


# Lines in external data that could hijack prompt instructions
_INJECTION_RE = re.compile(
    r"^\s*(SCORE\s*:|RATIONALE\s*:|IGNORE\s|SYSTEM\s*:|<\s*system"
    r"|assistant\s*:|user\s*:|human\s*:|forget\s+previous"
    r"|disregard\s+previous|new\s+instruction)",
    re.IGNORECASE | re.MULTILINE,
)

def _sanitize_field(value: str, max_len: int = 500) -> str:
    """
    Strip prompt-injection attempts from any external string before embedding
    it in a prompt.  Removes lines that try to hijack the output format or
    issue meta-instructions, then truncates to max_len.
    """
    sanitized = _INJECTION_RE.sub("[REDACTED]", value)
    # Collapse runs of blank lines that could hide injected blocks
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized[:max_len]


def _sanitize_rationale(text: str, max_len: int = 400) -> str:
    """
    HTML-escape model output and strip embedded URLs before the string
    reaches any UI layer (prevents XSS and exfiltration links).
    """
    safe = html.escape(text)
    safe = re.sub(r"https?://\S+", "[URL]", safe)
    return safe[:max_len]


def _sanitize_error(exc: Exception) -> str:
    """
    HTTP-client exceptions often embed Authorization headers or full request
    URLs (including API keys) in their message.  Redact before returning.
    """
    msg = str(exc)
    # Bearer tokens / Anthropic / Groq key prefixes
    msg = re.sub(
        r"(Bearer\s+|sk-|sk-ant-|gsk_)[A-Za-z0-9\-_]{8,}",
        "[REDACTED]",
        msg,
    )
    msg = re.sub(r"api[_-]?key[=:]\s*\S+", "api_key=[REDACTED]", msg, flags=re.IGNORECASE)
    return msg[:300]


_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
]

def _validate_base_url(url: str | None) -> str | None:
    """
    Guard against SSRF: reject private/loopback IPs and non-http(s) schemes.
    Plain http is only allowed for localhost (Ollama default).
    """
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must use http or https, got: {parsed.scheme!r}")
    host = parsed.hostname or ""
    if parsed.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("http base_url is only permitted for localhost endpoints")
    try:
        addr = ipaddress.ip_address(host)
        for net in _PRIVATE_NETS:
            if addr in net and host not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError(f"base_url targets private network: {net}")
    except ValueError as exc:
        if "private network" in str(exc):
            raise
        # host is a domain name — ip_address() raised, that's fine
    return url



# [ 3 PROMPT CONSTRUCTION ]


def _build_prompt(grant: GrantOpportunity, profile: NonprofitProfile) -> str:
    """
    Build the scoring prompt.

    All external-sourced fields are sanitized before interpolation.
    Data is wrapped in XML delimiters so the model cannot confuse
    grant content with instructions.
    """
    return f"""You are a nonprofit grant specialist. Score how well the grant below matches the nonprofit profile.

<nonprofit_profile>
{_sanitize_field(profile_summary(profile), max_len=1500)}
</nonprofit_profile>

<grant_opportunity>
  <title>{_sanitize_field(grant.title, max_len=200)}</title>
  <agency>{_sanitize_field(grant.agency, max_len=200)}</agency>
  <award_range>{grant.award_floor_fmt} – {grant.award_ceiling_fmt}</award_range>
  <deadline>{grant.deadline_display}</deadline>
  <categories>{_sanitize_field(", ".join(grant.categories) or "Not specified", max_len=300)}</categories>
  <eligible_applicants>{_sanitize_field(", ".join(grant.eligible_types) or "Not specified", max_len=300)}</eligible_applicants>
  <description>
{_sanitize_field(grant.description, max_len=1200)}
  </description>
</grant_opportunity>

Respond with ONLY the following two lines — no preamble, no extra text:
SCORE: [integer 0-10]
RATIONALE: [2-3 sentences on match quality, alignment points, and eligibility concerns]"""



# [ 4 RESPONSE PARSING ]

# Anchored regex — not startswith() — so an injected early SCORE: line cannot win
_SCORE_RE     = re.compile(r"^SCORE:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
_RATIONALE_RE = re.compile(r"^RATIONALE:\s*(.+)$", re.MULTILINE)


def _parse_response(response_text: str) -> tuple[float, str]:
    score_match     = _SCORE_RE.search(response_text)
    rationale_match = _RATIONALE_RE.search(response_text)

    if not score_match:
        raise ValueError("Model response missing a valid SCORE line")

    try:
        raw_score = float(score_match.group(1))
    except ValueError:
        raise ValueError(f"Non-numeric score value: {score_match.group(1)!r}")

    if not (0.0 <= raw_score <= 10.0):
        raise ValueError(f"Score out of expected range [0, 10]: {raw_score}")

    normalized = round(raw_score / 10.0, 4)
    rationale  = (
        _sanitize_rationale(rationale_match.group(1))
        if rationale_match
        else "No rationale provided."
    )
    return normalized, rationale



#  [ 5 PROVIDER BACKENDS ]
# -- ANTHROPIC -- OPENAI --

def _score_anthropic(
    prompt: str, api_key: str, model: str, base_url: str | None
) -> tuple[float, str]:
    import anthropic  # lazy — only needed when AI scoring is active
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_response(message.content[0].text)


def _score_openai(
    prompt: str, api_key: str, model: str, base_url: str | None
) -> tuple[float, str]:
    from openai import OpenAI  # lazy — only needed when AI scoring is active
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_response(response.choices[0].message.content)


# Registry: provider_name → (scorer_fn, install_package, pinned_default_model)
# Models are pinned to specific versions so provider alias updates cannot
# silently change behaviour and break output-format parsing.
_PROVIDERS: dict[str, tuple[Callable, str, str]] = {
    "anthropic": (_score_anthropic, "anthropic", "claude-haiku-4-5-20251001"),
    "openai":    (_score_openai,    "openai",    "gpt-4o-mini-2024-07-18"),
    "groq":      (_score_openai,    "openai",    "llama-3.1-8b-instant"),
    "ollama":    (_score_openai,    "openai",    "llama3.2"),
}

_VALID_PROVIDERS = frozenset(_PROVIDERS)


# [ 6  AI SCORING MODEL PUBLIC ENTRY POINT ]


def ai_score(
    grant: GrantOpportunity,
    profile: NonprofitProfile,
) -> tuple[float, str]:
    """
    Score grant relevance using whichever AI provider is configured.

    Provider auto-detection order (first key found wins):
        ANTHROPIC_API_KEY  → Anthropic Claude
        OPENAI_API_KEY     → OpenAI
        GROQ_API_KEY       → Groq  (OpenAI-compatible)
        OLLAMA_BASE_URL    → Ollama (local, no key needed)

    Returns:
        (score_0_to_1, rationale_string)
        Returns (0.0, "<message>") if no provider is configured or an
        error occurs.  The error message is sanitized — no key material.
    """

    # Validate AI_PROVIDER before using or echoing it anywhere 
    raw_provider_env = os.getenv("AI_PROVIDER", "").lower().strip()
    if raw_provider_env and raw_provider_env not in _VALID_PROVIDERS:
        return 0.0, (
            f"Invalid AI_PROVIDER value. "
            f"Choose from: {', '.join(sorted(_VALID_PROVIDERS))}"
        )

    def _resolve() -> tuple[str, str, str | None] | None:
        """Return (provider, api_key, base_url) or None if unconfigured."""
        raw_base_url = os.getenv("OPENAI_BASE_URL")

        if raw_provider_env:
            key_env: str | None = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai":    "OPENAI_API_KEY",
                "groq":      "GROQ_API_KEY",
                "ollama":    None,
            }[raw_provider_env]

            api_key = os.getenv(key_env) if key_env else None
            if key_env and not api_key:
                return None  # provider forced but required key is missing

            provider_urls: dict[str, str] = {
                "groq":   "https://api.groq.com/openai/v1",
                "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            }
            url = provider_urls.get(raw_provider_env, raw_base_url)
            return raw_provider_env, api_key or "none", url

        # Auto-detect — first key found wins
        if key := os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic", key, None
        if key := os.getenv("OPENAI_API_KEY"):
            return "openai", key, raw_base_url
        if key := os.getenv("GROQ_API_KEY"):
            return "groq", key, "https://api.groq.com/openai/v1"
        if url := os.getenv("OLLAMA_BASE_URL"):
            return "ollama", "none", url
        return None

    resolved = _resolve()
    if not resolved:
        return 0.0, (
            "No AI provider configured. "
            "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "GROQ_API_KEY, or OLLAMA_BASE_URL in your .env"
        )

    provider, api_key, base_url = resolved
    scorer_fn, package, default_model = _PROVIDERS[provider]
    model = os.getenv("AI_MODEL", "").strip() or default_model

    # SSRF guard on base_url
    try:
        safe_url = _validate_base_url(base_url)
    except ValueError as exc:
        return 0.0, f"Invalid base URL configuration: {exc}"

    prompt = _build_prompt(grant, profile)

    try:
        return scorer_fn(prompt, api_key, model, safe_url)
    except ImportError:
        msg = f"Install '{package}' package to use the {provider} provider."
        _log.warning("AI scoring skipped: %s", msg)
        return 0.0, msg
    except ValueError as exc:
        _log.warning("AI response parse error (%s): %s", provider, exc)
        return 0.0, f"AI response parsing error: {exc}"
    except Exception as exc:
        sanitized = _sanitize_error(exc)
        _log.warning("AI scoring error (%s): %s", provider, sanitized)
        return 0.0, f"AI scoring error ({provider}): {sanitized}"


# [ 7  MAIN MATCHING PIPELINE ]


def score_grants(
    grants:    list[GrantOpportunity],
    profile:   NonprofitProfile,
    use_ai:    bool = False,
    min_score: float = 0.10,
) -> list[GrantOpportunity]:
    """
    Score and rank a list of grants against a nonprofit profile.

    Args:
        grants:    Grant opportunities from grants_api.search_grants().
        profile:   The nonprofit's profile.
        use_ai:    Whether to run AI scoring (requires a configured provider).
        min_score: Drop grants below this combined score threshold.

    Returns:
        Sorted list of GrantOpportunity objects, highest score first,
        with .relevance_score and .score_breakdown populated.
    """
    profile_tokens = _get_profile_tokens(profile)

    def _score_one(grant: GrantOpportunity) -> GrantOpportunity:
        kw_score = keyword_score(grant, profile, profile_tokens)
        breakdown: dict = {"keyword_score": kw_score}

        if use_ai:
            ai_val, rationale = ai_score(grant, profile)
            breakdown["ai_score"]     = ai_val
            breakdown["ai_rationale"] = rationale

        # Blend scores — include AI weight even when ai_score == 0.0 (genuine poor match)
        if use_ai and "ai_score" in breakdown:
            combined = round(0.60 * kw_score + 0.40 * breakdown["ai_score"], 4)
        else:
            combined = kw_score

        breakdown["combined_score"] = combined
        grant.relevance_score = combined
        grant.score_breakdown  = breakdown
        return grant

    # Parallelize AI scoring (each call is a separate network request)
    if use_ai and len(grants) > 1:
        with ThreadPoolExecutor(max_workers=min(5, len(grants))) as pool:
            all_grants = list(pool.map(_score_one, grants))
    else:
        all_grants = [_score_one(g) for g in grants]

    results = [g for g in all_grants if g.relevance_score >= min_score]
    results.sort(key=lambda g: g.relevance_score, reverse=True)
    return results
