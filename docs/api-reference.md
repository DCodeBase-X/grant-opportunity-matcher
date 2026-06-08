# API & Module Reference

**Document:** Grant Opportunity Matcher — Internal Module Reference  
**Classification:** Internal — Unrestricted  
**Last Reviewed:** 2026-06-06

---

## Module: `src.grants_api`

### `GrantOpportunity` (dataclass)

Normalized record returned by the Grants.gov API or the mock dataset.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Grants.gov opportunity ID (cast to string) |
| `number` | `str` | Official grant number (e.g., `HUD-2024-001`) |
| `title` | `str` | Grant title |
| `agency` | `str` | Full agency name |
| `agency_code` | `str` | Short agency code (e.g., `HUD`, `DOL`) |
| `status` | `str` | `"posted"`, `"closed"`, or `"archived"` |
| `open_date` | `str \| None` | ISO 8601 open date (`YYYY-MM-DD`) |
| `close_date` | `str \| None` | ISO 8601 close date (`YYYY-MM-DD`) |
| `award_ceiling` | `int \| None` | Maximum award amount in USD |
| `award_floor` | `int \| None` | Minimum award amount in USD |
| `expected_awards` | `int \| None` | Expected number of awards |
| `categories` | `list[str]` | Funding activity categories |
| `eligible_types` | `list[str]` | Eligible applicant types |
| `description` | `str` | Full grant description / synopsis |
| `relevance_score` | `float` | Set by matcher; `0.0` until scored |
| `score_breakdown` | `dict` | Set by matcher; contains `keyword_score`, `ai_score`, `ai_rationale`, `combined_score` |

**Computed properties:**

| Property | Returns | Example |
|---|---|---|
| `award_ceiling_fmt` | `str` | `"$500,000"` or `"Not specified"` |
| `award_floor_fmt` | `str` | `"$50,000"` or `"Not specified"` |
| `deadline_display` | `str` | `"2024-12-31"` or `"Open / Rolling"` |

---

### `search_grants(keywords, max_results, status, funding_min, funding_max)`

Queries the Grants.gov search API and returns normalized results.

```python
grants: list[GrantOpportunity] = search_grants(
    keywords="reentry housing workforce",
    max_results=25,
    status="posted",          # "posted" | "closed" | "archived" | "all"
    funding_min=50_000,       # optional USD floor filter
    funding_max=500_000,      # optional USD ceiling filter
)
```

**Retry behavior:** Up to 3 attempts with exponential backoff (2s, 4s) on timeout and `429`/`503` responses.  
**Rate limiting:** A 2-second minimum interval is enforced between consecutive calls (thread-safe).  
**Raises:** `RuntimeError` on unrecoverable API failure (timeout exhausted, non-retryable HTTP error, unexpected exception).

---

### `fetch_grant_detail(opportunity_id)`

Fetches the full detail record for a single grant. Returns the raw `data` dict from the API response, or `None` on error.

```python
detail: dict | None = fetch_grant_detail("12345")
```

Used for future deep-dive functionality; not currently called by the UI.

---

### `MOCK_GRANTS`

A list of 5 `GrantOpportunity` objects covering housing, reentry, mental health, workforce, and youth services domains. Dates are computed dynamically at import time (`date.today() ± offset`) so they remain "active" regardless of when the application is run. Use via the "Use demo data" toggle in the UI.

---

## Module: `src.profile`

### `NonprofitProfile` (dataclass)

Canonical model for a nonprofit organization.

| Field | Type | Required | Max Length | Description |
|---|---|---|---|---|
| `name` | `str` | Yes | 150 | Organization name |
| `mission` | `str` | Yes | 600 | Mission statement |
| `focus_areas` | `list[str]` | Yes | — | Domain areas (e.g., `["reentry", "housing"]`) |
| `populations` | `list[str]` | Yes | — | Populations served |
| `location_state` | `str` | Yes | 2 | Two-letter state code (uppercased) |
| `location_city` | `str \| None` | No | 100 | City (optional) |
| `org_type` | `str` | No | — | Default: `"Nonprofit"` |
| `tax_status` | `str` | No | — | Default: `"501(c)(3)"` |
| `annual_budget` | `int \| None` | No | — | USD |
| `staff_count` | `int \| None` | No | — | |
| `years_operating` | `int \| None` | No | — | |
| `programs` | `list[str]` | No | — | Key program names |
| `keywords_extra` | `list[str]` | No | — | Additional search terms |
| `funding_min` | `int \| None` | No | — | Desired minimum award (USD) |
| `funding_max` | `int \| None` | No | — | Desired maximum award (USD) |

**Computed properties:**

| Property | Returns | Description |
|---|---|---|
| `search_keywords` | `str` | Deduplicated, max-30-token keyword string for Grants.gov query |
| `all_keywords` | `set[str]` | Full lowercase keyword set for local scoring |

---

### `profile_from_dict(data)` / `load_profile(path)`

Build a `NonprofitProfile` from a dictionary or JSON file.

```python
profile = profile_from_dict({
    "name": "Hope House",
    "mission": "We provide transitional housing for returning citizens.",
    "focus_areas": ["reentry", "housing"],
    "populations": ["justice-involved individuals"],
    "location_state": "NC",
})

profile = load_profile("examples/reentry_housing.json")
```

**Validation applied in `_dict_to_profile`:**

- Required field check (raises `ValueError` on missing fields)
- `name`, `mission`, `location_city`: injection pattern detection + max length enforcement
- `location_state`: must be exactly 2 alpha characters
- All text list fields: comma-string-to-list coercion via `_as_list`

---

## Module: `src.matcher`

### `score_grants(grants, profile, use_ai, min_score)`

Main pipeline entry point. Scores and ranks a list of grants against a profile.

```python
results: list[GrantOpportunity] = score_grants(
    grants=raw_grants,
    profile=profile,
    use_ai=True,              # enable AI scoring stage
    min_score=0.10,           # drop grants below this threshold
)
```

Returns a new list of shallow-copied `GrantOpportunity` objects (original inputs are not mutated) sorted by `relevance_score` descending, filtered to `>= min_score`. Each result has `.relevance_score` and `.score_breakdown` populated.

**Parallelization:** When `use_ai=True` and `len(grants) > 1`, AI calls are dispatched via `ThreadPoolExecutor(max_workers=min(5, len(grants)))`.

---

### `keyword_score(grant, profile, profile_tokens)`

Computes a keyword relevance score in `[0, 1]`.

```python
score: float = keyword_score(grant, profile)
```

**Algorithm:**
1. Tokenize grant title and description (lowercase, strip punctuation, remove stopwords, drop tokens ≤ 2 chars).
2. Compute TF for each field.
3. For each profile token: add `weight × TF` contribution (title weighted 2×; priority-domain terms weighted 2×).
4. Add +0.3 per focus area that appears in grant categories.
5. Normalize score to `[0, 0.95]` via `min(0.95, score / (score + 2.0) * 2.5)`.
6. Add +0.05 eligibility bonus if grant accepts nonprofits/faith-based/community orgs.
7. Clamp to `[0, 1]`.

---

### `ai_score(grant, profile)`

Calls the configured AI provider to score a single grant.

```python
score, rationale = ai_score(grant, profile)
# score: float in [0.0, 1.0]
# rationale: str, HTML-escaped, URL-stripped, max 400 chars
```

Returns `(0.0, "<message>")` if no provider is configured, or if a non-fatal error occurs. Errors are sanitized before returning (API key material redacted). Does not raise.

---

## Module: `src.exporter`

### `to_csv_bytes(grants, profile)`

Generates a UTF-8 CSV export as bytes, suitable for `st.download_button()`.

```python
csv_bytes: bytes = to_csv_bytes(matched_grants, profile)
```

**Columns:** Rank, Title, Agency, Grant Number, Relevance Score, Award Floor, Award Ceiling, Expected Awards, Open Date, Close Date, Categories, Eligible Applicants, AI Rationale, Description (truncated to 300 chars, newlines replaced with spaces).

---

### `to_markdown_report(grants, profile, top_n)`

Generates a human-readable Markdown report for leadership sharing.

```python
report: str = to_markdown_report(matched_grants, profile, top_n=10)
```

Includes organization name, generation date, top-N grants with all metadata, AI rationale (when present), and a profile summary footer. Descriptions truncated to 400 characters.

---

## Score Breakdown Dictionary

Every scored `GrantOpportunity` has a `.score_breakdown` dict with these keys:

| Key | Type | Present When | Description |
|---|---|---|---|
| `keyword_score` | `float` | Always | Stage 1 keyword score `[0, 1]` |
| `ai_score` | `float` | `use_ai=True` | Stage 2 AI score `[0, 1]` |
| `ai_rationale` | `str` | `use_ai=True` | HTML-escaped AI explanation |
| `combined_score` | `float` | Always | Final blended score |

**Blending formula:**
- With AI: `combined = 0.60 × keyword_score + 0.40 × ai_score`
- Without AI: `combined = keyword_score`
