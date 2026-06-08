# Security Controls

**Document:** Grant Opportunity Matcher — Security Controls Inventory  
**Classification:** Internal — Unrestricted  
**Last Reviewed:** 2026-06-06  
**Review Cadence:** Quarterly or upon any dependency or architecture change

---

## 1. Purpose

This document inventories the security controls implemented in Grant Opportunity Matcher, maps them to threat categories, and documents the residual risk posture for security review boards and compliance teams. The system is a **read-only, single-tenant intelligence tool** with no authentication layer, no database, and no persistent data storage.

---

## 2. Threat Model Summary

### Assets at Risk

| Asset | Sensitivity | Notes |
|---|---|---|
| AI provider API keys | **High** | Held in environment variables; never logged or transmitted to users |
| Nonprofit profile data | **Low–Medium** | Entered by the operator in-session; not persisted |
| Grants.gov API data | **Public** | No confidentiality concern; all data is federally published |
| Server process integrity | **Medium** | Prompt injection and output manipulation are the primary vectors |

### Threat Actors

| Actor | Motivation | Access Path |
|---|---|---|
| Malicious grant data (Grants.gov API) | Exfiltrate API keys; manipulate scores | Embeds injection strings in grant titles/descriptions |
| Malicious operator input | Same as above; test injection via profile form | Sidebar text fields |
| Network attacker (passive) | Intercept API keys in transit | HTTPS transport; mitigated |

### Out-of-Scope Threats

- Brute-force or credential-stuffing (no user authentication)
- Persistent SQL/NoSQL injection (no database)
- Server-side request forgery via user-supplied file paths (no file upload)
- Multi-tenant data leakage (single-tenant session model)

---

## 3. Control Inventory

### 3.1 Input Validation

| Control ID | Location | Description | Threat Mitigated |
|---|---|---|---|
| IV-01 | `app/app.py` — sidebar inputs | `max_chars` enforced on all Streamlit text fields (100–500 chars depending on field) | Oversized payloads; resource exhaustion |
| IV-02 | `app/app.py` — state code | Validates 2-letter alpha pattern via `st.error` + `st.stop()` before any processing | Malformed input passed to downstream |
| IV-03 | `src/profile.py:_validate_text` | Detects prompt-injection keywords (`SCORE:`, `RATIONALE:`, `SYSTEM:`, `<system`, `IGNORE`, `assistant:`, `forget previous`) via anchored multiline regex in `name`, `mission`, `location_city` fields | Operator-supplied prompt injection |
| IV-04 | `src/profile.py:_dict_to_profile` | Per-field maximum character lengths enforced at model layer (defense-in-depth after UI layer) | Bypass of UI-layer length controls |
| IV-05 | `src/profile.py:_dict_to_profile` | State code validated as exactly 2 alpha characters at model layer | Invalid state codes reaching API query |
| IV-06 | `app/app.py:243` | Grant ID validated against `^[\w\-]+$` before URL construction | Path traversal or injection in constructed Grants.gov URLs |

### 3.2 Prompt Injection Defense

The AI scoring stage embeds externally-sourced data (grant titles, descriptions, categories, eligible applicants) and user-supplied data (profile fields) into LLM prompts. The following layered controls defend against prompt injection:

| Control ID | Location | Description |
|---|---|---|
| PI-01 | `src/matcher.py:_sanitize_field` | Strips lines matching injection patterns (`SCORE:`, `RATIONALE:`, `SYSTEM:`, `<system`, `IGNORE`, `forget previous`, etc.) from all external fields before prompt interpolation; replaces with `[REDACTED]` |
| PI-02 | `src/matcher.py:_sanitize_field` | Collapses 3+ consecutive blank lines to prevent hidden injected blocks |
| PI-03 | `src/matcher.py:_sanitize_field` | Truncates each field to its maximum allowed length (200–1500 chars depending on field) |
| PI-04 | `src/matcher.py:_build_prompt` | All external data wrapped in XML delimiters (`<nonprofit_profile>`, `<grant_opportunity>`, `<title>`, etc.) to structurally separate data from instructions |
| PI-05 | `src/matcher.py:_parse_response` | Response parsed with anchored multiline regex (`^SCORE:`, `^RATIONALE:`) — a SCORE line injected in grant data cannot satisfy the `^` anchor and win over the model's real output |
| PI-06 | `src/matcher.py:_sanitize_rationale` | All model output HTML-escaped before reaching any UI layer; embedded URLs replaced with `[URL]` to block data exfiltration links |

**Residual Risk:** The `focus_areas`, `populations`, `programs`, and `keywords_extra` list items pass through `_as_list()` without the IV-03 rejection check applied to `name` and `mission`. They are still processed by PI-01–PI-04 in the matcher layer. An operator who intentionally embeds injection patterns mid-list-item (e.g., `"reentry, IGNORE previous instructions"`) would have that content processed by `_sanitize_field` before it reaches the model. The residual risk is **Low** given the layered matcher-level sanitization.

**Recommendation:** Apply `_validate_text` to each item produced by `_as_list` in a future iteration to eliminate this asymmetry.

### 3.3 Output Encoding / XSS Prevention

| Control ID | Location | Description | Threat Mitigated |
|---|---|---|---|
| OE-01 | `app/app.py:219–225` | `html.escape()` applied to `grant.number`, `grant.agency`, all categories, and all eligible types before `st.markdown()` render | Stored/reflected XSS via malicious grant metadata from Grants.gov |
| OE-02 | `src/matcher.py:_sanitize_rationale` | `html.escape()` on all model output before storage in `score_breakdown["ai_rationale"]` | Model-generated XSS payloads |
| OE-03 | `src/matcher.py:_sanitize_rationale` | HTTP/HTTPS URLs in model output replaced with `[URL]` | Exfiltration links in AI rationale displayed to user |

### 3.4 API Key Protection

| Control ID | Location | Description | Threat Mitigated |
|---|---|---|---|
| AK-01 | `src/matcher.py:_sanitize_error` | Exception messages scanned for Bearer tokens (`sk-`, `sk-ant-`, `gsk_` prefixes) and `api_key=` parameters; all redacted to `[REDACTED]` before the error string is returned or logged | Key leakage via exception messages surfaced in UI or logs |
| AK-02 | `.env.example` / `.gitignore` | `.env` gitignored; `.env.example` provides placeholder template with no real values | Accidental key commit |
| AK-03 | `src/matcher.py:ai_score` | `AI_PROVIDER` env value validated against a strict whitelist (`anthropic`, `openai`, `groq`, `ollama`) before use or reflection | Env value injection / unexpected provider routing |

### 3.5 SSRF Prevention

| Control ID | Location | Description | Threat Mitigated |
|---|---|---|---|
| SS-01 | `src/matcher.py:_validate_base_url` | Custom `OPENAI_BASE_URL` validated: must use `http` or `https`; `http` only permitted for localhost; host resolved as IP and checked against private RFC-1918 and loopback ranges (`10/8`, `172.16/12`, `192.168/16`, `127/8`, `169.254/16`, `::1/128`) | SSRF against internal network services via custom OpenAI-compatible endpoint |
| SS-02 | `src/matcher.py:_validate_base_url` | Non-http/https schemes (`file://`, `ftp://`, etc.) rejected | Protocol-based SSRF |

### 3.6 Transport Security

| Control ID | Location | Description |
|---|---|---|
| TS-01 | `grants_api.py` / `matcher.py` | All external API calls use HTTPS endpoints (`https://api.grants.gov`, `https://api.anthropic.com`, `https://api.openai.com`, `https://api.groq.com`) |
| TS-02 | `src/grants_api.py:_validate_base_url` | Custom Ollama endpoint (`http://localhost`) permitted for local-only deployment; HTTP to non-localhost is rejected |
| TS-03 | `.streamlit/config.toml` | `enableXsrfProtection = true` (Streamlit built-in CSRF protection enabled) |

### 3.7 Dependency Management

| Control ID | Description |
|---|---|
| DM-01 | All runtime dependencies pinned to compatible minor-version ranges (e.g., `requests>=2.31,<3`) in both `requirements.txt` and `pyproject.toml` |
| DM-02 | AI provider libraries declared as optional extras (`anthropic`, `openai`, `groq`) — not installed unless explicitly requested, reducing unnecessary attack surface |
| DM-03 | `pandas` dependency removed (was listed in prior `requirements.txt` but never used in source; unnecessary transitive dependency surface) |

---

## 4. Security Design Decisions

### Why No Authentication?
The application is designed as a **local or internal-network tool**. Network access control is expected to be provided at the deployment layer (VPN, reverse proxy with auth, or Streamlit Community Cloud team access control). Adding application-layer authentication was out of scope for the initial release.

**Recommendation for internet-facing deployments:** Place the application behind an authenticated reverse proxy (e.g., nginx with HTTP Basic Auth, or a Streamlit Community Cloud workspace with access restrictions).

### Why No Database?
Grants.gov data is publicly available and time-sensitive; there is no value in caching it beyond the session. User profile data is intentionally ephemeral — storing it would introduce data residency, retention, and access control obligations that are out of scope.

### AI Provider Key Isolation
The application only reads AI provider keys from environment variables. Keys are never:
- Written to disk
- Returned in API responses
- Included in export artifacts (CSV, Markdown)
- Logged by the application

The `_sanitize_error` control (AK-01) provides a defensive backstop in case an SDK exception includes key material in its message.

---

## 5. Security Testing Coverage

| Test Class | File | What Is Covered |
|---|---|---|
| Prompt injection sanitization | `tests/test_core.py:TestSanitizeField` | SCORE injection, SYSTEM injection, blank-line collapse, truncation, clean-text passthrough |
| Output encoding | `tests/test_core.py:TestSanitizeRationale` | HTML escaping, URL stripping, truncation |
| API key redaction | `tests/test_core.py:TestSanitizeError` | Bearer token redaction, `api_key=` redaction, truncation |
| Response parsing (injection resistance) | `tests/test_core.py:TestParseResponse` | Anchored regex prevents injected SCORE lines from winning |
| Profile validation | `tests/test_core.py:TestProfileFromDict` | Missing required fields, state code uppercasing |
| Grant ID validation | `app/app.py:243` | Inline regex guard (not currently unit-tested — **gap**) |

**Testing Gap:** The `_validate_base_url` SSRF guard (SS-01) has no dedicated unit tests. This should be added in the next sprint.

**Testing Gap:** The new `_validate_text` and `_INJECTION_RE` in `profile.py` have no dedicated unit tests. This should be added in the next sprint.

---

## 6. Residual Risk Register

| Risk ID | Description | Likelihood | Impact | Residual Risk | Mitigation Owner |
|---|---|---|---|---|---|
| RR-01 | Injection via list profile fields (`focus_areas`, etc.) bypasses IV-03 but is caught by PI-01 | Low | Low | **Low** | Engineering (future sprint) |
| RR-02 | No application-layer auth — relies entirely on network-layer access control | Medium (if internet-facing) | High | **Medium** | Deployment team |
| RR-03 | `_validate_base_url` SSRF guard untested | Low | Medium | **Low** | Engineering (next sprint) |
| RR-04 | `pandas` removal may break any undocumented use by a future developer who re-adds it without noting it was intentionally removed | Low | Low | **Low** | Documentation (done) |

---

## 7. Recommended Next Actions

| Priority | Action | Owner | Target |
|---|---|---|---|
| P1 | Add unit tests for `_validate_text`, `_INJECTION_RE`, and `_validate_base_url` | Engineering | Next sprint |
| P1 | Document network access control requirements for internet-facing deployments | TPM / DevOps | Before any public deployment |
| P2 | Apply `_validate_text` to `_as_list` items for `focus_areas`, `populations`, `programs`, `keywords_extra` | Engineering | Next sprint |
| P2 | Add Dependabot or equivalent SCA scanning to CI pipeline | DevOps / Security | Next quarter |
| P3 | Add a `SECURITY.md` with responsible disclosure process | Engineering / TPM | Next sprint |
