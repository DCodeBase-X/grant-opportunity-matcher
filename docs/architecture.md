# System Architecture

**Document:** Grant Opportunity Matcher — Architecture  
**Classification:** Internal — Unrestricted  
**Last Reviewed:** 2026-06-06

---

## 1. Overview

Grant Opportunity Matcher is a Python-based Streamlit application structured as a four-module pipeline. It requires no database, no user authentication, and no persistent storage. All data flows are inbound (Grants.gov API, optional LLM provider API) and outbound only in the form of file downloads initiated by the user.

---

## 2. Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (User Session)                   │
│   Sidebar Profile Input  │  Main Panel Results / Export      │
└──────────────┬───────────┴──────────────────────────────────┘
               │ HTTPS (Streamlit WebSocket)
┌──────────────▼──────────────────────────────────────────────┐
│                    app/app.py  (Streamlit UI)                 │
│  - Session state management                                   │
│  - Input rendering & validation (max_chars, state regex)      │
│  - Cache layer: @st.cache_data TTL=300s                       │
│  - Output rendering (html.escape on all external strings)     │
└────┬──────────────┬──────────────────────────────────────────┘
     │              │
     ▼              ▼
┌─────────┐   ┌──────────────────────────────────────────────┐
│profile.py│   │            matcher.py (Scoring Engine)        │
│          │   │  Stage 1: Keyword scoring (always runs)       │
│ Validate │   │  Stage 2: AI scoring (optional, parallel)     │
│ & model  │   │  Output:  ranked GrantOpportunity list        │
│ profile  │   └───────────────────┬──────────────────────────┘
└──────────┘                       │
                        ┌──────────┴──────────┐
                        │                     │
               ┌────────▼──────┐    ┌─────────▼──────────────┐
               │  grants_api.py│    │   AI Provider (external) │
               │  Grants.gov   │    │   Anthropic / OpenAI /  │
               │  REST API     │    │   Groq / Ollama (local)  │
               └───────────────┘    └─────────────────────────┘
                        │
               ┌────────▼──────┐
               │  exporter.py  │
               │  CSV / MD     │
               │  file download│
               └───────────────┘
```

---

## 3. Module Responsibilities

### `app/app.py` — Streamlit Dashboard
- Owns all user-facing input and output rendering.
- Enforces UI-layer input constraints (`max_chars`, state code regex).
- Applies `html.escape()` to all externally-sourced strings rendered in the UI (grant titles, agencies, categories, numbers).
- Manages a 5-minute result cache via `@st.cache_data(ttl=300)` to avoid redundant API calls.
- Validates that grant IDs match `^[\w\-]+$` before constructing Grants.gov URLs.

### `src/profile.py` — Profile Model
- Defines `NonprofitProfile`, the canonical data model for an organization.
- Enforces field-level validation in `_dict_to_profile()`:
  - Prompt-injection pattern detection on `name`, `mission`, and `location_city` fields.
  - Maximum character length enforcement per field.
  - State code format validation (exactly 2 alpha characters).
- Generates `search_keywords` (deduplicated, max 30 tokens) for the Grants.gov query.

### `src/grants_api.py` — Grants.gov API Client
- Implements `search_grants()` with exponential-backoff retry (max 3 attempts) on timeout and 429/503 responses.
- Enforces a 2-second minimum interval between consecutive API calls (token-bucket rate limiter) to avoid triggering Grants.gov server-side throttling.
- Parses and normalizes raw JSON into `GrantOpportunity` dataclass objects.
- Provides `MOCK_GRANTS` (5 records) with dynamically-computed dates for offline demos.

### `src/matcher.py` — Scoring Engine
- **Stage 1 (Keyword):** TF-weighted overlap between profile tokens and grant title/description. Title is weighted 2×. Priority domain terms (`reentry`, `housing`, `workforce`, etc.) are boosted 2×. An eligibility bonus (+0.05) is applied when the grant accepts nonprofits. Score normalized to [0, 1].
- **Stage 2 (AI):** Optional. Constructs a sanitized prompt and submits to the configured LLM provider. Responses parsed with anchored regex; output HTML-escaped and URL-stripped before display.
- **Final score:** `0.60 × keyword_score + 0.40 × ai_score` when AI is active; otherwise `keyword_score` alone.
- All scoring produces shallow copies of `GrantOpportunity` — original objects are never mutated.
- AI calls are parallelized via `ThreadPoolExecutor(max_workers=min(5, len(grants)))`.

### `src/exporter.py` — Export Engine
- Generates UTF-8 CSV (using stdlib `csv` — no third-party serialization) and Markdown reports.
- File content is assembled in-memory (`io.StringIO`) and returned as bytes; no files are written to disk.

---

## 4. Data Flow

```
User Input (sidebar)
    │
    ├─► profile.py: validate & model
    │
    ├─► grants_api.py: keyword query → Grants.gov API (HTTPS, no auth)
    │                       └─ returns GrantOpportunity[]
    │
    └─► matcher.py: score each grant
            ├─ Stage 1: keyword scoring (local computation)
            └─ Stage 2 (optional): AI provider API (HTTPS, API key in env)
                    └─ returns (score, rationale) per grant
    │
    └─► app.py: render ranked results → user's browser
    └─► exporter.py: generate CSV/Markdown → user downloads file
```

**Data never written to disk.** All intermediate state is in Python memory within the Streamlit server process and the browser session.

---

## 5. Trust Boundaries

| Boundary | Direction | Notes |
|---|---|---|
| Browser → Streamlit server | Inbound | User-controlled profile inputs. Validated at UI layer (max_chars) and model layer (`_dict_to_profile`). |
| Streamlit server → Grants.gov | Outbound | Read-only. No auth credentials transmitted. Public API. |
| Streamlit server → AI provider | Outbound | API key in environment variable. All grant data sanitized before prompt inclusion. Response sanitized before use. |
| Streamlit server → Browser | Outbound | All external string data HTML-escaped before render. |
| User → File download | Outbound | CSV/Markdown assembled in-memory; content reflects only grant data from Grants.gov plus the user's own profile. |

---

## 6. External Dependencies

| Dependency | Version Constraint | Purpose | Data Transmitted |
|---|---|---|---|
| `streamlit` | `>=1.32,<2` | Web framework | None externally |
| `requests` | `>=2.31,<3` | HTTP client | Keyword query to Grants.gov |
| `python-dotenv` | `>=1.0,<2` | Env file loading | None |
| `anthropic` | `>=0.25,<1` (optional) | Claude API client | Sanitized prompt |
| `openai` | `>=1.0,<2` (optional) | OpenAI/Groq client | Sanitized prompt |

---

## 7. Deployment Topology

The application is a single-process Streamlit server. It does not require a database, message queue, or shared file system. It can be deployed as:

- A Docker container (see [Deployment Guide](./deployment-guide.md))
- A systemd service on a Linux host
- A Streamlit Community Cloud deployment (see Secrets management in [Deployment Guide](./deployment-guide.md))

All state is session-scoped. Multi-user concurrency is handled by Streamlit's per-session execution model; no cross-session data sharing occurs.
