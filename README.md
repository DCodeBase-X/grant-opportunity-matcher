# Grant Opportunity Matcher

A Streamlit dashboard that matches nonprofit organization profiles to live federal grant opportunities from [Grants.gov](https://www.grants.gov), scored by keyword alignment and optional AI relevance scoring.

## Features

- **Live grant search** via the Grants.gov public API (no API key required)
- **Keyword scoring** — TF-weighted overlap between your profile and each grant's title, description, categories, and eligibility
- **AI relevance scoring** — optional second-stage scoring using Claude, OpenAI, Groq, or Ollama
- **Export** results to CSV or a shareable Markdown report
- **Example profiles** for workforce development, reentry housing, and youth services

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/DCodeBase-X/grant-opportunity-matcher.git
cd grant-opportunity-matcher
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API key (see .env.example for all options)
```

AI scoring is optional. If no key is configured, keyword scoring still runs.

### 3. Run the app

```bash
streamlit run app/app.py
```

## AI Provider Support

The app auto-detects which provider to use based on which key is set:

| Provider | Environment Variable | Notes |
|---|---|---|
| Anthropic Claude | `ANTHROPIC_API_KEY` | Default — uses `claude-haiku-4-5` |
| OpenAI | `OPENAI_API_KEY` | Uses `gpt-4o-mini` |
| Groq | `GROQ_API_KEY` | Fast inference, free tier available |
| Ollama | `OLLAMA_BASE_URL` | Local — no key needed |

Override with `AI_PROVIDER` and `AI_MODEL` env vars. See `.env.example` for details.

## Project Structure

```
grant-opportunity-matcher/
├── app/
│   └── app.py              # Streamlit dashboard
├── src/
│   ├── grants_api.py       # Grants.gov API client
│   ├── matcher.py          # Keyword + AI scoring engine
│   ├── profile.py          # Nonprofit profile model
│   └── exporter.py         # CSV and Markdown export
├── examples/               # Sample nonprofit profiles (JSON)
├── tests/                  # Unit tests
├── .env.example            # Environment variable reference
└── requirements.txt
```

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Built by

**Damarius McNair** · [Portfolio](https://dcodebase-x.github.io) · [GitHub](https://github.com/DCodeBase-X) · [LinkedIn](https://linkedin.com/in/damariusmcnair)
