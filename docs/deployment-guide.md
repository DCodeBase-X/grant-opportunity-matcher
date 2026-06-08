# Deployment Guide

**Document:** Grant Opportunity Matcher — Production Deployment  
**Classification:** Internal — Unrestricted  
**Last Reviewed:** 2026-06-06

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or later |
| pip | 23.0 or later |
| Network access | Outbound HTTPS to `api.grants.gov` (port 443) |
| AI provider access | Outbound HTTPS to provider API (optional) |

---

## 2. Installation

### 2.1 Clone and install

```bash
git clone https://github.com/DCodeBase-X/grant-opportunity-matcher.git
cd grant-opportunity-matcher

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                   # installs core runtime deps from pyproject.toml
```

### 2.2 Install an AI provider (optional)

Install only the package for the provider you are using:

```bash
pip install anthropic               # Anthropic Claude (recommended default)
pip install openai                  # OpenAI or Groq (Groq uses the OpenAI SDK)
```

Ollama requires no Python package — set `OLLAMA_BASE_URL` and ensure the Ollama service is running locally.

---

## 3. Environment Configuration

Copy the example file and populate your values:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Set exactly ONE provider key (or none — keyword scoring still works without AI):
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GROQ_API_KEY=gsk_...
# OLLAMA_BASE_URL=http://localhost:11434/v1

# Optional overrides:
# AI_PROVIDER=anthropic            # force a specific provider
# AI_MODEL=claude-haiku-4-5-20251001  # override the pinned default model
```

**Security:** `.env` is listed in `.gitignore` and must never be committed. In CI/CD or container deployments, inject secrets via the deployment platform's secrets manager rather than a `.env` file.

### 3.1 Environment Variable Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | No | — | Anthropic Claude API key |
| `OPENAI_API_KEY` | No | — | OpenAI API key |
| `GROQ_API_KEY` | No | — | Groq API key |
| `OLLAMA_BASE_URL` | No | — | Ollama base URL (e.g., `http://localhost:11434/v1`) |
| `AI_PROVIDER` | No | Auto-detect | Force a provider: `anthropic` \| `openai` \| `groq` \| `ollama` |
| `AI_MODEL` | No | Provider default | Override the pinned model for the selected provider |
| `OPENAI_BASE_URL` | No | Provider default | Custom base URL for any OpenAI-compatible endpoint |

### 3.2 Provider Auto-Detection Order

When `AI_PROVIDER` is not set, the application detects the provider by checking keys in this order:

1. `ANTHROPIC_API_KEY` → Anthropic Claude (`claude-haiku-4-5-20251001`)
2. `OPENAI_API_KEY` → OpenAI (`gpt-4o-mini-2024-07-18`)
3. `GROQ_API_KEY` → Groq (`llama-3.1-8b-instant`)
4. `OLLAMA_BASE_URL` → Ollama (`llama3.2`)

---

## 4. Running the Application

### 4.1 Local development

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

### 4.2 Production server

```bash
streamlit run app/app.py \
  --server.port 8501 \
  --server.headless true \
  --server.enableXsrfProtection true \
  --browser.gatherUsageStats false
```

These settings are also persisted in `.streamlit/config.toml` and apply automatically without flags.

### 4.3 Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e . anthropic

# Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]
```

```bash
docker build -t grant-matcher .
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  grant-matcher
```

**Do not bake API keys into the Docker image.** Inject at runtime via `-e` flags or a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault, etc.).

### 4.4 Streamlit Community Cloud

1. Fork the repository to your GitHub account.
2. Connect it to Streamlit Community Cloud.
3. Set secrets in the Streamlit Cloud dashboard under **App settings → Secrets** — do not use `.env` files in the repository.
4. The app's main file path is `app/app.py`.

---

## 5. Network Requirements

| Destination | Port | Protocol | Purpose | Required |
|---|---|---|---|---|
| `api.grants.gov` | 443 | HTTPS | Live grant search | Yes (unless demo mode only) |
| `api.anthropic.com` | 443 | HTTPS | Claude AI scoring | Only if `ANTHROPIC_API_KEY` set |
| `api.openai.com` | 443 | HTTPS | OpenAI scoring | Only if `OPENAI_API_KEY` set |
| `api.groq.com` | 443 | HTTPS | Groq scoring | Only if `GROQ_API_KEY` set |
| `localhost:11434` | 11434 | HTTP | Ollama (local) | Only if Ollama used |

No inbound ports beyond the Streamlit server port (8501 default) are required.

---

## 6. Access Control

The application has **no built-in authentication**. For any deployment accessible beyond a single trusted user's localhost, access must be controlled at the infrastructure layer:

| Deployment Type | Recommended Access Control |
|---|---|
| Local workstation | No additional control needed |
| Internal team server | VPN requirement + firewall rules restricting port 8501 to authorized IP ranges |
| Cloud VM | Security group / firewall allowing port 8501 only from known CIDR blocks |
| Public internet | Authenticated reverse proxy (nginx + HTTP Basic Auth or OAuth2 proxy) in front of the Streamlit process |
| Streamlit Community Cloud | Team workspace with invite-only access |

---

## 7. Operational Runbook

### 7.1 Restarting the application

The application is stateless. A restart clears all in-memory caches and active sessions. No data loss occurs.

```bash
# If running via systemd:
sudo systemctl restart grant-matcher

# If running manually — Ctrl+C then:
streamlit run app/app.py
```

### 7.2 Updating the application

```bash
git pull origin main
pip install -e .                   # picks up any new dependencies
sudo systemctl restart grant-matcher   # or restart manually
```

### 7.3 Checking connectivity to Grants.gov

```bash
curl -s -o /dev/null -w "%{http_code}" \
  https://api.grants.gov/v1/api/search2 \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"keyword":"housing","rows":1,"startRecordNum":0}'
```

Expected: `200`. If the response is `503` or a timeout, Grants.gov may be experiencing downtime — enable **demo data mode** in the sidebar as a fallback.

### 7.4 Rotating API keys

1. Generate a new key in the provider's dashboard.
2. Update the value in `.env` (local) or the secrets manager (production).
3. Restart the application process so the new environment variable is loaded.
4. Revoke the old key in the provider's dashboard.

The application does not cache API keys in-memory between requests; a restart is sufficient to pick up the new value.

---

## 8. Dependency Updates

Run quarterly or when a CVE is disclosed for a dependency:

```bash
pip install --upgrade -e .
pip install --upgrade anthropic     # or openai, if in use
```

Review the changelog for breaking changes before upgrading `streamlit` major versions (`<2.0.0` constraint currently enforced).

---

## 9. Logging

The application writes to stdout/stderr via Python's `logging` module at `WARNING` level by default (configured in `.streamlit/config.toml`). No PII or API key material is included in log output — `_sanitize_error()` redacts key material from exception messages before they reach the logger.

To increase verbosity for debugging:

```toml
# .streamlit/config.toml
[logger]
level = "info"
```

Or via environment variable:

```bash
STREAMLIT_LOGGER_LEVEL=info streamlit run app/app.py
```
