# Grant Opportunity Matcher — Technical Documentation

**Document Owner:** Engineering / Security & Compliance  
**Classification:** Internal — Unrestricted  
**Last Reviewed:** 2026-06-06  
**Review Cadence:** Quarterly or upon significant change

---

## Document Index

| Document | Purpose | Audience |
|---|---|---|
| [System Architecture](./architecture.md) | Component design, data flows, trust boundaries | Engineering, Architecture Review |
| [Security Controls](./security-controls.md) | Control inventory, threat mitigations, security design | Security, Compliance, Engineering |
| [Deployment Guide](./deployment-guide.md) | Production setup, environment config, operational procedures | DevOps, Engineering |
| [API & Module Reference](./api-reference.md) | Internal module interfaces and contracts | Engineering |
| [Compliance Considerations](./compliance-considerations.md) | Data handling, privacy posture, regulatory applicability | Compliance, Legal, TPM |

---

## System Summary

Grant Opportunity Matcher is a single-tenant Streamlit web application that matches nonprofit organization profiles to live federal grant opportunities sourced from the [Grants.gov](https://www.grants.gov) public API. An optional AI-powered relevance scoring stage supports multiple LLM providers (Anthropic Claude, OpenAI, Groq, Ollama).

The system operates as a **read-only intelligence tool** — it does not write to any external systems, store persistent user data, or authenticate users. All grant data is fetched in real time and held only in the active session.

---

## Key Contacts

| Role | Owner |
|---|---|
| Technical Lead / Author | Damarius McNair |
| Security Review | To be assigned per deployment |
| Compliance Review | To be assigned per deployment |
