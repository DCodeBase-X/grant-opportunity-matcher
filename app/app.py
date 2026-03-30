"""
Grant Opportunity Matcher
Streamlit dashboard that matches nonprofit profiles to live Grants.gov
opportunities using keyword scoring and optional Claude AI relevance scoring.

Run:
    streamlit run app/app.py
"""

import os
import sys
import json
from pathlib import Path

import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grants_api import search_grants, MOCK_GRANTS, GrantOpportunity
from src.profile    import NonprofitProfile, profile_from_dict, load_profile
from src.matcher    import score_grants
from src.exporter   import to_csv_bytes, to_markdown_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grant Opportunity Matcher",
    page_icon= "app/assets/all_matches.png", 
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# ── Sidebar — Profile Input ───────────────────────────────────────────────────
with st.sidebar:
    st.title("Grant Matcher")
    st.caption("Match your nonprofit to live Grants.gov opportunities")
    st.divider()

    # Load from file or build manually
    profile_mode = st.radio(
        "Profile source",
        ["Load example profile", "Build profile here"],
        horizontal=True,
    )

    if profile_mode == "Load example profile":
        example_files = sorted(EXAMPLES_DIR.glob("*.json"))
        example_names = {f.stem.replace("_", " ").title(): f for f in example_files}
        chosen = st.selectbox("Select example", list(example_names.keys()))
        profile_path  = example_names[chosen]
        profile_data  = json.loads(profile_path.read_text())
        st.info(f"**{profile_data['name']}**\n\n{profile_data['mission']}")
    else:
        st.subheader("Your Organization")
        profile_data = {
            "name":           st.text_input("Organization name", "My Nonprofit"),
            "mission":        st.text_area("Mission statement",
                                "We provide housing and reentry services for justice-involved adults.", height=80),
            "focus_areas":    st.text_input("Focus areas (comma-separated)",
                                "reentry, housing, workforce development"),
            "populations":    st.text_input("Populations served (comma-separated)",
                                "justice-involved individuals, low-income adults"),
            "location_state": st.text_input("State (2-letter)", "NC").upper(),
            "location_city":  st.text_input("City (optional)", ""),
            "tax_status":     st.selectbox("Tax status", ["501(c)(3)", "501(c)(4)", "Government", "Other"]),
            "programs":       st.text_input("Key programs (comma-separated)",
                                "transitional housing, job training, case management"),
            "funding_min":    st.number_input("Min award sought ($)", 0, 5_000_000, 50_000, step=10_000),
            "funding_max":    st.number_input("Max award sought ($)", 0, 10_000_000, 500_000, step=10_000),
        }

    st.divider()
    st.subheader("Search Settings")
    max_results = st.slider("Max results from Grants.gov", 10, 100, 25, step=5)
    grant_status = st.selectbox("Grant status", ["posted", "all", "closed"])
    use_ai       = st.toggle(
        "AI relevance scoring (Claude)",
        value=bool(os.getenv("ANTHROPIC_API_KEY")),
        help="Requires ANTHROPIC_API_KEY in your .env file. Uses claude-haiku for cost efficiency.",
    )
    use_mock     = st.toggle("Use demo data (no live API call)", value=False,
                             help="Uses built-in sample grants instead of calling Grants.gov. Great for demos.")
    min_score    = st.slider("Minimum relevance score", 0.0, 1.0, 0.10, 0.05)

    run_btn = st.button("🔍  Find Matching Grants", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("Grant Opportunity Matcher")
st.markdown(
    "Matches your nonprofit's mission and focus areas to live federal grant opportunities "
    "from [Grants.gov](https://www.grants.gov), scored by keyword alignment"
    + (" and Claude AI relevance scoring." if use_ai else ".")
)

if not run_btn:
    st.info(
        "Configure your organization profile in the sidebar and click **Find Matching Grants** "
        "to surface relevant federal funding opportunities.",
        icon="👈",
    )
    st.stop()

# ── Run the pipeline ──────────────────────────────────────────────────────────
try:
    profile = profile_from_dict(profile_data)
except ValueError as e:
    st.error(f"Profile error: {e}")
    st.stop()

with st.spinner("Searching Grants.gov…" if not use_mock else "Loading demo grants…"):
    if use_mock:
        raw_grants = MOCK_GRANTS
    else:
        try:
            raw_grants = search_grants(
                keywords    = profile.search_keywords,
                max_results = max_results,
                status      = grant_status,
                funding_min = profile.funding_min or None,
                funding_max = profile.funding_max or None,
            )
        except RuntimeError as e:
            st.error(f"**Grants.gov API error:** {e}\n\nTry enabling demo data mode.")
            st.stop()

if not raw_grants:
    st.warning("No grants returned. Try broadening your keywords or enabling demo data mode.")
    st.stop()

ai_label = " + AI" if use_ai else ""
with st.spinner(f"Scoring {len(raw_grants)} opportunities{ai_label}…"):
    matched = score_grants(raw_grants, profile, use_ai=use_ai, min_score=min_score)

# ── Summary metrics ───────────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Grants Fetched",   f"{len(raw_grants)}")
col2.metric("Matches Found",    f"{len(matched)}")
col3.metric("Top Score",        f"{matched[0].relevance_score:.0%}" if matched else "—")
col4.metric("AI Scored",        "Yes" if use_ai else "No")

# ── Export controls ───────────────────────────────────────────────────────────
if matched:
    ex1, ex2 = st.columns(2)
    with ex1:
        csv_data = to_csv_bytes(matched, profile)
        st.download_button(
            "⬇ Download CSV",
            data=csv_data,
            file_name=f"grant_matches_{profile.name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        md_data = to_markdown_report(matched, profile)
        st.download_button(
            "⬇ Download Report (Markdown)",
            data=md_data.encode("utf-8"),
            file_name=f"grant_report_{profile.name.replace(' ', '_').lower()}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ── Results list ──────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"Top Matches for {profile.name}")

for i, grant in enumerate(matched, 1):
    score_pct = f"{grant.relevance_score:.0%}"
    score_color = (
        "🟢" if grant.relevance_score >= 0.65
        else "🟡" if grant.relevance_score >= 0.35
        else "🔴"
    )

    with st.expander(
        f"{score_color} **#{i} — {grant.title}**  ·  {score_pct} match  ·  {grant.agency}",
        expanded=(i <= 3),
    ):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Award Floor",  grant.award_floor_fmt)
        c2.metric("Award Ceiling", grant.award_ceiling_fmt)
        c3.metric("Deadline",      grant.deadline_display)
        c4.metric("Relevance",     score_pct)

        st.markdown(f"**Grant #:** `{grant.number}`   |   **Agency:** {grant.agency}")

        if grant.categories:
            st.markdown(f"**Categories:** {', '.join(grant.categories)}")
        if grant.eligible_types:
            st.markdown(f"**Eligible Applicants:** {', '.join(grant.eligible_types)}")

        # AI rationale
        rationale = grant.score_breakdown.get("ai_rationale", "")
        if rationale and not rationale.startswith("Set ") and not rationale.startswith("AI scoring"):
            st.info(f"**AI Analysis:** {rationale}", icon="🤖")

        # Score breakdown
        with st.expander("Score breakdown", expanded=False):
            bd = grant.score_breakdown
            st.write({
                "Keyword score":  f"{bd.get('keyword_score', 0):.0%}",
                "AI score":       f"{bd.get('ai_score', 0):.0%}" if use_ai else "N/A",
                "Combined score": f"{bd.get('combined_score', 0):.0%}",
            })

        st.markdown("**Description:**")
        st.markdown(grant.description[:800] + ("…" if len(grant.description) > 800 else ""))

        if grant.id and not grant.id.startswith("MOCK"):
            grants_url = f"https://www.grants.gov/search-results-detail/{grant.id}"
            st.markdown(f"[View on Grants.gov ↗]({grants_url})")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by **Damarius McNair** · "
    "[Portfolio](https://dcodebase-x.github.io) · "
    "[GitHub](https://github.com/DCodeBase-X) · "
    "[LinkedIn](https://linkedin.com/in/damariusmcnair)"
)
