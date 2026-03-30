"""
grants_api.py

Client for the Grants.gov public search API.
No API key required — fully open access.

Docs: https://www.grants.gov/api-reference
"""

import logging
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)

SEARCH_URL  = "https://api.grants.gov/v1/api/search2"
DETAIL_URL  = "https://api.grants.gov/v1/api/fetchOpportunity"
TIMEOUT     = 15   # seconds
MAX_RETRIES = 3


@dataclass
class GrantOpportunity:
    """Normalized grant record from Grants.gov."""
    id:              str
    number:          str
    title:           str
    agency:          str
    agency_code:     str
    status:          str
    open_date:       Optional[str]
    close_date:      Optional[str]
    award_ceiling:   Optional[int]
    award_floor:     Optional[int]
    expected_awards: Optional[int]
    categories:      list[str]  = field(default_factory=list)
    eligible_types:  list[str]  = field(default_factory=list)
    description:     str        = ""

    # Set by the matcher after retrieval
    relevance_score: float = 0.0
    score_breakdown: dict  = field(default_factory=dict)

    @property
    def award_ceiling_fmt(self) -> str:
        if self.award_ceiling is None:
            return "Not specified"
        return f"${self.award_ceiling:,}"

    @property
    def award_floor_fmt(self) -> str:
        if self.award_floor is None:
            return "Not specified"
        return f"${self.award_floor:,}"

    @property
    def deadline_display(self) -> str:
        return self.close_date or "Open / Rolling"


def _parse_opportunity(raw: dict) -> GrantOpportunity:
    """Map a raw Grants.gov JSON record → GrantOpportunity."""
    cats = raw.get("categoryOfFundingActivity") or []
    if isinstance(cats, str):
        cats = [cats]

    elig = raw.get("eligibleApplicants") or []
    if isinstance(elig, str):
        elig = [elig]

    return GrantOpportunity(
        id            = str(raw.get("id", "")),
        number        = raw.get("number", ""),
        title         = raw.get("title", "Untitled"),
        agency        = raw.get("agencyName", ""),
        agency_code   = raw.get("agencyCode", ""),
        status        = raw.get("opportunityStatus", ""),
        open_date     = raw.get("openDate"),
        close_date    = raw.get("closeDate"),
        award_ceiling = raw.get("awardCeiling"),
        award_floor   = raw.get("awardFloor"),
        expected_awards = raw.get("expectedNumberOfAwards"),
        categories    = cats,
        eligible_types = elig,
        description   = raw.get("synopsis", raw.get("description", "")),
    )


def search_grants(
    keywords:       str,
    max_results:    int  = 25,
    status:         str  = "posted",   # "posted" | "closed" | "archived" | "all"
    funding_min:    Optional[int] = None,
    funding_max:    Optional[int] = None,
) -> list[GrantOpportunity]:
    """
    Search Grants.gov for opportunities matching the given keywords.

    Args:
        keywords:     Space-separated keyword string drawn from nonprofit profile.
        max_results:  Maximum grants to return (API max is 2000 per call).
        status:       Filter by opportunity status.
        funding_min:  Minimum award floor filter (optional).
        funding_max:  Maximum award ceiling filter (optional).

    Returns:
        List of GrantOpportunity objects, or empty list on failure.
    """
    payload: dict = {
        "keyword":        keywords,
        "oppStatus":      status if status != "all" else "",
        "rows":           min(max_results, 100),
        "startRecordNum": 0,
        "sortBy":         "openDate|desc",
    }
    if funding_min is not None:
        payload["awardFloor"] = funding_min
    if funding_max is not None:
        payload["awardCeiling"] = funding_max

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(SEARCH_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            # Grants.gov wraps results differently across API versions
            opp_list = (
                data.get("data", {}).get("oppHits")
                or data.get("oppHits")
                or data.get("opportunities")
                or []
            )
            return [_parse_opportunity(o) for o in opp_list]

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    "Grants.gov API timed out after multiple retries. "
                    "Try again later or reduce max_results."
                )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code in (429, 503) and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Grants.gov API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error fetching grants: {e}")

    return []


def fetch_grant_detail(opportunity_id: str) -> Optional[dict]:
    """Fetch full details for a single grant by its Grants.gov opportunity ID."""
    try:
        resp = requests.get(
            DETAIL_URL,
            params={"oppId": opportunity_id},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("data", {})
    except requests.exceptions.Timeout:
        _log.warning("fetch_grant_detail timed out for opportunity %s", opportunity_id)
        return None
    except requests.exceptions.HTTPError as e:
        _log.warning("fetch_grant_detail HTTP %s for opportunity %s", e.response.status_code if e.response is not None else "error", opportunity_id)
        return None
    except Exception as e:
        _log.warning("fetch_grant_detail unexpected %s for opportunity %s", type(e).__name__, opportunity_id)
        return None


# ── Mock data for offline / testing use ──────────────────────────────────────
MOCK_GRANTS = [
    GrantOpportunity(
        id="MOCK-001", number="HUD-2024-001",
        title="Reentry Housing Assistance Program",
        agency="Dept. of Housing and Urban Development", agency_code="HUD",
        status="posted", open_date="2024-01-15", close_date="2024-09-30",
        award_ceiling=500_000, award_floor=50_000, expected_awards=10,
        categories=["Housing", "Recovery Act"],
        eligible_types=["Nonprofits", "Faith-Based Organizations"],
        description=(
            "Funding to support transitional and permanent housing for individuals "
            "returning from incarceration. Eligible programs include case management, "
            "life skills, vocational training, and wraparound reentry services."
        ),
    ),
    GrantOpportunity(
        id="MOCK-002", number="DOJ-BJA-2024-012",
        title="Second Chance Act Reentry Program",
        agency="Dept. of Justice — Bureau of Justice Assistance", agency_code="DOJ",
        status="posted", open_date="2024-02-01", close_date="2024-10-15",
        award_ceiling=1_000_000, award_floor=200_000, expected_awards=5,
        categories=["Law, Justice and Legal Services"],
        eligible_types=["Nonprofits", "State Governments", "Local Governments"],
        description=(
            "Supports reentry programs providing employment, housing, substance use "
            "treatment, mental health services, and family support for individuals "
            "leaving state and local correctional facilities."
        ),
    ),
    GrantOpportunity(
        id="MOCK-003", number="SAMHSA-2024-007",
        title="Community Mental Health and Substance Use Treatment",
        agency="SAMHSA", agency_code="HHS",
        status="posted", open_date="2024-03-01", close_date="2024-11-01",
        award_ceiling=750_000, award_floor=100_000, expected_awards=8,
        categories=["Health", "Recovery and Reinvestment"],
        eligible_types=["Nonprofits", "Faith-Based Organizations", "Tribal Governments"],
        description=(
            "Grants for community-based organizations providing integrated mental health "
            "and substance use disorder treatment services, with priority for underserved "
            "populations including justice-involved individuals and homeless adults."
        ),
    ),
    GrantOpportunity(
        id="MOCK-004", number="DOL-ETA-2024-003",
        title="Workforce Innovation and Opportunity Act — Adult Program",
        agency="Dept. of Labor — Employment and Training Administration", agency_code="DOL",
        status="posted", open_date="2024-01-20", close_date="2024-08-31",
        award_ceiling=2_000_000, award_floor=250_000, expected_awards=4,
        categories=["Employment, Labor and Training"],
        eligible_types=["Nonprofits", "State Agencies", "Local Governments"],
        description=(
            "Funding for adult workforce development programs including skills training, "
            "occupational certifications, job placement, and career advancement services. "
            "Priority populations include low-income adults, individuals with barriers to "
            "employment, and formerly incarcerated individuals."
        ),
    ),
    GrantOpportunity(
        id="MOCK-005", number="ACF-2024-YOUTH-01",
        title="Youth Services and Family Strengthening Grant",
        agency="Administration for Children and Families", agency_code="HHS",
        status="posted", open_date="2024-02-15", close_date="2024-12-31",
        award_ceiling=400_000, award_floor=75_000, expected_awards=12,
        categories=["Children and Families"],
        eligible_types=["Nonprofits", "Faith-Based Organizations"],
        description=(
            "Supports community-based organizations delivering youth development, "
            "mentorship, family strengthening, and afterschool programs in underserved "
            "communities. Eligible services include tutoring, life skills, sports, arts, "
            "parenting support, and child welfare case management."
        ),
    ),
]
