"""
profile.py
==========
Loads, validates, and processes nonprofit organization profiles
used as the basis for grant matching.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NonprofitProfile:
    """
    Represents a nonprofit organization's profile for grant matching.

    All text fields are used to generate a keyword search query and
    compute relevance scores against grant descriptions.
    """
    name:              str
    mission:           str
    focus_areas:       list[str]          # e.g. ["reentry", "housing", "workforce"]
    populations:       list[str]          # e.g. ["justice-involved", "youth", "veterans"]
    location_state:    str                # two-letter state code, e.g. "NC"
    location_city:     Optional[str]      = None
    org_type:          str                = "Nonprofit"
    tax_status:        str                = "501(c)(3)"
    annual_budget:     Optional[int]      = None   # USD
    staff_count:       Optional[int]      = None
    years_operating:   Optional[int]      = None
    programs:          list[str]          = field(default_factory=list)
    keywords_extra:    list[str]          = field(default_factory=list)
    funding_min:       Optional[int]      = None   # desired award floor
    funding_max:       Optional[int]      = None   # desired award ceiling

    @property
    def search_keywords(self) -> str:
        """
        Combines all profile fields into a single keyword string
        for the Grants.gov API query.
        """
        parts = (
            [self.mission]
            + self.focus_areas
            + self.populations
            + self.programs
            + self.keywords_extra
        )
        # Deduplicate and join
        seen  = set()
        clean = []
        for p in parts:
            tokens = p.lower().split()
            for t in tokens:
                t = t.strip(".,;:()")
                if t and t not in seen:
                    seen.add(t)
                    clean.append(t)
        return " ".join(clean[:30])  # API performs best with focused queries

    @property
    def all_keywords(self) -> set[str]:
        """Full set of lowercase keywords for local scoring."""
        raw = (
            self.mission + " "
            + " ".join(self.focus_areas)
            + " ".join(self.populations)
            + " ".join(self.programs)
            + " ".join(self.keywords_extra)
        )
        return {w.strip(".,;:()").lower() for w in raw.split() if len(w) > 2}


def load_profile(path: str | Path) -> NonprofitProfile:
    """Load a nonprofit profile from a JSON file."""
    data = json.loads(Path(path).read_text())
    return _dict_to_profile(data)


def profile_from_dict(data: dict) -> NonprofitProfile:
    """Build a NonprofitProfile from a plain dictionary (e.g. from Streamlit form)."""
    return _dict_to_profile(data)


def _dict_to_profile(data: dict) -> NonprofitProfile:
    required = {"name", "mission", "focus_areas", "populations", "location_state"}
    missing  = required - data.keys()
    if missing:
        raise ValueError(f"Profile missing required fields: {missing}")

    return NonprofitProfile(
        name            = data["name"],
        mission         = data["mission"],
        focus_areas     = _as_list(data["focus_areas"]),
        populations     = _as_list(data["populations"]),
        location_state  = data["location_state"].upper(),
        location_city   = data.get("location_city"),
        org_type        = data.get("org_type", "Nonprofit"),
        tax_status      = data.get("tax_status", "501(c)(3)"),
        annual_budget   = data.get("annual_budget"),
        staff_count     = data.get("staff_count"),
        years_operating = data.get("years_operating"),
        programs        = _as_list(data.get("programs", [])),
        keywords_extra  = _as_list(data.get("keywords_extra", [])),
        funding_min     = data.get("funding_min"),
        funding_max     = data.get("funding_max"),
    )


def _as_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return []


def profile_summary(p: NonprofitProfile) -> str:
    """Return a plain-text summary of the profile (used for LLM prompts)."""
    lines = [
        f"Organization: {p.name}",
        f"Mission: {p.mission}",
        f"Focus areas: {', '.join(p.focus_areas)}",
        f"Populations served: {', '.join(p.populations)}",
        f"Location: {p.location_city + ', ' if p.location_city else ''}{p.location_state}",
        f"Tax status: {p.tax_status}",
    ]
    if p.programs:
        lines.append(f"Programs: {', '.join(p.programs)}")
    if p.annual_budget:
        lines.append(f"Annual budget: ${p.annual_budget:,}")
    if p.funding_min or p.funding_max:
        lo = f"${p.funding_min:,}" if p.funding_min else "any"
        hi = f"${p.funding_max:,}" if p.funding_max else "any"
        lines.append(f"Target award range: {lo} – {hi}")
    return "\n".join(lines)
