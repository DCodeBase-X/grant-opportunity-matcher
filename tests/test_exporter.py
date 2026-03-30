"""
Unit tests for exporter.py — CSV and Markdown export.
"""

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exporter import to_csv_bytes, to_markdown_report
from src.grants_api import GrantOpportunity
from src.profile import NonprofitProfile


#  Fixtures 

def _make_profile(**kwargs) -> NonprofitProfile:
    defaults = dict(
        name="Test Org",
        mission="We provide reentry housing for justice-involved adults.",
        focus_areas=["reentry", "housing"],
        populations=["justice-involved individuals"],
        location_state="NC",
    )
    defaults.update(kwargs)
    return NonprofitProfile(**defaults)


def _make_grant(**kwargs) -> GrantOpportunity:
    defaults = dict(
        id="G-001", number="HUD-2024-001",
        title="Reentry Housing Program",
        agency="HUD", agency_code="HUD",
        status="posted", open_date="2024-01-01", close_date="2024-12-31",
        award_ceiling=500_000, award_floor=50_000, expected_awards=5,
        categories=["Housing"],
        eligible_types=["Nonprofits"],
        description="Supports transitional housing for returning citizens.",
        relevance_score=0.85,
        score_breakdown={"ai_rationale": "Strong alignment with reentry housing mission."},
    )
    defaults.update(kwargs)
    return GrantOpportunity(**defaults)


#  to_csv_bytes 

class TestToCsvBytes:
    def test_returns_bytes(self):
        result = to_csv_bytes([_make_grant()], _make_profile())
        assert isinstance(result, bytes)

    def test_utf8_decodable(self):
        result = to_csv_bytes([_make_grant()], _make_profile())
        result.decode("utf-8")  # should not raise

    def test_header_row_present(self):
        result = to_csv_bytes([_make_grant()], _make_profile())
        reader = csv.reader(io.StringIO(result.decode("utf-8")))
        header = next(reader)
        assert "Title" in header
        assert "Relevance Score" in header
        assert "Agency" in header

    def test_one_data_row_per_grant(self):
        grants = [_make_grant(id="A", number="A"), _make_grant(id="B", number="B")]
        result = to_csv_bytes(grants, _make_profile())
        reader = csv.reader(io.StringIO(result.decode("utf-8")))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 grants

    def test_rank_increments(self):
        grants = [_make_grant(id="A", number="A"), _make_grant(id="B", number="B")]
        result = to_csv_bytes(grants, _make_profile())
        reader = csv.reader(io.StringIO(result.decode("utf-8")))
        rows = list(reader)
        assert rows[1][0] == "1"
        assert rows[2][0] == "2"

    def test_relevance_score_formatted_as_percent(self):
        grant = _make_grant(relevance_score=0.75)
        result = to_csv_bytes([grant], _make_profile())
        reader = csv.reader(io.StringIO(result.decode("utf-8")))
        next(reader)  # skip header
        row = next(reader)
        assert "75%" in row[4]

    def test_ai_rationale_included(self):
        grant = _make_grant(score_breakdown={"ai_rationale": "Great match."})
        result = to_csv_bytes([grant], _make_profile())
        assert b"Great match." in result

    def test_empty_grant_list(self):
        result = to_csv_bytes([], _make_profile())
        reader = csv.reader(io.StringIO(result.decode("utf-8")))
        rows = list(reader)
        assert len(rows) == 1  # header only

    def test_missing_dates_empty_string(self):
        grant = _make_grant(open_date=None, close_date=None)
        result = to_csv_bytes([grant], _make_profile())
        reader = csv.reader(io.StringIO(result.decode("utf-8")))
        next(reader)
        row = next(reader)
        assert row[8] == ""   # open_date
        assert row[9] == ""   # close_date

    def test_description_truncated_to_300(self):
        long_desc = "x" * 500
        grant = _make_grant(description=long_desc)
        result = to_csv_bytes([grant], _make_profile())
        assert b"x" * 301 not in result

    def test_newlines_in_description_replaced(self):
        grant = _make_grant(description="line1\nline2\nline3")
        result = to_csv_bytes([grant], _make_profile())
        reader = csv.reader(io.StringIO(result.decode("utf-8")))
        next(reader)
        row = next(reader)
        desc_col = row[-1]
        assert "\n" not in desc_col

    def test_none_expected_awards_shows_na(self):
        grant = _make_grant(expected_awards=None)
        result = to_csv_bytes([grant], _make_profile())
        assert b"N/A" in result


#  to_markdown_report 

class TestToMarkdownReport:
    def test_returns_string(self):
        result = to_markdown_report([_make_grant()], _make_profile())
        assert isinstance(result, str)

    def test_contains_org_name(self):
        result = to_markdown_report([_make_grant()], _make_profile(name="Hope House"))
        assert "Hope House" in result

    def test_contains_grant_title(self):
        result = to_markdown_report([_make_grant(title="Special Housing Grant")], _make_profile())
        assert "Special Housing Grant" in result

    def test_contains_relevance_score(self):
        result = to_markdown_report([_make_grant(relevance_score=0.92)], _make_profile())
        assert "92%" in result

    def test_contains_ai_rationale(self):
        grant = _make_grant(score_breakdown={"ai_rationale": "Excellent alignment."})
        result = to_markdown_report([grant], _make_profile())
        assert "Excellent alignment." in result

    def test_no_rationale_section_when_empty(self):
        grant = _make_grant(score_breakdown={})
        result = to_markdown_report([grant], _make_profile())
        assert "> " not in result

    def test_top_n_limits_output(self):
        grants = [_make_grant(id=str(i), number=str(i), title=f"Grant {i}") for i in range(10)]
        result = to_markdown_report(grants, _make_profile(), top_n=3)
        assert "Grant 0" in result
        assert "Grant 1" in result
        assert "Grant 2" in result
        assert "Grant 3" not in result

    def test_description_truncated_to_400(self):
        long_desc = "y" * 600
        grant = _make_grant(description=long_desc)
        result = to_markdown_report([grant], _make_profile())
        assert "y" * 401 not in result
        assert "…" in result

    def test_short_description_no_ellipsis(self):
        grant = _make_grant(description="Short desc.")
        result = to_markdown_report([grant], _make_profile())
        assert "…" not in result

    def test_profile_section_present(self):
        result = to_markdown_report([_make_grant()], _make_profile())
        assert "## Profile Used" in result
        assert "reentry" in result

    def test_empty_grant_list(self):
        result = to_markdown_report([], _make_profile())
        assert "## Profile Used" in result
        assert "Top 0 Matches" in result

    def test_award_range_formatted(self):
        grant = _make_grant(award_floor=50_000, award_ceiling=500_000)
        result = to_markdown_report([grant], _make_profile())
        assert "$50,000" in result
        assert "$500,000" in result

    def test_deadline_shows_rolling_when_none(self):
        grant = _make_grant(close_date=None)
        result = to_markdown_report([grant], _make_profile())
        assert "Open / Rolling" in result
