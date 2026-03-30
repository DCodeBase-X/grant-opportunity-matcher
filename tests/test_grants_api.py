"""
Unit tests for grants_api.py — record parsing and search_grants error handling.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grants_api import (
    GrantOpportunity,
    _parse_opportunity,
    search_grants,
    MOCK_GRANTS,
)


#  _parse_opportunity 

class TestParseOpportunity:
    def _raw(self, **overrides) -> dict:
        base = {
            "id": 12345,
            "number": "HUD-2024-001",
            "title": "Reentry Housing Program",
            "agencyName": "Dept. of Housing",
            "agencyCode": "HUD",
            "opportunityStatus": "posted",
            "openDate": "2024-01-01",
            "closeDate": "2024-12-31",
            "awardCeiling": 500_000,
            "awardFloor": 50_000,
            "expectedNumberOfAwards": 5,
            "categoryOfFundingActivity": ["Housing"],
            "eligibleApplicants": ["Nonprofits"],
            "synopsis": "Transitional housing for returning citizens.",
        }
        base.update(overrides)
        return base

    def test_returns_grant_opportunity(self):
        result = _parse_opportunity(self._raw())
        assert isinstance(result, GrantOpportunity)

    def test_id_cast_to_string(self):
        result = _parse_opportunity(self._raw(id=99999))
        assert result.id == "99999"
        assert isinstance(result.id, str)

    def test_basic_fields_mapped(self):
        result = _parse_opportunity(self._raw())
        assert result.number == "HUD-2024-001"
        assert result.title == "Reentry Housing Program"
        assert result.agency == "Dept. of Housing"
        assert result.agency_code == "HUD"
        assert result.status == "posted"
        assert result.open_date == "2024-01-01"
        assert result.close_date == "2024-12-31"
        assert result.award_ceiling == 500_000
        assert result.award_floor == 50_000
        assert result.expected_awards == 5

    def test_categories_as_list(self):
        result = _parse_opportunity(self._raw(categoryOfFundingActivity=["Housing", "Health"]))
        assert result.categories == ["Housing", "Health"]

    def test_categories_as_string_wrapped_in_list(self):
        result = _parse_opportunity(self._raw(categoryOfFundingActivity="Housing"))
        assert result.categories == ["Housing"]

    def test_categories_none_defaults_to_empty(self):
        result = _parse_opportunity(self._raw(categoryOfFundingActivity=None))
        assert result.categories == []

    def test_eligible_types_as_list(self):
        result = _parse_opportunity(self._raw(eligibleApplicants=["Nonprofits", "Tribes"]))
        assert result.eligible_types == ["Nonprofits", "Tribes"]

    def test_eligible_types_as_string_wrapped(self):
        result = _parse_opportunity(self._raw(eligibleApplicants="Nonprofits"))
        assert result.eligible_types == ["Nonprofits"]

    def test_eligible_types_none_defaults_to_empty(self):
        result = _parse_opportunity(self._raw(eligibleApplicants=None))
        assert result.eligible_types == []

    def test_synopsis_used_as_description(self):
        result = _parse_opportunity(self._raw(synopsis="Synopsis text."))
        assert result.description == "Synopsis text."

    def test_description_fallback_when_no_synopsis(self):
        raw = self._raw()
        del raw["synopsis"]
        raw["description"] = "Fallback description."
        result = _parse_opportunity(raw)
        assert result.description == "Fallback description."

    def test_empty_description_when_neither_present(self):
        raw = self._raw()
        del raw["synopsis"]
        result = _parse_opportunity(raw)
        assert result.description == ""

    def test_missing_optional_fields_are_none(self):
        raw = self._raw()
        del raw["openDate"]
        del raw["closeDate"]
        del raw["awardCeiling"]
        del raw["awardFloor"]
        del raw["expectedNumberOfAwards"]
        result = _parse_opportunity(raw)
        assert result.open_date is None
        assert result.close_date is None
        assert result.award_ceiling is None
        assert result.award_floor is None
        assert result.expected_awards is None

    def test_missing_title_defaults_to_untitled(self):
        raw = self._raw()
        del raw["title"]
        result = _parse_opportunity(raw)
        assert result.title == "Untitled"

    def test_relevance_score_defaults_to_zero(self):
        result = _parse_opportunity(self._raw())
        assert result.relevance_score == 0.0

    def test_score_breakdown_defaults_to_empty_dict(self):
        result = _parse_opportunity(self._raw())
        assert result.score_breakdown == {}


# ── GrantOpportunity properties ───────────────────────────────────────────────

class TestGrantOpportunityProperties:
    def _grant(self, **kwargs) -> GrantOpportunity:
        defaults = dict(
            id="1", number="X", title="T", agency="A", agency_code="A",
            status="posted", open_date=None, close_date=None,
            award_ceiling=None, award_floor=None, expected_awards=None,
        )
        defaults.update(kwargs)
        return GrantOpportunity(**defaults)

    def test_award_ceiling_fmt_with_value(self):
        g = self._grant(award_ceiling=500_000)
        assert g.award_ceiling_fmt == "$500,000"

    def test_award_ceiling_fmt_none(self):
        g = self._grant(award_ceiling=None)
        assert g.award_ceiling_fmt == "Not specified"

    def test_award_floor_fmt_with_value(self):
        g = self._grant(award_floor=50_000)
        assert g.award_floor_fmt == "$50,000"

    def test_award_floor_fmt_none(self):
        g = self._grant(award_floor=None)
        assert g.award_floor_fmt == "Not specified"

    def test_deadline_display_with_date(self):
        g = self._grant(close_date="2024-12-31")
        assert g.deadline_display == "2024-12-31"

    def test_deadline_display_none_is_rolling(self):
        g = self._grant(close_date=None)
        assert g.deadline_display == "Open / Rolling"


#  search_grants error handling 

class TestSearchGrantsErrors:
    def test_timeout_retries_and_raises(self):
        import requests
        with patch("src.grants_api.requests.post", side_effect=requests.exceptions.Timeout):
            with pytest.raises(RuntimeError, match="timed out"):
                search_grants("housing", max_results=5)

    def test_http_error_non_retryable_raises(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        err = requests.exceptions.HTTPError(response=mock_resp)
        with patch("src.grants_api.requests.post", side_effect=err):
            with pytest.raises(RuntimeError, match="Grants.gov API error"):
                search_grants("housing", max_results=5)

    def test_unexpected_error_raises(self):
        with patch("src.grants_api.requests.post", side_effect=ValueError("bad")):
            with pytest.raises(RuntimeError, match="Unexpected error"):
                search_grants("housing", max_results=5)

    def test_returns_list_of_grant_opportunities(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "oppHits": [
                    {
                        "id": 1, "number": "X", "title": "Test Grant",
                        "agencyName": "HUD", "agencyCode": "HUD",
                        "opportunityStatus": "posted",
                        "openDate": None, "closeDate": None,
                        "awardCeiling": None, "awardFloor": None,
                        "expectedNumberOfAwards": None,
                    }
                ]
            }
        }
        with patch("src.grants_api.requests.post", return_value=mock_resp):
            results = search_grants("housing", max_results=5)
        assert len(results) == 1
        assert isinstance(results[0], GrantOpportunity)
        assert results[0].title == "Test Grant"

    def test_empty_results_returns_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": {"oppHits": []}}
        with patch("src.grants_api.requests.post", return_value=mock_resp):
            results = search_grants("housing", max_results=5)
        assert results == []


#  MOCK_GRANTS  for testing and demos

class TestMockGrants:
    def test_mock_grants_is_list(self):
        assert isinstance(MOCK_GRANTS, list)

    def test_all_mock_grants_are_grant_opportunities(self):
        assert all(isinstance(g, GrantOpportunity) for g in MOCK_GRANTS)

    def test_mock_grants_have_descriptions(self):
        assert all(len(g.description) > 0 for g in MOCK_GRANTS)

    def test_mock_grants_have_unique_ids(self):
        ids = [g.id for g in MOCK_GRANTS]
        assert len(ids) == len(set(ids))
