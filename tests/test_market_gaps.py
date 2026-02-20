"""Tests for src.models.market_gaps — niche scoring and revenue estimation."""

import pandas as pd
import pytest

from src.models.market_gaps import (
    build_opportunity_table,
    compute_recency_trend,
    estimate_new_entrant_revenue,
    score_niches,
)


class TestScoreNiches:
    def _niche_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"niche": "Action + RPG", "supply": 100, "demand_proxy": 500_000,
             "engagement": 120, "satisfaction": 0.85, "median_revenue": 15_000,
             "revenue_p25": 5_000, "revenue_p75": 30_000, "avg_price": 15.0},
            {"niche": "Indie + Casual", "supply": 500, "demand_proxy": 200_000,
             "engagement": 60, "satisfaction": 0.70, "median_revenue": 3_000,
             "revenue_p25": 1_000, "revenue_p75": 8_000, "avg_price": 5.0},
            {"niche": "Strategy + Sim", "supply": 50, "demand_proxy": 300_000,
             "engagement": 200, "satisfaction": 0.90, "median_revenue": 20_000,
             "revenue_p25": 8_000, "revenue_p75": 50_000, "avg_price": 20.0},
        ])

    def test_adds_opportunity_score(self):
        result = score_niches(self._niche_df())
        assert "opportunity_score" in result.columns

    def test_scores_non_negative(self):
        result = score_niches(self._niche_df())
        assert (result["opportunity_score"] >= 0).all()

    def test_ranked_descending(self):
        result = score_niches(self._niche_df())
        scores = result["opportunity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_column_one_indexed(self):
        result = score_niches(self._niche_df())
        assert result["rank"].iloc[0] == 1
        assert result["rank"].iloc[-1] == len(result)

    def test_revenue_estimates_carried_through(self):
        result = score_niches(self._niche_df())
        assert "revenue_estimate_low" in result.columns
        assert "revenue_estimate_high" in result.columns

    def test_single_niche(self):
        """Edge case: only one niche means normalisation ranges are zero."""
        df = pd.DataFrame([{
            "niche": "Solo", "supply": 10, "demand_proxy": 100_000,
            "engagement": 90, "satisfaction": 0.75, "median_revenue": 10_000,
            "revenue_p25": 5_000, "revenue_p75": 20_000, "avg_price": 12.0,
        }])
        result = score_niches(df)
        assert len(result) == 1
        assert result["opportunity_score"].iloc[0] >= 0


class TestEstimateNewEntrantRevenue:
    def test_returns_expected_keys(self):
        row = pd.Series({
            "niche": "Test Niche", "supply": 50, "demand_proxy": 100_000,
            "engagement": 100, "satisfaction": 0.8,
            "revenue_estimate_low": 5_000, "median_revenue": 15_000,
            "revenue_estimate_high": 30_000, "avg_price": 12.0,
        })
        result = estimate_new_entrant_revenue(row)
        assert result["niche"] == "Test Niche"
        assert result["revenue_low"] == 5_000
        assert result["revenue_mid"] == 15_000
        assert result["revenue_high"] == 30_000
        assert "assumptions" in result

    def test_recency_multiplier_applied(self):
        row = pd.Series({
            "niche": "Test", "supply": 10, "demand_proxy": 50_000,
            "engagement": 80, "satisfaction": 0.7,
            "revenue_estimate_low": 1_000, "median_revenue": 5_000,
            "revenue_estimate_high": 10_000, "avg_price": 10.0,
        })
        result = estimate_new_entrant_revenue(row, recency_multiplier=2.0)
        assert result["revenue_low"] == pytest.approx(2_000)
        assert result["revenue_mid"] == pytest.approx(10_000)
        assert result["revenue_high"] == pytest.approx(20_000)

    def test_ordering_low_mid_high(self):
        row = pd.Series({
            "niche": "Test", "supply": 10, "demand_proxy": 50_000,
            "engagement": 80, "satisfaction": 0.7,
            "revenue_estimate_low": 1_000, "median_revenue": 5_000,
            "revenue_estimate_high": 10_000, "avg_price": 10.0,
        })
        result = estimate_new_entrant_revenue(row)
        assert result["revenue_low"] <= result["revenue_mid"] <= result["revenue_high"]


class TestBuildOpportunityTable:
    def test_limits_to_top_n(self):
        scored = score_niches(pd.DataFrame([
            {"niche": f"Niche {i}", "supply": 10 + i, "demand_proxy": 100_000 - i * 1000,
             "engagement": 100, "satisfaction": 0.8, "median_revenue": 10_000,
             "revenue_p25": 5_000, "revenue_p75": 20_000, "avg_price": 12.0}
            for i in range(10)
        ]))
        table = build_opportunity_table(scored, top_n=3)
        assert len(table) == 3

    def test_columns_renamed_for_presentation(self):
        scored = score_niches(pd.DataFrame([
            {"niche": "Test", "supply": 10, "demand_proxy": 100_000,
             "engagement": 100, "satisfaction": 0.8, "median_revenue": 10_000,
             "revenue_p25": 5_000, "revenue_p75": 20_000, "avg_price": 12.0}
        ]))
        table = build_opportunity_table(scored)
        assert "# Games" in table.columns
        assert "Opportunity Score" in table.columns


class TestComputeRecencyTrend:
    def test_returns_one_when_insufficient_data(self):
        games = pd.DataFrame([
            {"tags": ["Action", "RPG"], "released": "2024-01-01", "estimated_revenue": 1000},
        ])
        result = compute_recency_trend(games, ["Action", "RPG"])
        assert result == 1.0

    def test_returns_one_when_no_tag_column(self):
        games = pd.DataFrame([{"released": "2024-01-01", "estimated_revenue": 1000}])
        result = compute_recency_trend(games, ["Action"])
        assert result == 1.0

    def test_recent_outperformance(self):
        old = [{"tags": ["A", "B"], "released": "2018-01-01", "estimated_revenue": 100}] * 5
        new = [{"tags": ["A", "B"], "released": "2025-01-01", "estimated_revenue": 300}] * 5
        games = pd.DataFrame(old + new)
        result = compute_recency_trend(games, ["A", "B"], recent_years=3)
        assert result == pytest.approx(3.0)
