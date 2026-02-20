"""Tests for src.evaluation.metrics — IR metrics with textbook-verifiable values."""

import pytest

from src.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    revenue_weighted_hit_rate,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_partial_precision(self):
        assert precision_at_k([1, 4, 2], {1, 2, 3}, k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_list(self):
        # Only 2 items recommended, k=5 → denominator is still k
        assert precision_at_k([1, 2], {1, 2, 3}, k=5) == pytest.approx(2 / 5)

    def test_k_one(self):
        assert precision_at_k([1, 2, 3], {1}, k=1) == 1.0
        assert precision_at_k([2, 1, 3], {1}, k=1) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_partial_recall(self):
        assert recall_at_k([1, 4, 5], {1, 2, 3}, k=3) == pytest.approx(1 / 3)

    def test_empty_relevant_set(self):
        assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # All relevant items at the top
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_no_relevant_items(self):
        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_single_hit_at_position_1(self):
        # 1 relevant item at position 1: DCG = 1/log2(2) = 1.0, IDCG = 1.0
        assert ndcg_at_k([1, 4, 5], {1}, k=3) == pytest.approx(1.0)

    def test_single_hit_at_position_2(self):
        # 1 relevant at pos 2: DCG = 1/log2(3) ≈ 0.6309, IDCG = 1/log2(2) = 1.0
        expected = (1 / 1.5849625) / 1.0
        assert ndcg_at_k([4, 1, 5], {1}, k=3) == pytest.approx(expected, rel=1e-3)


class TestRevenueWeightedHitRate:
    def test_perfect_hits(self):
        revenues = {1: 100, 2: 200, 3: 300}
        result = revenue_weighted_hit_rate([1, 2, 3], {1, 2, 3}, revenues, k=3)
        assert result == pytest.approx(1.0)

    def test_zero_hits(self):
        revenues = {1: 100, 2: 200, 4: 50, 5: 50}
        result = revenue_weighted_hit_rate([4, 5], {1, 2}, revenues, k=2)
        assert result == 0.0

    def test_revenue_weighted(self):
        # Hit on item 2 (revenue 200), miss on item 1 (revenue 100)
        revenues = {1: 100, 2: 200}
        result = revenue_weighted_hit_rate([2, 3], {1, 2}, revenues, k=2)
        # Weighted hits: 200, total relevant revenue: 300
        assert result == pytest.approx(200 / 300)

    def test_empty_relevant(self):
        result = revenue_weighted_hit_rate([1, 2], set(), {1: 100}, k=2)
        assert result == 0.0

    def test_zero_revenue_relevant(self):
        revenues = {1: 0, 2: 0}
        result = revenue_weighted_hit_rate([1, 2], {1, 2}, revenues, k=2)
        assert result == 0.0
