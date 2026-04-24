"""Tests for neural_memory.retrieval.hybrid_ranker — Multi-signal ranking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from neural_memory.config import RankerConfig
from neural_memory.core.neuron import NeuronFactory
from neural_memory.retrieval.hybrid_ranker import HybridRanker, RankedResult


def _make_neuron(content="test", strength=0.5, hours_ago=0):
    n = NeuronFactory.create(content=content)
    n.strength = strength
    n.last_accessed = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return n


class TestHybridRankerRank:
    def test_empty_candidates(self):
        ranker = HybridRanker()
        results = ranker.rank(candidates={})
        assert results == []

    def test_single_candidate(self):
        n = _make_neuron("single")
        ranker = HybridRanker()
        results = ranker.rank(
            candidates={n.id: n},
            semantic_scores={n.id: 0.9},
        )
        assert len(results) == 1
        assert results[0].neuron.id == n.id
        assert results[0].semantic_score == pytest.approx(0.9)

    def test_ranking_order_by_semantic(self):
        cfg = RankerConfig(
            semantic_weight=1.0,
            fts_weight=0.0,
            activation_weight=0.0,
            recency_weight=0.0,
            strength_weight=0.0,
        )
        n1 = _make_neuron("high")
        n2 = _make_neuron("low")
        ranker = HybridRanker(cfg)
        results = ranker.rank(
            candidates={n1.id: n1, n2.id: n2},
            semantic_scores={n1.id: 0.9, n2.id: 0.3},
        )
        assert results[0].neuron.id == n1.id
        assert results[1].neuron.id == n2.id

    def test_ranking_order_by_activation(self):
        cfg = RankerConfig(
            semantic_weight=0.0,
            fts_weight=0.0,
            activation_weight=1.0,
            recency_weight=0.0,
            strength_weight=0.0,
        )
        n1 = _make_neuron("a")
        n2 = _make_neuron("b")
        ranker = HybridRanker(cfg)
        results = ranker.rank(
            candidates={n1.id: n1, n2.id: n2},
            activation_scores={n1.id: 0.3, n2.id: 0.9},
        )
        assert results[0].neuron.id == n2.id

    def test_top_k_limit(self):
        ranker = HybridRanker()
        candidates = {}
        for i in range(10):
            n = _make_neuron(f"n{i}")
            candidates[n.id] = n
        results = ranker.rank(candidates=candidates, top_k=3)
        assert len(results) == 3

    def test_recency_favors_recent(self):
        cfg = RankerConfig(
            semantic_weight=0.0,
            fts_weight=0.0,
            activation_weight=0.0,
            recency_weight=1.0,
            strength_weight=0.0,
        )
        recent = _make_neuron("recent", hours_ago=1)
        old = _make_neuron("old", hours_ago=7 * 24)  # 7 days ago
        ranker = HybridRanker(cfg)
        results = ranker.rank(candidates={recent.id: recent, old.id: old})
        assert results[0].neuron.id == recent.id
        assert results[0].recency_score > results[1].recency_score

    def test_strength_component(self):
        cfg = RankerConfig(
            semantic_weight=0.0,
            fts_weight=0.0,
            activation_weight=0.0,
            recency_weight=0.0,
            strength_weight=1.0,
        )
        strong = _make_neuron("strong", strength=0.9)
        weak = _make_neuron("weak", strength=0.1)
        ranker = HybridRanker(cfg)
        results = ranker.rank(candidates={strong.id: strong, weak.id: weak})
        assert results[0].neuron.id == strong.id

    def test_missing_scores_default_zero(self):
        ranker = HybridRanker()
        n = _make_neuron("no scores")
        results = ranker.rank(candidates={n.id: n})
        assert len(results) == 1
        assert results[0].semantic_score == 0.0
        assert results[0].fts_score == 0.0
        assert results[0].activation_score == 0.0

    def test_semantic_score_clamped(self):
        ranker = HybridRanker()
        n = _make_neuron("clamp")
        results = ranker.rank(
            candidates={n.id: n},
            semantic_scores={n.id: 1.5},  # Above 1.0
        )
        assert results[0].semantic_score == 1.0

    def test_fts_normalization(self):
        """FTS scores normalized by max value."""
        cfg = RankerConfig(
            semantic_weight=0.0,
            fts_weight=1.0,
            activation_weight=0.0,
            recency_weight=0.0,
            strength_weight=0.0,
        )
        n1 = _make_neuron("a")
        n2 = _make_neuron("b")
        ranker = HybridRanker(cfg)
        results = ranker.rank(
            candidates={n1.id: n1, n2.id: n2},
            fts_scores={n1.id: 10.0, n2.id: 5.0},
        )
        # n1 has higher raw FTS → should be first
        assert results[0].neuron.id == n1.id
        # After normalization: n1=1.0, n2=0.5
        assert results[0].fts_score == pytest.approx(1.0)
        assert results[1].fts_score == pytest.approx(0.5)


class TestRankedResult:
    def test_to_dict(self):
        n = _make_neuron("dict test")
        rr = RankedResult(
            neuron=n,
            final_score=0.85,
            semantic_score=0.9,
            fts_score=0.5,
            activation_score=0.3,
            recency_score=0.7,
            strength_score=0.4,
        )
        d = rr.to_dict()
        assert d["score"] == 0.85
        assert "score_breakdown" in d
        assert d["score_breakdown"]["semantic"] == 0.9
        assert d["content"] == "dict test"


class TestRecencyScore:
    def test_just_accessed_is_one(self):
        now = datetime.now(timezone.utc)
        n = _make_neuron("fresh")
        n.last_accessed = now
        score = HybridRanker._recency_score(n, now)
        assert score == pytest.approx(1.0)

    def test_seven_days_is_half(self):
        now = datetime.now(timezone.utc)
        n = _make_neuron("week old")
        n.last_accessed = now - timedelta(days=7)
        score = HybridRanker._recency_score(n, now)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_old_memory_low_score(self):
        now = datetime.now(timezone.utc)
        n = _make_neuron("ancient")
        n.last_accessed = now - timedelta(days=60)
        score = HybridRanker._recency_score(n, now)
        assert score < 0.01
