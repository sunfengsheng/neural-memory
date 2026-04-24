"""Hybrid ranker: combines semantic, FTS, activation, recency, and strength scores.

Final score formula:
    score = w_sem * semantic + w_fts * fts + w_act * activation
          + w_rec * recency + w_str * strength

All sub-scores normalized to [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from neural_memory.config import RankerConfig
from neural_memory.core.neuron import Neuron


@dataclass
class RankedResult:
    """A single ranked search result with score breakdown."""
    neuron: Neuron
    final_score: float
    semantic_score: float = 0.0
    fts_score: float = 0.0
    activation_score: float = 0.0
    recency_score: float = 0.0
    strength_score: float = 0.0

    def to_dict(self) -> dict:
        d = self.neuron.to_dict()
        d["score"] = round(self.final_score, 4)
        d["score_breakdown"] = {
            "semantic": round(self.semantic_score, 4),
            "fts": round(self.fts_score, 4),
            "activation": round(self.activation_score, 4),
            "recency": round(self.recency_score, 4),
            "strength": round(self.strength_score, 4),
        }
        return d


class HybridRanker:
    """Combines multiple ranking signals into a single score."""

    def __init__(self, config: RankerConfig | None = None):
        self._cfg = config or RankerConfig()

    def rank(
        self,
        candidates: dict[str, Neuron],
        semantic_scores: dict[str, float] | None = None,
        fts_scores: dict[str, float] | None = None,
        activation_scores: dict[str, float] | None = None,
        top_k: int = 20,
        now: datetime | None = None,
    ) -> list[RankedResult]:
        """Rank candidate neurons by combined score.

        Args:
            candidates: neuron_id -> Neuron mapping (the union of all candidates).
            semantic_scores: neuron_id -> cosine similarity [0, 1].
            fts_scores: neuron_id -> FTS rank score (will be normalized).
            activation_scores: neuron_id -> spreading activation value.
            top_k: Return top K results.
            now: Reference time for recency scoring.

        Returns:
            Sorted list of RankedResult, highest score first.
        """
        if not candidates:
            return []

        if now is None:
            now = datetime.now(timezone.utc)

        semantic_scores = semantic_scores or {}
        fts_scores = fts_scores or {}
        activation_scores = activation_scores or {}

        # Normalize activation scores to [0, 1] via max
        max_act = max(activation_scores.values(), default=1.0) or 1.0

        # Normalize FTS scores (bm25 is negative; larger magnitude = worse in sqlite)
        # fts_scores here is assumed to be a "goodness" score in [0,1] already,
        # caller should convert if needed.
        max_fts = max(fts_scores.values(), default=1.0) or 1.0

        results: list[RankedResult] = []

        for nid, neuron in candidates.items():
            sem = semantic_scores.get(nid, 0.0)
            # Clamp semantic similarity into [0, 1]
            sem = max(0.0, min(1.0, sem))

            fts_raw = fts_scores.get(nid, 0.0)
            fts = fts_raw / max_fts if max_fts > 0 else 0.0
            fts = max(0.0, min(1.0, fts))

            act_raw = activation_scores.get(nid, 0.0)
            act = act_raw / max_act if max_act > 0 else 0.0
            act = max(0.0, min(1.0, act))

            recency = self._recency_score(neuron, now)
            strength = max(0.0, min(1.0, neuron.strength))

            final = (
                self._cfg.semantic_weight * sem
                + self._cfg.fts_weight * fts
                + self._cfg.activation_weight * act
                + self._cfg.recency_weight * recency
                + self._cfg.strength_weight * strength
            )

            results.append(
                RankedResult(
                    neuron=neuron,
                    final_score=final,
                    semantic_score=sem,
                    fts_score=fts,
                    activation_score=act,
                    recency_score=recency,
                    strength_score=strength,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _recency_score(neuron: Neuron, now: datetime) -> float:
        """Compute a recency score in [0, 1] based on last_accessed.

        Uses exponential decay with ~7-day half-life.
        """
        hours = (now - neuron.last_accessed).total_seconds() / 3600.0
        if hours <= 0:
            return 1.0
        # Half-life ~ 168 hours (7 days)
        decay_lambda = math.log(2) / 168.0
        return math.exp(-decay_lambda * hours)
