"""RankMixer model for balancing CTR/CVR in ecommerce ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Iterable, Mapping, TypedDict

CandidateValue = str | int | float | None


class RankedCandidate(TypedDict, total=False):
    ctr: float
    cvr: float
    sku: str
    rank_mixer_score: float


@dataclass(frozen=True)
class RankMixerConfig:
    """Configuration for CTR/CVR score mixing."""

    ctr_weight: float = 0.5
    cvr_weight: float = 0.5
    epsilon: float = 1e-12

    def normalized_weights(self) -> tuple[float, float]:
        if not isfinite(self.ctr_weight) or not isfinite(self.cvr_weight):
            raise ValueError("ctr_weight and cvr_weight must be finite numbers")
        if self.ctr_weight < 0 or self.cvr_weight < 0:
            raise ValueError("ctr_weight and cvr_weight must be non-negative")
        total = self.ctr_weight + self.cvr_weight
        if total <= 0:
            raise ValueError("ctr_weight + cvr_weight must be positive")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        return self.ctr_weight / total, self.cvr_weight / total


class RankMixer:
    """Mixes CTR/CVR predictions into a single ranking score."""

    def __init__(self, config: RankMixerConfig | None = None) -> None:
        self._config = config or RankMixerConfig()
        self._ctr_weight, self._cvr_weight = self._config.normalized_weights()

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}")

    @staticmethod
    def _to_float(name: str, value: CandidateValue) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric, got {value!r}") from None

    def score(self, ctr: CandidateValue, cvr: CandidateValue) -> float:
        """Weighted geometric blend of CTR and CVR probabilities."""
        ctr = self._to_float("ctr", ctr)
        cvr = self._to_float("cvr", cvr)
        self._validate_probability("ctr", ctr)
        self._validate_probability("cvr", cvr)
        ctr = max(ctr, self._config.epsilon)
        cvr = max(cvr, self._config.epsilon)
        return exp(self._ctr_weight * log(ctr) + self._cvr_weight * log(cvr))

    def rank(
        self, candidates: Iterable[Mapping[str, CandidateValue]]
    ) -> list[RankedCandidate]:
        """Return enriched candidates sorted by rank_mixer_score (desc)."""
        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            ctr = self._to_float("ctr", candidate["ctr"])
            cvr = self._to_float("cvr", candidate["cvr"])
            enriched: RankedCandidate = dict(candidate)
            enriched["rank_mixer_score"] = self.score(ctr=ctr, cvr=cvr)
            ranked.append(enriched)
        ranked.sort(key=lambda item: item["rank_mixer_score"], reverse=True)
        return ranked
