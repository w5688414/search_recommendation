"""RankMixer model for balancing CTR/CVR in ecommerce ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable, Mapping

CandidateValue = str | int | float | bool | None


@dataclass(frozen=True)
class RankMixerConfig:
    """Configuration for CTR/CVR score mixing."""

    ctr_weight: float = 0.5
    cvr_weight: float = 0.5
    epsilon: float = 1e-12

    def normalized_weights(self) -> tuple[float, float]:
        total = self.ctr_weight + self.cvr_weight
        if total <= 0:
            raise ValueError("ctr_weight + cvr_weight must be positive")
        return self.ctr_weight / total, self.cvr_weight / total


class RankMixer:
    """Mixes CTR/CVR predictions into a single ranking score."""

    def __init__(self, config: RankMixerConfig | None = None) -> None:
        self._config = config or RankMixerConfig()
        self._ctr_weight, self._cvr_weight = self._config.normalized_weights()

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}")

    def score(self, ctr: float, cvr: float) -> float:
        """Weighted geometric blend of CTR and CVR probabilities."""
        self._validate_probability("ctr", ctr)
        self._validate_probability("cvr", cvr)
        ctr = max(ctr, self._config.epsilon)
        cvr = max(cvr, self._config.epsilon)
        return exp(self._ctr_weight * log(ctr) + self._cvr_weight * log(cvr))

    def rank(
        self, candidates: Iterable[Mapping[str, CandidateValue]]
    ) -> list[dict[str, CandidateValue]]:
        """Return candidates sorted by RankMixer score in descending order."""
        ranked: list[dict[str, float]] = []
        for candidate in candidates:
            ctr = float(candidate["ctr"])
            cvr = float(candidate["cvr"])
            enriched = dict(candidate)
            enriched["rank_mixer_score"] = self.score(ctr=ctr, cvr=cvr)
            ranked.append(enriched)
        ranked.sort(key=lambda item: item["rank_mixer_score"], reverse=True)
        return ranked
