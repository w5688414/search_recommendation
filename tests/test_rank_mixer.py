import unittest

from search_recommendation.rank_mixer import RankMixer, RankMixerConfig


class RankMixerTests(unittest.TestCase):
    def test_score_increases_with_higher_cvr(self) -> None:
        mixer = RankMixer()
        low_cvr = mixer.score(ctr=0.2, cvr=0.05)
        high_cvr = mixer.score(ctr=0.2, cvr=0.2)
        self.assertGreater(high_cvr, low_cvr)

    def test_ranking_uses_mixed_ctr_cvr_score(self) -> None:
        mixer = RankMixer(RankMixerConfig(ctr_weight=0.4, cvr_weight=0.6))
        ranked = mixer.rank(
            [
                {"sku": "high_ctr_low_cvr", "ctr": 0.9, "cvr": 0.02},
                {"sku": "balanced", "ctr": 0.5, "cvr": 0.2},
                {"sku": "low_ctr_high_cvr", "ctr": 0.2, "cvr": 0.35},
            ]
        )
        self.assertEqual(ranked[0]["sku"], "balanced")
        self.assertGreater(ranked[0]["rank_mixer_score"], ranked[1]["rank_mixer_score"])

    def test_probability_validation(self) -> None:
        mixer = RankMixer()
        with self.assertRaises(ValueError):
            mixer.score(ctr=1.1, cvr=0.2)
        with self.assertRaises(ValueError):
            mixer.score(ctr=0.2, cvr=-0.1)
        with self.assertRaisesRegex(ValueError, "finite number"):
            mixer.score(ctr=float("nan"), cvr=0.1)
        with self.assertRaisesRegex(ValueError, "ctr must be numeric"):
            mixer.score(ctr="not-a-number", cvr=0.1)
        with self.assertRaisesRegex(ValueError, "ctr must be numeric"):
            mixer.score(ctr=True, cvr=0.1)

    def test_weights_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            RankMixer(RankMixerConfig(ctr_weight=0.0, cvr_weight=0.0))
        with self.assertRaises(ValueError):
            RankMixer(RankMixerConfig(ctr_weight=-0.1, cvr_weight=0.5))

    def test_rank_rejects_non_numeric_candidate_probability(self) -> None:
        mixer = RankMixer()
        with self.assertRaisesRegex(ValueError, "ctr must be numeric"):
            mixer.rank([{"sku": "bad", "ctr": "NaN-ish", "cvr": 0.2}])
        with self.assertRaisesRegex(ValueError, "ctr must be numeric"):
            mixer.rank([{"sku": "bad", "ctr": True, "cvr": 0.2}])

    def test_rank_rejects_nan_candidate_probability(self) -> None:
        mixer = RankMixer()
        with self.assertRaisesRegex(ValueError, "finite number"):
            mixer.rank([{"sku": "bad", "ctr": float("nan"), "cvr": 0.2}])


if __name__ == "__main__":
    unittest.main()
