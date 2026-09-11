# search_recommendation

This repository now includes a minimal **RankMixer** model for ecommerce ranking that combines CTR and CVR predictions into a single ranking score.

## RankMixer

- Implementation: `search_recommendation/rank_mixer.py`
- Tests: `tests/test_rank_mixer.py`

The model uses a weighted geometric blend of CTR/CVR probabilities:

- higher CTR and higher CVR both increase final rank score
- zero-valued CTR/CVR inputs are clamped to a tiny `epsilon` value before log-space mixing
- invalid probabilities are rejected
- candidate lists can be ranked by the computed `rank_mixer_score`