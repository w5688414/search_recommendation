# search_recommendation

This repository now includes a minimal **RankMixer** model for ecommerce ranking that combines CTR and CVR predictions into a single ranking score.

## RankMixer

- Implementation: `/home/runner/work/search_recommendation/search_recommendation/search_recommendation/rank_mixer.py`
- Tests: `/home/runner/work/search_recommendation/search_recommendation/tests/test_rank_mixer.py`

The model uses a weighted geometric blend of CTR/CVR probabilities:

- higher CTR and higher CVR both increase final rank score
- invalid probabilities are rejected
- candidate lists can be ranked by the computed `rank_mixer_score`