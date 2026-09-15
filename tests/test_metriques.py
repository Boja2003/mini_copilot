"""Tests des metriques d'evaluation.

Une metrique fausse est pire qu'une absence de metrique : elle donne une
fausse confiance chiffree. D'ou des tests sur des cas calcules a la main.
"""

import pytest

from eval.metriques import hit_at_k, moyenne, precision_at_k, reciprocal_rank


def test_hit_at_k() -> None:
    assert hit_at_k([False, False, True], k=3) == 1.0
    assert hit_at_k([False, False, True], k=2) == 0.0
    assert hit_at_k([], k=5) == 0.0


def test_reciprocal_rank_recompense_la_tete() -> None:
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False, False, False, True]) == pytest.approx(0.2)
    assert reciprocal_rank([False, False]) == 0.0


def test_precision_at_k_mesure_l_encombrement() -> None:
    assert precision_at_k([True, False, False, False, False], k=5) == pytest.approx(0.2)
    assert precision_at_k([True, True, True, True, True], k=5) == 1.0
    # Moins de resultats que k : les places vides comptent comme non pertinentes.
    assert precision_at_k([True], k=5) == pytest.approx(0.2)


def test_precision_at_k_refuse_k_nul() -> None:
    with pytest.raises(ValueError):
        precision_at_k([True], k=0)


def test_moyenne() -> None:
    assert moyenne([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert moyenne([]) == 0.0
