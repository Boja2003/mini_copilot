"""Tests de la synthese de mesures repetees.

Elle sert a decider si un ecart entre deux variantes est reel : une erreur
ici ferait passer du bruit pour un resultat.
"""

import pytest

from eval.synthese import resumer, se_chevauchent, synthetiser, verifier_comparables


def _resultat(nom: str, mrr: float, empreinte: str | None = "abc") -> dict:
    agregat = {"n": 1, "hit@5": 1.0, "mrr": mrr, "precision@5": 0.5}
    return {
        "nom": nom,
        "golden_set": empreinte,
        "latence_ms": {"mediane": 900, "p90": 1500},
        "agregats": {c: agregat for c in ["TOUTES", "fr_eloigne", "fr_proche", "en"]},
    }


def test_resumer() -> None:
    assert resumer([0.9, 0.8, 1.0]) == {
        "moyenne": pytest.approx(0.9),
        "min": 0.8,
        "max": 1.0,
        "n": 3,
    }
    assert resumer([]) is None


def test_etendues_qui_se_recouvrent() -> None:
    assert se_chevauchent({"min": 0.90, "max": 0.95}, {"min": 0.94, "max": 0.97})
    # Bornes egales : on ne peut pas departager.
    assert se_chevauchent({"min": 0.90, "max": 0.94}, {"min": 0.94, "max": 0.97})


def test_etendues_disjointes() -> None:
    assert not se_chevauchent({"min": 0.75, "max": 0.78}, {"min": 0.94, "max": 0.97})


def test_synthese_sur_repetitions() -> None:
    s = synthetiser([_resultat("r1", 0.94), _resultat("r2", 0.96)])
    assert s[("TOUTES", "mrr")]["min"] == 0.94
    assert s[("TOUTES", "mrr")]["max"] == 0.96
    assert s[("latence", "mediane_ms")]["moyenne"] == 900


def test_refuse_de_melanger_des_golden_sets_differents() -> None:
    # Une etiquette corrigee change le score sans que le retrieval bouge :
    # comparer des mesures notees differemment n'aurait aucun sens.
    with pytest.raises(ValueError, match="differents"):
        verifier_comparables([_resultat("a", 0.9, "abc"), _resultat("b", 0.9, "def")])


def test_refuse_un_resultat_sans_empreinte() -> None:
    with pytest.raises(ValueError, match="sans empreinte"):
        verifier_comparables([_resultat("ancien", 0.9, None)])
