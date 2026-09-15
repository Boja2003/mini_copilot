"""Garde-fous sur le golden set lui-meme.

Le golden set est une donnee editee a la main : une coquille (id duplique,
page rattachee au mauvais document, paire orpheline) fausserait les
chiffres sans lever la moindre erreur. Ce test tourne en CI, sans base ni
API : la structure est verifiee a chaque modification.
"""

import pytest

from eval.retrieval import CATEGORIES, GOLDEN_SET, charger_golden_set


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return charger_golden_set()


def test_golden_set_valide(questions: list[dict]) -> None:
    # charger_golden_set leve ValueError a la moindre incoherence.
    assert len(questions) >= 40


def test_chaque_categorie_est_representee(questions: list[dict]) -> None:
    assert {q["categorie"] for q in questions} == CATEGORIES


def test_categories_assez_fournies_pour_comparer(questions: list[dict]) -> None:
    # En dessous, une seule question fait bouger la moyenne de plus de
    # 10 points : la comparaison FR proche / FR eloigne n'aurait pas de sens.
    effectifs = {c: sum(q["categorie"] == c for q in questions) for c in CATEGORIES}
    assert effectifs["fr_proche"] >= 10
    assert effectifs["fr_eloigne"] >= 10


def test_paires_relient_francais_et_anglais(questions: list[dict]) -> None:
    par_id = {q["id"]: q for q in questions}
    for q in questions:
        if q.get("paire"):
            assert q["categorie"].startswith("fr_")
            assert par_id[q["paire"]]["categorie"] == "en"
            assert par_id[q["paire"]]["documents"] == q["documents"]


def test_doublon_detecte(tmp_path) -> None:
    fichier = tmp_path / "gs.yaml"
    fichier.write_text(
        "- {id: a, question: q, categorie: en, documents: [d]}\n"
        "- {id: a, question: q, categorie: en, documents: [d]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="double"):
        charger_golden_set(fichier)


def test_page_hors_documents_attendus_detectee(tmp_path) -> None:
    fichier = tmp_path / "gs.yaml"
    fichier.write_text(
        "- {id: a, question: q, categorie: en, documents: [d], pages: {autre: [1]}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pages hors"):
        charger_golden_set(fichier)


def test_golden_set_versionne() -> None:
    assert GOLDEN_SET.exists()
