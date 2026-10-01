"""Tests des controles de generation.

Aucun appel reseau : ces fonctions ne font qu'analyser un texte deja
produit. Une regle fausse ici donnerait un taux de reussite rassurant et
faux — le pire resultat possible pour une evaluation.
"""

from eval.generation import est_un_refus, evaluer_reponse, extraire_citations

PASSAGES = ["III. Breadth first search", "IV. Depth first search"]


def test_extraire_citations() -> None:
    assert extraire_citations("Le BFS utilise une file [1], contrairement [2].") == {
        1,
        2,
    }
    assert extraire_citations("Aucune citation ici.") == set()
    # Un meme passage cite deux fois ne compte qu'une.
    assert extraire_citations("Voir [1] et encore [1].") == {1}


def test_reconnaitre_un_refus() -> None:
    assert est_un_refus("Je ne trouve pas la réponse dans tes supports de cours.")
    # Accents et casse ne doivent pas faire rater le refus.
    assert est_un_refus("JE NE TROUVE PAS cela dans les passages fournis")
    assert not est_un_refus("Le parcours en largeur utilise une file [1].")


def test_citation_hors_bornes_detectee() -> None:
    # Deux passages fournis, la reponse en cite un troisieme : invention.
    resultat = evaluer_reponse(
        "D'apres [1] et [3]", PASSAGES, ["III. Breadth first search"]
    )
    assert resultat["citations_hors_bornes"] == [3]
    assert resultat["citations_valides"] is False


def test_citations_dans_les_bornes() -> None:
    resultat = evaluer_reponse(
        "D'apres [1] et [2]", PASSAGES, ["III. Breadth first search"]
    )
    assert resultat["citations_hors_bornes"] == []
    assert resultat["citations_valides"] is True


def test_reponse_sans_source() -> None:
    resultat = evaluer_reponse(
        "Le BFS explore niveau par niveau.", PASSAGES, PASSAGES[:1]
    )
    assert resultat["cite_une_source"] is False
    # Sans citation, le controle des citations ne s'applique pas.
    assert "citations_valides" not in resultat


def test_refus_sur_hors_sujet_est_correct() -> None:
    resultat = evaluer_reponse(
        "Je ne trouve pas la réponse dans tes supports.", PASSAGES, []
    )
    assert resultat["refus_correct"] is True
    # Une question hors-sujet n'a pas de refus injustifie possible.
    assert "refus_injustifie" not in resultat


def test_absence_de_refus_sur_hors_sujet() -> None:
    resultat = evaluer_reponse(
        "La tarte aux pommes se prepare ainsi [1].", PASSAGES, []
    )
    assert resultat["refus_correct"] is False


def test_refus_injustifie_quand_le_bon_passage_etait_fourni() -> None:
    # Le document attendu figure parmi les passages : le refus n'est
    # imputable qu'a la generation.
    resultat = evaluer_reponse(
        "Je ne trouve pas la réponse dans tes supports.",
        PASSAGES,
        ["III. Breadth first search"],
    )
    assert resultat["refus_injustifie"] is True
    # Un refus n'a pas a citer : on ne lui compte pas l'absence de source.
    assert "cite_une_source" not in resultat


def test_refus_non_imputable_si_le_bon_passage_manquait() -> None:
    # Le retrieval n'a pas fourni le bon document : le refus est legitime,
    # la faute revient a la recherche, mesuree a l'etape 4.
    resultat = evaluer_reponse(
        "Je ne trouve pas la réponse dans tes supports.",
        PASSAGES,
        ["IX Coloring"],
    )
    assert "refus_injustifie" not in resultat
