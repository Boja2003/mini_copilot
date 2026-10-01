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


def test_citation_avec_numero_de_page() -> None:
    # Cas rencontre en production : le modele ajoute la page dans le crochet.
    # La premiere version du controle ne voyait aucune citation ici, et
    # classait une reponse correctement sourcee comme non sourcee.
    assert extraire_citations("D'apres [1, p. 7], l'ACP sert a...") == {1}
    assert extraire_citations("Voir [1, p. 1] et [3, p. 7].") == {1, 3}


def test_le_numero_de_page_n_est_pas_une_citation() -> None:
    # Le 10 est une page, pas un passage : le compter inventerait une
    # citation hors bornes sur une reponse pourtant correcte.
    assert extraire_citations("D'apres [2, p. 10]") == {2}


def test_plusieurs_passages_dans_un_crochet() -> None:
    assert extraire_citations("Les statistiques [2, 5] le montrent.") == {2, 5}


def test_lien_markdown_ignore() -> None:
    assert extraire_citations("Voir [la documentation](https://exemple.fr).") == set()


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


def test_citation_hors_bornes_avec_page() -> None:
    # L'angle mort de la premiere version : une citation inventee ecrite
    # avec un numero de page passait inapercue.
    resultat = evaluer_reponse(
        "D'apres [7, p. 3]", PASSAGES, ["III. Breadth first search"]
    )
    assert resultat["citations_hors_bornes"] == [7]
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


def test_intervalle_mathematique_ignore() -> None:
    # Cas rencontres en production, sur l'ensemble convexe et la section
    # doree : les crochets y notent des intervalles, pas des citations.
    assert extraire_citations(r"pour tout \(\lambda\) dans \([0, 1]\)") == set()
    assert extraire_citations("l'intervalle $[1, 2]$ est ferme") == set()
    # Une borne non entiere ne doit evidemment pas devenir une citation.
    assert extraire_citations(r"\([0, 0.618]\)") == set()


def test_zero_n_est_jamais_une_citation() -> None:
    # Les passages sont numerotes a partir de 1.
    assert extraire_citations("[0]") == set()


def test_citations_et_intervalles_melanges() -> None:
    texte = r"D'apres [2], sur l'intervalle \([0, 1]\), voir aussi [1, p. 4]."
    assert extraire_citations(texte) == {1, 2}
