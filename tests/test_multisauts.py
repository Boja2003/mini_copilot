"""Tests de la mesure multi-sauts.

Sans reseau ni base : la couverture ne fait que comparer des (document,
page). Une regle fausse ici donnerait un gain d'agent imaginaire.
"""

import pytest

from app.retrieval import Passage
from eval.multisauts import (
    agreger,
    charger,
    empreinte,
    evaluer_question,
    faces_couvertes,
    pages_en_entiers,
)

FACES = [
    {"nom": "Dijkstra", "pages": {"VII. Shortest paths": [9, 10, 11]}},
    {"nom": "Bellman", "pages": {"VII. Shortest paths": [38, 39, 40]}},
]


def _passage(document: str, page: int) -> Passage:
    return Passage(
        contenu="peu importe",
        source=f"{document}.pptx",
        titre=document,
        page=page,
        distance=0.2,
    )


def test_une_face_couverte_sur_deux() -> None:
    # Le cas typique d'une recherche unique : les cinq passages se
    # concentrent sur la face la mieux appariee a la question.
    passages = [_passage("VII. Shortest paths", p) for p in (9, 10, 11, 12, 13)]
    assert faces_couvertes(FACES, passages) == [True, False]

    resultat = evaluer_question(FACES, passages)
    assert resultat["couverture"] == 0.5
    assert resultat["couverture_totale"] == 0.0
    assert resultat["faces_manquantes"] == ["Bellman"]


def test_les_deux_faces_couvertes() -> None:
    passages = [
        _passage("VII. Shortest paths", 10),
        _passage("VII. Shortest paths", 39),
    ]
    resultat = evaluer_question(FACES, passages)
    assert resultat["couverture"] == 1.0
    assert resultat["couverture_totale"] == 1.0
    assert resultat["faces_manquantes"] == []


def test_bon_document_mauvaise_page_ne_couvre_pas() -> None:
    # Couvrir une face, c'est tomber sur SES pages : le bon document ne
    # suffit pas, sinon une question intra-document serait toujours gagnee.
    passages = [_passage("VII. Shortest paths", 25)]
    assert faces_couvertes(FACES, passages) == [False, False]


def test_bonne_page_mauvais_document_ne_couvre_pas() -> None:
    passages = [_passage("IV. Depth first search", 9)]
    assert faces_couvertes(FACES, passages) == [False, False]


def test_aucun_passage() -> None:
    resultat = evaluer_question(FACES, [])
    assert resultat["couverture"] == 0.0
    assert resultat["faces_manquantes"] == ["Dijkstra", "Bellman"]


def test_golden_set_multisauts_valide() -> None:
    questions = charger()
    assert len(questions) >= 10
    # Chaque question doit vraiment etre multi-sauts.
    assert all(len(q["faces"]) >= 2 for q in questions)
    assert len(empreinte()) == 12


def test_une_seule_face_est_refusee(tmp_path) -> None:
    fichier = tmp_path / "gs.yaml"
    fichier.write_text(
        '- id: a\n  question: q\n  faces:\n    - nom: seule\n      pages: {"D": [1]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="2 faces"):
        charger(fichier)


def test_face_sans_page_refusee(tmp_path) -> None:
    fichier = tmp_path / "gs.yaml"
    fichier.write_text(
        "- id: a\n  question: q\n  faces:\n"
        '    - nom: une\n      pages: {"D": [1]}\n'
        "    - nom: deux\n      pages: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sans page"):
        charger(fichier)


def test_pages_en_plages() -> None:
    from eval.multisauts import pages_en_entiers

    # Les faces valent des sections entieres : les ecrire page par page
    # rendrait le fichier illisible.
    assert pages_en_entiers("9-12") == [9, 10, 11, 12]
    assert pages_en_entiers("16,22") == [16, 22]
    assert pages_en_entiers("23") == [23]
    assert pages_en_entiers([1, 2]) == [1, 2]


def test_plages_normalisees_au_chargement() -> None:
    questions = {q["id"]: q for q in charger()}
    dijkstra = questions["dijkstra-vs-bellman"]["faces"][0]["pages"]
    pages = dijkstra["VII. Shortest paths"]
    assert pages[0] == 9 and pages[-1] == 37
    # Les deux faces d'un meme document ne doivent jamais se recouvrir,
    # sinon un seul passage gagnerait une question de comparaison.
    bellman = questions["dijkstra-vs-bellman"]["faces"][1]["pages"]
    assert not set(pages) & set(bellman["VII. Shortest paths"])


def test_agregats_comptent_les_recherches_sur_la_question_brute() -> None:
    # Garde-fou de tracabilite : si la decomposition tombe en repli, les
    # agregats doivent le dire. La premiere mesure multi-sauts a conclu a un
    # gain de la decomposition alors qu'elle n'avait jamais tourne.
    lignes = [
        {
            "question": "q1",
            "requetes": ["q1"],
            "duree_ms": 10,
            "couverture": 1.0,
            "couverture_totale": 1,
        },
        {
            "question": "q2",
            "requetes": ["aspect a", "aspect b"],
            "duree_ms": 20,
            "couverture": 0.5,
            "couverture_totale": 0,
        },
    ]
    assert agreger(lignes)["question_brute"] == 1


def test_faces_d_un_meme_document_ne_se_recouvrent_pas() -> None:
    # Garde-fou enonce dans l'en-tete du golden set, longtemps reste un
    # commentaire : si deux faces d'une question partagent une page du meme
    # document, un seul passage suffit a couvrir les deux, et la question
    # n'exige plus le moindre saut.
    for q in charger():
        pages_par_document: dict[str, set[int]] = {}
        for face in q["faces"]:
            for document, pages in face["pages"].items():
                deja = pages_par_document.setdefault(document, set())
                communes = deja & set(pages_en_entiers(pages))
                assert not communes, (
                    f"{q['id']} : la face « {face['nom']} » partage les pages "
                    f"{sorted(communes)} de « {document} » avec une autre face"
                )
                deja |= set(pages_en_entiers(pages))
