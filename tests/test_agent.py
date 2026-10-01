"""Tests de la couche agent.

Aucun appel reseau : le modele et la recherche sont remplaces par des
doubles. L'entrelacement, lui, est une fonction pure — c'est la piece dont
depend toute la couverture multi-sauts.
"""

import pytest

from app import agent
from app.mistral import LLMError
from app.retrieval import Passage


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-mistral-de-test")
    monkeypatch.setenv("API_KEY", "cle-de-service-de-test")
    agent.get_settings.cache_clear()
    yield
    agent.get_settings.cache_clear()


def _p(ident: int, document: str = "Doc", page: int = 1) -> Passage:
    return Passage(
        contenu=f"contenu {ident}",
        source="s",
        titre=document,
        page=page,
        distance=0.2,
        id=ident,
    )


def _llm_repond(monkeypatch: pytest.MonkeyPatch, sortie: str) -> None:
    async def faux(messages, **kwargs):
        return sortie

    monkeypatch.setattr(agent, "appeler_chat", faux)


# --- decomposition -----------------------------------------------------------


async def test_decomposition_une_requete_par_aspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llm_repond(
        monkeypatch,
        '{"requetes": ["dijkstra shortest path", "bellman negative weights"]}',
    )
    assert await agent.decomposer("Compare Dijkstra et Bellman") == [
        "dijkstra shortest path",
        "bellman negative weights",
    ]


async def test_decomposition_bornee(monkeypatch: pytest.MonkeyPatch) -> None:
    # Chaque sous-question coute une recherche complete : le modele ne
    # decide pas tout seul de la facture.
    _llm_repond(monkeypatch, '{"requetes": ["a", "b", "c", "d", "e"]}')
    assert len(await agent.decomposer("q")) == agent.NB_SOUS_QUESTIONS_MAX


async def test_json_invalide_retombe_sur_la_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llm_repond(monkeypatch, "voici les requetes : ...")
    assert await agent.decomposer("ma question") == ["ma question"]


async def test_panne_llm_retombe_sur_la_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def panne(messages, **kwargs):
        raise LLMError("LLM HTTP 429")

    monkeypatch.setattr(agent, "appeler_chat", panne)
    assert await agent.decomposer("ma question") == ["ma question"]


async def test_liste_vide_retombe_sur_la_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llm_repond(monkeypatch, '{"requetes": []}')
    assert await agent.decomposer("ma question") == ["ma question"]


async def test_liste_d_objets_retombe_sur_la_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Le defaut qui a fausse l'evaluation multi-sauts : le modele repond une
    # liste d'OBJETS au lieu d'une liste de chaines. Le filtre sur les
    # chaines vidait la liste, et l'agent retombait sur la question seule 14
    # fois sur 14 — en silence.
    _llm_repond(
        monkeypatch,
        '{"requetes": [{"title": "t", "keywords": ["a", "b"]}]}',
    )
    assert await agent.decomposer("ma question") == ["ma question"]


def test_sortie_non_conforme_ne_passe_pas_en_silence() -> None:
    # Corollaire du test precedent : le repli doit rester visible. Une liste
    # non vide dont rien n'est exploitable est une erreur, pas un resultat
    # vide, sinon le repli ne laisse aucune trace dans les journaux.
    with pytest.raises(ValueError):
        agent._extraire_requetes('{"requetes": [{"title": "t"}]}')
    # Une liste vide, elle, est une reponse vide et non une sortie cassee.
    assert agent._extraire_requetes('{"requetes": []}') == []


def test_prompt_exige_un_tableau_de_chaines() -> None:
    # L'exemple du prompt doit etre du JSON valide dont les elements sont
    # des chaines : c'est exactement ce que le code sait lire. Ce prompt
    # n'est pas passe a str.format, contrairement a celui de reecriture.py
    # — des accolades doublees s'y retrouveraient telles quelles.
    import json

    exemples = [
        ligne.strip()
        for ligne in agent.PROMPT_DECOMPOSITION.splitlines()
        if ligne.lstrip().startswith(chr(123) + chr(34) + "requetes")
    ]
    assert exemples, "l'exemple JSON du prompt est introuvable"
    for ligne in exemples:
        requetes = json.loads(ligne)["requetes"]
        assert requetes and all(isinstance(r, str) for r in requetes)


def test_prompt_sans_notion_du_corpus() -> None:
    # Garde-fou contre la fuite : un exemple tire du corpus ou du golden set
    # multi-sauts souignerait la reponse aux questions d'evaluation.
    prompt = agent.PROMPT_DECOMPOSITION.lower()
    for terme in [
        "dijkstra",
        "bellman",
        "breadth",
        "depth",
        "spanning",
        "chromatic",
        "eigen",
        "convex",
        "newton",
        "gradient",
        "khi",
        "student",
    ]:
        assert terme not in prompt


# --- entrelacement -----------------------------------------------------------


def test_entrelacement_donne_sa_place_a_chaque_liste() -> None:
    # Le defaut mesure : une seule recherche ramene cinq passages sur un
    # seul aspect. Le tour de table garantit une place a chaque face.
    dijkstra = [_p(1), _p(2), _p(3), _p(4), _p(5)]
    bellman = [_p(11), _p(12), _p(13), _p(14), _p(15)]
    retenus = agent.entrelacer([dijkstra, bellman], nb=5)
    assert [p.id for p in retenus] == [1, 11, 2, 12, 3]


def test_entrelacement_sans_doublon() -> None:
    # Un passage trouve par deux sous-questions ne doit pas occuper deux
    # places du contexte.
    a = [_p(1), _p(2)]
    b = [_p(1), _p(3)]
    assert [p.id for p in agent.entrelacer([a, b], nb=4)] == [1, 2, 3]


def test_entrelacement_liste_plus_courte() -> None:
    a = [_p(1)]
    b = [_p(11), _p(12), _p(13)]
    assert [p.id for p in agent.entrelacer([a, b], nb=4)] == [1, 11, 12, 13]


def test_entrelacement_une_seule_liste_preserve_l_ordre() -> None:
    # Si la decomposition echoue, il ne reste qu'une liste : le resultat
    # doit etre celui de la recherche simple.
    liste = [_p(1), _p(2), _p(3)]
    assert [p.id for p in agent.entrelacer([liste], nb=2)] == [1, 2]


def test_entrelacement_sans_liste() -> None:
    assert agent.entrelacer([], nb=5) == []


# --- orchestration -------------------------------------------------------------


async def test_une_recherche_par_sous_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lancees: list[str] = []

    async def faux_decomposer(question):
        return ["dijkstra", "bellman"]

    async def fausse_recherche(requete, nb):
        lancees.append(requete)
        return [_p(1 if requete == "dijkstra" else 11, page=1)]

    monkeypatch.setattr(agent, "decomposer", faux_decomposer)
    monkeypatch.setattr(agent, "chercher", fausse_recherche)

    detail = await agent.chercher_par_faces("Compare Dijkstra et Bellman", nb=5)

    assert lancees == ["dijkstra", "bellman"]
    assert detail.requetes == ["dijkstra", "bellman"]
    assert {p.id for p in detail.passages} == {1, 11}
