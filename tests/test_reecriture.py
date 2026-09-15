"""Tests de la reecriture de requete et de la fusion multi-requetes.

Aucun appel reseau : le LLM, les embeddings et la base sont remplaces par
des doubles.
"""

import pytest

from app import reecriture, retrieval
from app.mistral import LLMError
from app.retrieval import Passage, fusionner_rrf


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-mistral-de-test")
    monkeypatch.setenv("API_KEY", "cle-de-service-de-test")
    reecriture.get_settings.cache_clear()
    yield
    reecriture.get_settings.cache_clear()


def _p(ident: int, distance: float = 0.2) -> Passage:
    return Passage(
        contenu=f"contenu {ident}",
        source="s",
        titre="Doc",
        page=1,
        distance=distance,
        id=ident,
    )


def _llm_repond(monkeypatch: pytest.MonkeyPatch, sortie: str) -> None:
    async def faux(messages, **kwargs):
        return sortie

    monkeypatch.setattr(reecriture, "appeler_chat", faux)


# --- reformuler --------------------------------------------------------------


async def test_question_d_origine_conservee_en_tete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llm_repond(monkeypatch, '{"requetes": ["graph search", "traversal order"]}')
    assert await reecriture.reformuler("question fr") == [
        "question fr",
        "graph search",
        "traversal order",
    ]


async def test_json_invalide_retombe_sur_la_question_seule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llm_repond(monkeypatch, "desole, voici des requetes : ...")
    assert await reecriture.reformuler("q") == ["q"]


async def test_cle_absente_retombe_sur_la_question_seule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llm_repond(monkeypatch, '{"queries": ["a", "b"]}')
    assert await reecriture.reformuler("q") == ["q"]


async def test_panne_llm_retombe_sur_la_question_seule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # La reecriture ne doit jamais faire tomber /chat : sans elle, on
    # retrouve exactement la recherche d'avant.
    async def panne(messages, **kwargs):
        raise LLMError("LLM HTTP 429")

    monkeypatch.setattr(reecriture, "appeler_chat", panne)
    assert await reecriture.reformuler("q") == ["q"]


async def test_doublons_et_requetes_vides_ecartes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llm_repond(monkeypatch, '{"requetes": ["Q", "", "  ", "x", "x"]}')
    assert await reecriture.reformuler("q") == ["q", "x"]


async def test_nombre_de_reformulations_borne(monkeypatch: pytest.MonkeyPatch) -> None:
    # Chaque requete coute une recherche en base : le modele ne decide pas
    # tout seul de la facture.
    _llm_repond(monkeypatch, '{"requetes": ["a", "b", "c", "d"]}')
    assert await reecriture.reformuler("q") == ["q", "a", "b"]


async def test_appel_deterministe_en_mode_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[dict] = []

    async def faux(messages, **kwargs):
        captures.append(kwargs)
        return '{"requetes": []}'

    monkeypatch.setattr(reecriture, "appeler_chat", faux)
    await reecriture.reformuler("q")
    assert captures[0]["temperature"] == 0.0
    assert captures[0]["format_json"] is True


def test_prompt_sans_terme_du_golden_set() -> None:
    # Garde-fou contre la fuite : un exemple tire du golden set dans le
    # prompt gonflerait artificiellement le score de la reecriture.
    prompt = reecriture.PROMPT_REECRITURE.lower()
    for terme in ["breadth", "largeur", "depth", "profondeur", "spanning", "eigen"]:
        assert terme not in prompt


# --- fusion ------------------------------------------------------------------


def test_une_seule_liste_preserve_l_ordre() -> None:
    # Si la reecriture echoue, il ne reste qu'une liste : le resultat doit
    # etre celui de la recherche simple.
    assert fusionner_rrf([[_p(1), _p(2), _p(3)]], nb=2) == [_p(1), _p(2)]


def test_passage_present_dans_deux_listes_remonte() -> None:
    # 3 est dernier dans chaque liste, mais cumule deux votes :
    # 2/(60+3) = 0.0317 > 1/(60+1) = 0.0164.
    a = [_p(1), _p(2), _p(3)]
    b = [_p(9), _p(8), _p(3)]
    assert fusionner_rrf([a, b], nb=1)[0].id == 3


def test_fusion_garde_la_meilleure_distance() -> None:
    fusion = fusionner_rrf([[_p(1, distance=0.30)], [_p(1, distance=0.12)]], nb=1)
    assert fusion[0].distance == 0.12


def test_fusion_sans_identifiant_reconnait_le_meme_passage() -> None:
    p1 = Passage(contenu="x", source="s", titre="t", page=1, distance=0.2)
    p2 = Passage(contenu="x", source="s", titre="t", page=1, distance=0.1)
    assert len(fusionner_rrf([[p1], [p2]], nb=5)) == 1


# --- orchestration -------------------------------------------------------------


async def test_un_seul_appel_embeddings_pour_toutes_les_requetes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lots: list[list[str]] = []

    async def faux_reformuler(question):
        return [question, "english query"]

    async def faux_embarquer(textes):
        lots.append(list(textes))
        return [[0.1, 0.2, 0.3] for _ in textes]

    async def faux_chercher_vecteur(vecteur, texte, nb):
        return [_p(1 if texte == "english query" else 2)]

    monkeypatch.setattr(retrieval, "reformuler", faux_reformuler)
    monkeypatch.setattr(retrieval, "embarquer", faux_embarquer)
    monkeypatch.setattr(retrieval, "_chercher_vecteur", faux_chercher_vecteur)

    passages = await retrieval.chercher_avec_reecriture("question fr", nb=5)

    assert lots == [["question fr", "english query"]]
    assert {p.id for p in passages} == {1, 2}


async def test_detail_renvoie_les_requetes_utilisees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Les requetes s'enregistrent au moment de la mesure : la reecriture
    # n'est pas reproductible, rappeler le LLM apres coup ne redonne pas
    # forcement les memes.
    async def faux_reformuler(question):
        return [question, "english query"]

    async def faux_rechercher(requetes, nb):
        return [_p(1)]

    monkeypatch.setattr(retrieval, "reformuler", faux_reformuler)
    monkeypatch.setattr(retrieval, "rechercher_requetes", faux_rechercher)

    detail = await retrieval.chercher_avec_reecriture_detail("q fr", nb=5)

    assert detail.requetes == ["q fr", "english query"]
    assert detail.passages == [_p(1)]
