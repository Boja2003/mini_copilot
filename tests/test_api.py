"""Tests de la couche HTTP.

Ils ne touchent ni l'API Mistral ni Postgres : les deux sont remplaces par
des doubles. Un test qui depend du reseau ou d'un conteneur n'est pas un
test, c'est un pari — et ca casserait la CI.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.llm import Reponse
from app.retrieval import Passage

CLE = "cle-de-service-de-test"
AUTH = {"X-API-Key": CLE}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-mistral-de-test")
    monkeypatch.setenv("API_KEY", CLE)
    main.get_settings.cache_clear()

    # Le lifespan ouvre la base : on la neutralise, ces tests portent sur
    # la couche HTTP.
    async def rien(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "init_schema", rien)
    monkeypatch.setattr(main, "open_pool", rien)
    monkeypatch.setattr(main, "close_pool", rien)
    monkeypatch.setattr(main, "close_client", rien)

    with TestClient(main.app) as test_client:
        yield test_client
    main.get_settings.cache_clear()


def _passage(distance: float = 0.2) -> Passage:
    return Passage(
        contenu="Dijkstra calcule les plus courts chemins.",
        source="VII. Shortest paths.pptx",
        titre="VII. Shortest paths",
        page=10,
        distance=distance,
    )


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reste_ouvert(client: TestClient) -> None:
    # Fly.io interroge /health sans header : il doit rester public.
    assert client.get("/health").status_code == 200


def test_chat_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def faux(question: str) -> Reponse:
        return Reponse(texte=f"reponse a : {question}", passages=[_passage()])

    monkeypatch.setattr(main, "generate_answer", faux)

    response = client.post("/chat", json={"question": "bonjour"}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["answer"] == "reponse a : bonjour"


def test_chat_expose_les_sources(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sans sources, une reponse de RAG n'est pas verifiable.
    async def faux(question: str) -> Reponse:
        return Reponse(texte="peu importe", passages=[_passage(0.1234)])

    monkeypatch.setattr(main, "generate_answer", faux)

    sources = client.post("/chat", json={"question": "q"}, headers=AUTH).json()[
        "sources"
    ]
    assert sources == [
        {"document": "VII. Shortest paths", "page": 10, "distance": 0.1234}
    ]


def test_chat_sans_passage_renvoie_sources_vides(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def faux(question: str) -> Reponse:
        return Reponse(texte="Je ne trouve pas...", passages=[])

    monkeypatch.setattr(main, "generate_answer", faux)

    assert (
        client.post("/chat", json={"question": "q"}, headers=AUTH).json()["sources"]
        == []
    )


def test_chat_question_vide_est_rejetee(client: TestClient) -> None:
    # Pydantic refuse tout seul : on ne code pas cette validation.
    assert client.post("/chat", json={"question": ""}, headers=AUTH).status_code == 422
    assert client.post("/chat", json={}, headers=AUTH).status_code == 422


def test_chat_renvoie_502_si_le_llm_tombe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def casse(question: str) -> Reponse:
        raise main.LLMError("le LLM a repondu 500")

    monkeypatch.setattr(main, "generate_answer", casse)

    assert client.post("/chat", json={"question": "b"}, headers=AUTH).status_code == 502


def test_chat_sans_cle_est_refuse(client: TestClient) -> None:
    assert client.post("/chat", json={"question": "bonjour"}).status_code == 401


def test_chat_avec_mauvaise_cle_est_refuse(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"question": "bonjour"}, headers={"X-API-Key": "mauvaise-cle"}
    )
    assert response.status_code == 401


def test_cle_invalide_ne_touche_jamais_le_llm(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # L'authentification doit couper AVANT tout appel externe : sinon un
    # attaquant consomme le quota meme en echouant.
    async def ne_doit_pas_etre_appele(question: str) -> Reponse:
        raise AssertionError("le LLM a ete appele malgre une cle invalide")

    monkeypatch.setattr(main, "generate_answer", ne_doit_pas_etre_appele)
    assert client.post("/chat", json={"question": "bonjour"}).status_code == 401


def test_la_page_est_servie_sans_cle(client: TestClient) -> None:
    # La page ne contient aucun secret : la proteger n'apporterait rien, et
    # obligerait a poser la cle avant de pouvoir la saisir.
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert reponse.headers["content-type"].startswith("text/html")
    assert "<title>mini_copilot</title>" in reponse.text


def test_la_page_ne_contient_aucune_cle(client: TestClient) -> None:
    # Garde-fou : la cle est saisie par l'utilisateur, jamais ecrite dans la
    # page. Une cle en dur ici serait publiee a chaque deploiement.
    page = client.get("/").text
    assert CLE not in page
    assert "sessionStorage" in page


def test_la_page_appelle_chat_en_en_tete(client: TestClient) -> None:
    # La cle doit voyager en en-tete, pas en parametre d'URL : un parametre
    # finirait dans les journaux d'acces de Fly et du navigateur.
    page = client.get("/").text
    assert '"X-API-Key": cle' in page
    assert "/chat?" not in page


def test_la_page_echappe_avant_d_inserer(client: TestClient) -> None:
    # Verification faite a la main dans un navigateur : une reponse contenant
    # <script> s'affiche bien comme du texte. Ce test garde l'invariant qui
    # le permet — le texte du modele passe par rendre(), qui echappe, et rien
    # d'autre n'est affecte a innerHTML. Un RAG recopie des passages d'un
    # corpus : sa sortie n'est pas du HTML de confiance.
    page = client.get("/").text
    assert 'replace(/&/g, "&amp;")' in page
    assert 'replace(/</g, "&lt;")' in page
    affectations = [
        ligne.strip() for ligne in page.splitlines() if ".innerHTML" in ligne
    ]
    assert affectations, "le rendu de la reponse a change, revoir ce test"
    for ligne in affectations:
        assert "rendre(" in ligne, f"insertion HTML sans echappement : {ligne}"
