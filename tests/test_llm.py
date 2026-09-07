"""Tests de la couche LLM.

Deux niveaux distincts :
  - `_appeler_mistral` : la traduction des reponses HTTP du fournisseur ;
  - `generate_answer`  : l'orchestration RAG (chercher puis repondre).
Le retrieval est remplace par un double : ces tests ne touchent pas Postgres.
"""

import httpx
import pytest

from app import llm
from app.retrieval import Passage


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-mistral-de-test")
    monkeypatch.setenv("API_KEY", "cle-de-service-de-test")
    llm.get_settings.cache_clear()
    yield
    llm.get_settings.cache_clear()


def _simuler_reponse(monkeypatch: pytest.MonkeyPatch, reponse: httpx.Response) -> None:
    async def faux_post(self, url, **kwargs):
        reponse.request = httpx.Request("POST", "http://test" + url)
        return reponse

    monkeypatch.setattr(httpx.AsyncClient, "post", faux_post)
    import app.mistral

    app.mistral._client = None


def _reponse_chat(contenu) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": contenu}}]})


def _passage(distance: float = 0.2) -> Passage:
    return Passage(
        contenu="Dijkstra calcule les plus courts chemins.",
        source="VII. Shortest paths.pptx",
        titre="VII. Shortest paths",
        page=10,
        distance=distance,
    )


# --- couche HTTP -----------------------------------------------------------


async def test_reponse_normale(monkeypatch: pytest.MonkeyPatch) -> None:
    _simuler_reponse(monkeypatch, _reponse_chat("Bonjour !"))
    assert await llm._appeler_mistral([{"role": "user", "content": "s"}]) == "Bonjour !"


@pytest.mark.parametrize("contenu_vide", ["", "   ", None])
async def test_contenu_vide_devient_une_erreur(
    monkeypatch: pytest.MonkeyPatch, contenu_vide
) -> None:
    # Le symptome a ne jamais laisser passer en 200 : une reponse vide.
    _simuler_reponse(monkeypatch, _reponse_chat(contenu_vide))
    with pytest.raises(llm.LLMError, match="vide"):
        await llm._appeler_mistral([{"role": "user", "content": "s"}])


async def test_429_explique_le_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    _simuler_reponse(
        monkeypatch, httpx.Response(429, json={"message": "Rate limit exceeded"})
    )
    with pytest.raises(llm.LLMError) as exc:
        await llm._appeler_mistral([{"role": "user", "content": "s"}])
    assert "429" in str(exc.value)
    assert "par modele" in str(exc.value)
    assert "Rate limit exceeded" in str(exc.value)


async def test_401_pointe_vers_la_cle(monkeypatch: pytest.MonkeyPatch) -> None:
    _simuler_reponse(monkeypatch, httpx.Response(401, json={"message": "Unauthorized"}))
    with pytest.raises(llm.LLMError, match="MISTRAL_API_KEY"):
        await llm._appeler_mistral([{"role": "user", "content": "s"}])


# --- orchestration RAG -----------------------------------------------------


async def test_les_passages_sont_injectes_dans_le_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le coeur du RAG : sans passage dans le prompt, ce n'est plus du RAG."""
    captures: list[list[dict]] = []

    async def faux_chercher(question, nb=5):
        return [_passage()]

    async def faux_appel(messages):
        captures.append(messages)
        return "peu importe"

    monkeypatch.setattr(llm, "chercher", faux_chercher)
    monkeypatch.setattr(llm, "_appeler_mistral", faux_appel)

    reponse = await llm.generate_answer("Explique Dijkstra")

    prompt_utilisateur = captures[0][1]["content"]
    assert "Dijkstra calcule les plus courts chemins." in prompt_utilisateur
    assert "VII. Shortest paths (p. 10)" in prompt_utilisateur
    assert reponse.passages == [_passage()]


async def test_sans_passage_le_modele_recoit_la_consigne_de_refus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[list[dict]] = []

    async def aucun_passage(question, nb=5):
        return []

    async def faux_appel(messages):
        captures.append(messages)
        return "Je ne trouve pas."

    monkeypatch.setattr(llm, "chercher", aucun_passage)
    monkeypatch.setattr(llm, "_appeler_mistral", faux_appel)

    reponse = await llm.generate_answer("recette de tarte")

    assert reponse.passages == []
    assert captures[0][0]["content"] == llm.PROMPT_SANS_PASSAGE
