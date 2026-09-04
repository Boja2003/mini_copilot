"""Tests de la couche LLM : on simule les reponses du fournisseur."""

import httpx
import pytest

from app import llm


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-de-test")
    llm.get_settings.cache_clear()
    yield
    llm.get_settings.cache_clear()


def _mock_transport(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> None:
    async def fake_post(self, url, **kwargs):
        response.request = httpx.Request("POST", "http://test" + url)
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    llm._client = None


def _chat_response(content) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_reponse_normale(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_transport(monkeypatch, _chat_response("Bonjour !"))
    assert await llm.generate_answer("salut") == "Bonjour !"


@pytest.mark.parametrize("contenu_vide", ["", "   ", None])
async def test_contenu_vide_devient_une_erreur(
    monkeypatch: pytest.MonkeyPatch, contenu_vide
) -> None:
    # Le symptome a ne jamais laisser passer en 200 : une reponse vide.
    _mock_transport(monkeypatch, _chat_response(contenu_vide))
    with pytest.raises(llm.LLMError, match="vide"):
        await llm.generate_answer("salut")


async def test_429_explique_le_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_transport(
        monkeypatch,
        httpx.Response(429, json={"message": "Rate limit exceeded"}),
    )
    with pytest.raises(llm.LLMError) as exc:
        await llm.generate_answer("salut")
    assert "429" in str(exc.value)
    assert "quota" in str(exc.value)
    assert "Rate limit exceeded" in str(exc.value)


async def test_401_pointe_vers_la_cle(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_transport(monkeypatch, httpx.Response(401, json={"message": "Unauthorized"}))
    with pytest.raises(llm.LLMError, match="MISTRAL_API_KEY"):
        await llm.generate_answer("salut")
