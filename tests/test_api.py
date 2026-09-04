"""Tests de l'etape 1.

Ils ne touchent jamais l'API Mistral : on remplace `generate_answer` par
un double. Un test qui depend du reseau n'est pas un test, c'est un pari
— et ca casserait la CI.
"""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-de-test")
    main.get_settings.cache_clear()
    with TestClient(main.app) as test_client:
        yield test_client
    main.get_settings.cache_clear()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_answer(question: str) -> str:
        return f"reponse a : {question}"

    monkeypatch.setattr(main, "generate_answer", fake_generate_answer)

    response = client.post("/chat", json={"question": "bonjour"})
    assert response.status_code == 200
    assert response.json()["answer"] == "reponse a : bonjour"


def test_chat_question_vide_est_rejetee(client: TestClient) -> None:
    # Pydantic refuse tout seul : on ne code pas cette validation.
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_renvoie_502_si_le_llm_tombe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_generate_answer(question: str) -> str:
        raise main.LLMError("le LLM a repondu 500")

    monkeypatch.setattr(main, "generate_answer", broken_generate_answer)

    response = client.post("/chat", json={"question": "bonjour"})
    assert response.status_code == 502
