"""Tests de l'observabilite.

Le tracage est verifie EN MEMOIRE : un exportateur OpenTelemetry remplace
l'envoi vers Langfuse. On controle ce que contient une trace sans reseau et
sans polluer le projet Langfuse.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app import main, mistral, observabilite
from app.llm import Reponse
from app.observabilite import identifiant_trace, observer

CLE = "cle-de-service-de-test"


@pytest.fixture(scope="module")
def client_memoire():
    exportateur = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        environment="test",
        span_exporter=exportateur,
    )
    yield client, exportateur
    client.shutdown()


@pytest.fixture
def etapes(client_memoire, monkeypatch: pytest.MonkeyPatch):
    """Branche le client en memoire ; renvoie une fonction qui liste les
    etapes exportees, par nom."""
    client, exportateur = client_memoire
    exportateur.clear()
    monkeypatch.setattr(observabilite, "_client", client)

    def lire() -> dict:
        client.flush()
        return {s.name: s for s in exportateur.get_finished_spans()}

    return lire


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-mistral-de-test")
    monkeypatch.setenv("API_KEY", CLE)
    main.get_settings.cache_clear()
    yield
    main.get_settings.cache_clear()


def test_sans_cles_le_tracage_est_desactive(monkeypatch: pytest.MonkeyPatch) -> None:
    # Meme avec l'interrupteur ouvert : sans cles, rien ne part.
    monkeypatch.setenv("LANGFUSE_ACTIVE", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    main.get_settings.cache_clear()

    assert not observabilite.tracage_actif()
    with observer("etape"):
        assert identifiant_trace() is None


def test_desactiver_l_emporte_sur_les_cles(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ingestion et evaluation appellent desactiver() : meme avec des cles
    # valides, elles ne doivent produire aucune trace.
    monkeypatch.setenv("LANGFUSE_ACTIVE", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    main.get_settings.cache_clear()
    monkeypatch.setattr(observabilite, "_desactive_force", False)

    assert observabilite.tracage_actif()
    observabilite.desactiver()
    assert not observabilite.tracage_actif()


def test_tokens_mistral_traduits() -> None:
    usage = {"prompt_tokens": 6, "completion_tokens": 161, "total_tokens": 167}
    assert observabilite.tokens_mistral(usage) == {
        "input": 6,
        "output": 161,
        "total": 167,
    }
    # Un compteur absent de la reponse est omis, pas remplace par zero.
    assert observabilite.tokens_mistral({"prompt_tokens": 3}) == {"input": 3}


def _app(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    async def rien(*args, **kwargs):
        return None

    for nom in ["init_schema", "open_pool", "close_pool", "close_client"]:
        monkeypatch.setattr(main, nom, rien)
    # Le client en memoire sert a tout le module : l'arret de l'app ne doit
    # pas le fermer.
    monkeypatch.setattr(main, "fermer_traces", lambda: None)
    return TestClient(main.app)


def test_chat_renvoie_l_identifiant_de_sa_trace(
    etapes, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def faux(question: str) -> Reponse:
        # Une etape imbriquee, comme dans le vrai pipeline.
        with observer("recherche", "retriever", input={"question": question}):
            pass
        return Reponse(texte="ok", passages=[])

    monkeypatch.setattr(main, "generate_answer", faux)
    with _app(monkeypatch) as client:
        reponse = client.post(
            "/chat", json={"question": "Dijkstra ?"}, headers={"X-API-Key": CLE}
        )

    assert reponse.status_code == 200
    trace_id = reponse.json()["trace_id"]
    spans = etapes()
    racine, enfant = spans["chat"], spans["recherche"]
    assert racine.parent is None
    assert enfant.parent.span_id == racine.context.span_id
    assert format(racine.context.trace_id, "032x") == trace_id
    assert racine.attributes["langfuse.trace.name"] == "chat"


def test_requete_refusee_ne_produit_pas_de_trace(
    etapes, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _app(monkeypatch) as client:
        reponse = client.post("/chat", json={"question": "q"})

    assert reponse.status_code == 401
    assert "chat" not in etapes()


def _reponse_http(statut: int, corps: dict):
    async def faux_post(self, url, **kwargs):
        reponse = httpx.Response(statut, json=corps)
        reponse.request = httpx.Request("POST", "http://test" + url)
        return reponse

    return faux_post


async def test_generation_enregistre_modele_et_tokens(
    etapes, monkeypatch: pytest.MonkeyPatch
) -> None:
    corps = {
        "choices": [{"message": {"content": "Bonjour"}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 161, "total_tokens": 167},
    }
    monkeypatch.setattr(httpx.AsyncClient, "post", _reponse_http(200, corps))
    monkeypatch.setattr(mistral, "_client", None)

    await mistral.appeler_chat(
        [{"role": "user", "content": "salut"}],
        nom="llm-reponse",
        modele="ministral-3b-latest",
    )

    generation = etapes()["llm-reponse"]
    assert generation.attributes["langfuse.observation.type"] == "generation"
    assert (
        generation.attributes["langfuse.observation.model.name"]
        == "ministral-3b-latest"
    )
    assert json.loads(generation.attributes["langfuse.observation.usage_details"]) == {
        "input": 6,
        "output": 161,
        "total": 167,
    }


async def test_panne_du_llm_marque_l_etape_en_erreur(
    etapes, monkeypatch: pytest.MonkeyPatch
) -> None:
    corps = {"message": "Rate limit exceeded"}
    monkeypatch.setattr(httpx.AsyncClient, "post", _reponse_http(429, corps))
    monkeypatch.setattr(mistral, "_client", None)

    with pytest.raises(mistral.LLMError):
        await mistral.appeler_chat(
            [{"role": "user", "content": "s"}], nom="llm-reponse"
        )

    generation = etapes()["llm-reponse"]
    niveaux = [v for k, v in generation.attributes.items() if k.endswith(".level")]
    assert niveaux == ["ERROR"]
