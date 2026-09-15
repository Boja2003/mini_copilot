"""Garde-fou global : aucun test n'envoie de trace vers Langfuse.

Le `.env` local contient les vraies cles Langfuse, et pydantic-settings le
lit. Sans ce garde-fou, lancer `pytest` publierait des traces de test dans
le projet ou arrivent les traces de production.

Une variable d'environnement l'emporte sur le `.env` : on coupe le tracage
avant chaque test, et on repart d'un client neuf.
"""

import pytest

from app import observabilite
from app.config import get_settings


@pytest.fixture(autouse=True)
def _tracage_coupe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_ACTIVE", "false")
    get_settings.cache_clear()
    observabilite._client = None
    yield
    observabilite._client = None
    get_settings.cache_clear()
