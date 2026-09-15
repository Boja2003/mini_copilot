"""Configuration de l'application.

Principe non negociable : aucun secret dans le code. Tout vient de
l'environnement — du fichier `.env` en local, de vraies variables
d'environnement en production (Fly.io / Railway).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Obligatoire : si elle manque, l'app refuse de demarrer avec un
    # message clair, plutot que d'echouer plus tard sur un 401.
    mistral_api_key: str

    # Cle de TON service (a ne pas confondre avec celle de Mistral) : elle
    # protege /chat une fois l'app en ligne. Obligatoire, et c'est
    # volontaire — un service expose sur Internet qui demarre sans
    # authentification parce qu'on a oublie une variable, c'est la facon
    # classique de se faire vider son quota.
    api_key: str

    # 5433 en local (docker-compose expose la base la), "db" dans le
    # reseau Compose. Fly.io fournira sa propre URL.
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"

    mistral_model: str = "ministral-8b-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_embed_model: str = "mistral-embed"
    # Reecriture de requete (voir app/reecriture.py). Desactivee par defaut :
    # elle ajoute un appel LLM a chaque question, et ne s'active que si eval/
    # montre qu'elle en vaut le cout.
    reecriture_requetes: bool = False
    mistral_reecriture_model: str = "ministral-8b-latest"

    # Observabilite (voir app/observabilite.py). Sans les deux cles, le
    # tracage est desactive et l'application tourne exactement comme avant.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "local"
    # Interrupteur explicite : les tests le coupent (tests/conftest.py).
    langfuse_active: bool = True
    # 30 s suffisaient avant le RAG. Avec 5 passages en contexte, la
    # reponse est bien plus longue a generer : mesure en production,
    # une question sur un algorithme depassait les 30 s et tombait en
    # httpx.ReadTimeout.
    request_timeout_seconds: float = 90.0


@lru_cache
def get_settings() -> Settings:
    """Instance unique, lue une seule fois au premier appel."""
    return Settings()
