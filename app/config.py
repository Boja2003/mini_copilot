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

    mistral_model: str = "ministral-8b-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    request_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    """Instance unique, lue une seule fois au premier appel."""
    return Settings()
