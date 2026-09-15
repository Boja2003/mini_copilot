"""Observabilite : chaque requete /chat devient une trace Langfuse.

Pourquoi maintenant : apres l'activation de la reecriture de requete, une
reponse est passee de 6,5 s a 16,9 s en production, et on n'a pu que
SUPPOSER la cause. Une trace decompose la requete en etapes mesurees —
reecriture, embeddings, recherches SQL, generation — avec leur duree, leurs
entrees, leurs sorties et les tokens consommes.

Trois regles :
  - sans cles Langfuse, le tracage est desactive et l'application tourne
    exactement comme avant ;
  - les outils en ligne de commande (ingestion, evaluation) ne tracent pas :
    1076 chunks ou 63 questions produiraient des milliers d'observations
    sans requete a laquelle les rattacher ;
  - aucun test n'envoie de trace (voir tests/conftest.py).

Le reste du code ne parle qu'a ce module, jamais directement au SDK :
changer d'outil d'observabilite ne toucherait qu'ici.
"""

import logging
from contextlib import AbstractContextManager
from typing import Any

from langfuse import Langfuse, propagate_attributes

from .config import get_settings

logger = logging.getLogger(__name__)

_client: Langfuse | None = None
_desactive_force = False


def desactiver() -> None:
    """Coupe le tracage pour tout le processus (outils en ligne de commande)."""
    global _desactive_force
    _desactive_force = True


def tracage_actif() -> bool:
    settings = get_settings()
    return (
        not _desactive_force
        and settings.langfuse_active
        and bool(settings.langfuse_public_key)
        and bool(settings.langfuse_secret_key)
    )


def creer_client(**options: Any) -> Langfuse:
    if not tracage_actif():
        # Desactivation EXPLICITE, avec des cles factices. Sans cle, le SDK se
        # desactive aussi de lui-meme, mais en journalisant une
        # « Authentication error » a chaque demarrage : une fausse alerte
        # permanente finit par masquer les vraies.
        return Langfuse(
            public_key="pk-lf-desactive",
            secret_key="sk-lf-desactive",
            tracing_enabled=False,
            **options,
        )
    settings = get_settings()
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        tracing_enabled=True,
        **options,
    )


def get_langfuse() -> Langfuse:
    global _client
    if _client is None:
        _client = creer_client()
        logger.info("Tracage Langfuse %s", "actif" if tracage_actif() else "desactive")
    return _client


def observer(nom: str, type_: str = "span", **attributs: Any) -> AbstractContextManager:
    """Ouvre une etape de la trace courante, ou une nouvelle trace a la racine.

    `type_` suit les types Langfuse : span, generation, embedding, retriever,
    chain... L'objet renvoye expose `.update(output=..., level=...)`.
    """
    return get_langfuse().start_as_current_observation(
        name=nom, as_type=type_, **attributs
    )


def attributs_de_trace(**attributs: Any) -> AbstractContextManager:
    """Nom, tags et metadonnees de la trace, propages a toutes ses etapes."""
    return propagate_attributes(**attributs)


def identifiant_trace() -> str | None:
    """Identifiant de la trace courante, None si le tracage est desactive.

    On ne construit PAS l'URL de la trace ici : `get_trace_url` du SDK fait un
    appel reseau (pour connaitre le projet) et peut lever une exception. Pas
    question d'ajouter un aller-retour et un point de panne a chaque requete.
    """
    return get_langfuse().get_current_trace_id()


def tokens_mistral(usage: dict) -> dict[str, int]:
    """Traduit le compteur de tokens de Mistral dans le vocabulaire Langfuse."""
    correspondance = {
        "input": "prompt_tokens",
        "output": "completion_tokens",
        "total": "total_tokens",
    }
    return {
        cle: usage[source]
        for cle, source in correspondance.items()
        if isinstance(usage.get(source), int)
    }


def fermer() -> None:
    """Envoie les etapes encore en attente, sans arreter le client.

    Appele a l'arret de l'application : sur Fly.io, les machines s'arretent
    faute de trafic, et des etapes encore en file d'attente seraient perdues.

    `flush()` et jamais `shutdown()`, et c'est mesure. Les clients Langfuse
    partagent un gestionnaire de ressources par cle publique
    (`LangfuseResourceManager._instances`). `shutdown()` arrete ses threads
    d'envoi ; un nouveau client avec la meme cle reutilise alors ce
    gestionnaire arrete, et la fermeture suivante attend pour toujours une
    file sans consommateur. C'est ce qui bloquait la suite de tests, ou chaque
    test demarre puis arrete l'application. L'arret definitif reste assure :
    le SDK enregistre lui-meme son `shutdown()` via `atexit`.
    """
    global _client
    client, _client = _client, None
    # Un client desactive n'a rien a envoyer.
    if client is not None and tracage_actif():
        client.flush()
