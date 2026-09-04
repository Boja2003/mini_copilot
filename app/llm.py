"""La couture du projet : tout ce qui parle au LLM vit ici.

L'appel est ecrit en HTTP brut avec httpx, pas avec le SDK Mistral :
un POST, un header d'authentification, un corps JSON. Aucune magie.

C'est ici — et nulle part ailleurs — que le RAG viendra se greffer a
l'etape 2 : entre `question` et `messages`, on inserera le retrieval et
on collera les passages retrouves dans le prompt. Le reste du code ne
bougera pas.
"""

import logging

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Tu es un assistant precis et concis. Reponds en francais. "
    "Si tu ne sais pas, dis-le franchement plutot que d'inventer."
)

_client: httpx.AsyncClient | None = None


class LLMError(RuntimeError):
    """Le fournisseur de LLM n'a pas pu repondre."""


def get_client() -> httpx.AsyncClient:
    """Client HTTP partage : on reutilise les connexions au lieu d'en
    rouvrir une a chaque requete."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = httpx.AsyncClient(
            base_url=settings.mistral_base_url,
            timeout=settings.request_timeout_seconds,
        )
    return _client


async def close_client() -> None:
    """Appele a l'extinction de l'app (voir le lifespan dans main.py)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def generate_answer(question: str) -> str:
    settings = get_settings()
    payload = {
        "model": settings.mistral_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}

    try:
        response = await get_client().post(
            "/chat/completions", json=payload, headers=headers
        )
    except httpx.HTTPError as exc:
        # Reseau coupe, DNS, timeout... la dependance externe est tombee.
        logger.exception("Appel LLM impossible")
        raise LLMError(f"Appel au LLM impossible : {exc}") from exc

    if response.is_error:
        logger.error("LLM HTTP %s : %s", response.status_code, response.text[:500])
        raise LLMError(_describe_http_error(response))

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        logger.exception("Reponse LLM inattendue")
        raise LLMError("Reponse du LLM inexploitable") from exc

    # Le fournisseur peut renvoyer 200 avec un contenu vide (filtrage, budget
    # de tokens epuise...). On refuse de faire passer ca pour une reussite :
    # une reponse vide qui remonte en 200 est un bug qu'on cherche pendant
    # des heures.
    if not content or not content.strip():
        logger.error("Le LLM a renvoye un contenu vide : %s", response.text[:500])
        raise LLMError("Le LLM a renvoye une reponse vide")

    return content


def _describe_http_error(response: httpx.Response) -> str:
    """Transforme l'erreur du fournisseur en message actionnable.

    Un `502 : le LLM a repondu 429` n'aide personne ; il faut savoir *quoi*
    faire. On remonte donc le message du fournisseur et, pour les causes
    frequentes, la marche a suivre.
    """
    try:
        provider_message = response.json().get("message", "")
    except ValueError:
        provider_message = response.text[:200]

    hints = {
        401: "cle API invalide : verifie MISTRAL_API_KEY dans .env",
        403: "cle API sans les droits necessaires",
        404: "modele inconnu : verifie MISTRAL_MODEL",
        422: "requete refusee par le fournisseur",
        # Chez Mistral, le quota est PAR MODELE : un compte parfaitement
        # actif peut avoir 0 req/min sur un modele et 750 sur un autre.
        # L'en-tete x-ratelimit-limit-req-minute tranche en une seconde.
        429: (
            "quota epuise pour CE modele (la limite est par modele) — "
            "regarde l'en-tete x-ratelimit-limit-req-minute ; s'il vaut 0, "
            "change MISTRAL_MODEL"
        ),
    }
    hint = hints.get(response.status_code, "erreur cote fournisseur")

    detail = f"LLM HTTP {response.status_code} ({hint})"
    if provider_message:
        detail += f" : {provider_message}"
    return detail
