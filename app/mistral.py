"""Client HTTP partage vers l'API Mistral.

Extrait de llm.py pour une raison concrete : llm.py appelle desormais le
retrieval, qui appelle les embeddings, qui ont eux aussi besoin de ce
client. Sans ce module, l'import tournerait en rond
(llm -> retrieval -> embeddings -> llm). Mettre le socle commun a part est
la facon propre de casser un cycle d'imports.
"""

import logging

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

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


def decrire_erreur_http(response: httpx.Response) -> str:
    """Transforme l'erreur du fournisseur en message actionnable.

    Un `502 : le LLM a repondu 429` n'aide personne ; il faut savoir *quoi*
    faire. On remonte donc le message du fournisseur et, pour les causes
    frequentes, la marche a suivre.
    """
    try:
        message_fournisseur = response.json().get("message", "")
    except ValueError:
        message_fournisseur = response.text[:200]

    indices = {
        401: "cle API invalide : verifie MISTRAL_API_KEY dans .env",
        403: "cle API sans les droits necessaires",
        404: "modele inconnu : verifie MISTRAL_MODEL",
        422: "requete refusee par le fournisseur",
        # Chez Mistral, le quota est PAR MODELE : un compte parfaitement
        # actif peut avoir 0 req/min sur un modele et 750 sur un autre.
        429: (
            "quota epuise pour CE modele (la limite est par modele) — "
            "regarde l'en-tete x-ratelimit-limit-req-minute ; s'il vaut 0, "
            "change MISTRAL_MODEL"
        ),
    }
    indice = indices.get(response.status_code, "erreur cote fournisseur")

    detail = f"LLM HTTP {response.status_code} ({indice})"
    if message_fournisseur:
        detail += f" : {message_fournisseur}"
    return detail


async def appeler_chat(
    messages: list[dict[str, str]],
    *,
    modele: str | None = None,
    temperature: float | None = None,
    format_json: bool = False,
) -> str:
    """Appel a /chat/completions, pannes traduites en LLMError.

    Deplace depuis llm.py : la reecriture de requete, appelee par le
    retrieval, en a besoin aussi. Or llm.py importe deja le retrieval — le
    laisser la-bas aurait recree un cycle d'imports.
    """
    settings = get_settings()
    payload: dict = {"model": modele or settings.mistral_model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if format_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}

    try:
        response = await get_client().post(
            "/chat/completions", json=payload, headers=headers
        )
    except httpx.HTTPError as exc:
        # Reseau coupe, DNS, timeout... la dependance externe est tombee.
        # On nomme le TYPE d'exception : httpx.ReadTimeout a un str() vide,
        # et « Appel au LLM impossible :  » n'aide personne a 3h du matin.
        logger.exception("Appel LLM impossible")
        raise LLMError(
            f"Appel au LLM impossible ({type(exc).__name__}) : {exc}".rstrip(" :")
        ) from exc

    if response.is_error:
        logger.error("LLM HTTP %s : %s", response.status_code, response.text[:500])
        raise LLMError(decrire_erreur_http(response))

    try:
        contenu = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        logger.exception("Reponse LLM inattendue")
        raise LLMError("Reponse du LLM inexploitable") from exc

    # Le fournisseur peut renvoyer 200 avec un contenu vide (filtrage, budget
    # de tokens epuise...). On refuse de faire passer ca pour une reussite :
    # une reponse vide qui remonte en 200 est un bug qu'on cherche pendant
    # des heures.
    if not contenu or not contenu.strip():
        logger.error("Le LLM a renvoye un contenu vide : %s", response.text[:500])
        raise LLMError("Le LLM a renvoye une reponse vide")

    return contenu
