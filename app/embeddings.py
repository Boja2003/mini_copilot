"""Transformation du texte en vecteurs, via l'API mistral-embed.

Deux contraintes dictent tout ce fichier :
  - le quota est de 60 requetes/minute sur ce compte, donc on envoie les
    chunks par lots au lieu d'un appel par chunk (1076 appels tiendraient
    18 minutes ; en lots de 32, ca tient en une minute) ;
  - une API distante echoue parfois. Un 429 n'est pas une erreur fatale,
    c'est une invitation a ralentir : on reessaie avec un delai croissant.
"""

import asyncio
import logging

import httpx

from .config import get_settings
from .mistral import LLMError, get_client
from .observabilite import observer, tokens_mistral

logger = logging.getLogger(__name__)

# 32 x ~300 tokens = ~10k tokens par requete : large marge sous les limites
# de l'API, et assez gros pour que le quota ne soit pas le facteur limitant.
TAILLE_LOT = 32
TENTATIVES_MAX = 5
# Au-dela, on ne recopie pas les textes dans la trace : un lot d'ingestion
# publierait des pages entieres de cours pour rien.
TEXTES_TRACES_MAX = 4


async def _appeler_embeddings(textes: list[str]) -> list[list[float]]:
    settings = get_settings()
    with observer(
        "mistral-embeddings",
        "embedding",
        model=settings.mistral_embed_model,
        input=textes if len(textes) <= TEXTES_TRACES_MAX else f"{len(textes)} textes",
    ) as etape:
        try:
            vecteurs, usage, attente = await _executer_embeddings(textes)
        except LLMError as exc:
            etape.update(level="ERROR", status_message=str(exc))
            raise
        etape.update(
            output=f"{len(vecteurs)} vecteurs",
            usage_details=tokens_mistral(usage),
            # Le temps passe a attendre le quota apparait dans la duree de
            # l'etape ; on le chiffre pour ne pas l'imputer a l'API.
            metadata={"attente_quota_s": attente},
        )
        return vecteurs


async def _executer_embeddings(
    textes: list[str],
) -> tuple[list[list[float]], dict, float]:
    settings = get_settings()
    payload = {"model": settings.mistral_embed_model, "input": textes}
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}

    delai = 2.0
    attente = 0.0
    for tentative in range(1, TENTATIVES_MAX + 1):
        try:
            reponse = await get_client().post(
                "/embeddings", json=payload, headers=headers, timeout=120.0
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Appel embeddings impossible : {exc}") from exc

        if reponse.status_code == 429:
            # Back-off exponentiel : on laisse la fenetre de quota se vider
            # au lieu de marteler l'API, ce qui ne ferait que la prolonger.
            logger.warning(
                "Quota embeddings atteint (tentative %s/%s), pause de %.0fs",
                tentative,
                TENTATIVES_MAX,
                delai,
            )
            await asyncio.sleep(delai)
            attente += delai
            delai *= 2
            continue

        if reponse.is_error:
            raise LLMError(
                f"Embeddings HTTP {reponse.status_code} : {reponse.text[:200]}"
            )

        corps = reponse.json()
        donnees = corps["data"]
        # L'API garantit l'ordre, mais on trie sur l'index par securite :
        # un decalage ici associerait chaque texte au vecteur d'un autre,
        # et le bug serait invisible jusqu'a ce que le retrieval deraille.
        donnees.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in donnees], corps.get("usage") or {}, attente

    raise LLMError(
        f"Quota embeddings toujours sature apres {TENTATIVES_MAX} tentatives"
    )


async def embarquer(textes: list[str]) -> list[list[float]]:
    """Vectorise une liste de textes, par lots, en preservant l'ordre."""
    vecteurs: list[list[float]] = []
    for debut in range(0, len(textes), TAILLE_LOT):
        lot = textes[debut : debut + TAILLE_LOT]
        vecteurs.extend(await _appeler_embeddings(lot))
        logger.info("Embeddings : %s/%s", len(vecteurs), len(textes))
    return vecteurs


async def embarquer_un(texte: str) -> list[float]:
    """Cas de la question posee a /chat : un seul texte a vectoriser."""
    return (await _appeler_embeddings([texte]))[0]
