"""La couture du projet : question -> passages -> reponse.

C'est ici que le RAG se fait, exactement a l'endroit annonce a l'etape 1 :
entre « recevoir la question » et « appeler le modele », on va chercher les
passages pertinents du corpus et on les colle dans le prompt.

Le modele ne « connait » pas tes cours. On les lui met sous les yeux a
chaque question, et on lui interdit de repondre autre chose.
"""

import logging
from dataclasses import dataclass

import httpx

from .config import get_settings
from .mistral import LLMError, decrire_erreur_http, get_client
from .retrieval import Passage, chercher, construire_contexte

logger = logging.getLogger(__name__)

# Le calibrage a montre que la distance vectorielle ne suffit pas a
# distinguer une question du cours d'une question hors-sujet (les deux
# distributions se chevauchent). C'est donc le prompt qui porte le
# garde-fou : on ordonne au modele de s'en tenir aux passages fournis et
# d'admettre quand ils ne repondent pas. Un RAG qui invente est pire
# qu'inutile : il est credible ET faux.
PROMPT_SYSTEME = """Tu es un assistant qui repond a partir des supports de \
cours fournis, et UNIQUEMENT a partir d'eux.

Regles :
- Appuie chaque affirmation sur les passages numerotes ci-dessous, et cite \
la source entre crochets, par exemple [2].
- Si les passages ne contiennent pas de quoi repondre, dis-le franchement : \
« Je ne trouve pas la reponse dans tes supports de cours. » N'utilise alors \
PAS tes connaissances generales.
- Ne complete jamais un passage par ce que tu crois savoir par ailleurs.
- Reponds en francais, de facon claire et pedagogique."""

PROMPT_SANS_PASSAGE = """Tu es un assistant qui repond a partir des supports \
de cours de l'utilisateur. Aucun passage pertinent n'a ete trouve pour cette \
question. Reponds uniquement : « Je ne trouve pas la reponse dans tes \
supports de cours. », puis propose en une phrase de reformuler la question."""


@dataclass(frozen=True)
class Reponse:
    texte: str
    passages: list[Passage]


async def _appeler_mistral(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    payload = {"model": settings.mistral_model, "messages": messages}
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


async def generate_answer(question: str) -> Reponse:
    """Le pipeline RAG complet : chercher, puis repondre a partir du trouve."""
    passages = await chercher(question)

    if not passages:
        texte = await _appeler_mistral(
            [
                {"role": "system", "content": PROMPT_SANS_PASSAGE},
                {"role": "user", "content": question},
            ]
        )
        return Reponse(texte=texte, passages=[])

    contexte = construire_contexte(passages)
    texte = await _appeler_mistral(
        [
            {"role": "system", "content": PROMPT_SYSTEME},
            {
                "role": "user",
                "content": f"Passages issus de mes cours :\n\n{contexte}\n\n"
                f"Question : {question}",
            },
        ]
    )
    return Reponse(texte=texte, passages=passages)
