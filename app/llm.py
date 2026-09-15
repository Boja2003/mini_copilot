"""La couture du projet : question -> passages -> reponse.

C'est ici que le RAG se fait, exactement a l'endroit annonce a l'etape 1 :
entre « recevoir la question » et « appeler le modele », on va chercher les
passages pertinents du corpus et on les colle dans le prompt.

Le modele ne « connait » pas tes cours. On les lui met sous les yeux a
chaque question, et on lui interdit de repondre autre chose.
"""

import logging
from dataclasses import dataclass

from .config import get_settings
from .mistral import LLMError, appeler_chat
from .retrieval import (
    Passage,
    chercher,
    chercher_avec_reecriture,
    construire_contexte,
)

__all__ = ["LLMError", "Reponse", "generate_answer"]

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

# Nom historique conserve dans ce module : la fonction vit desormais dans
# mistral.py (voir sa docstring pour la raison du deplacement).
_appeler_mistral = appeler_chat


@dataclass(frozen=True)
class Reponse:
    texte: str
    passages: list[Passage]


async def _rechercher(question: str) -> list[Passage]:
    """Choisit la strategie de retrieval selon la configuration.

    La reecriture de requete ajoute un appel LLM a chaque question : elle
    ne s'active que si les chiffres de eval/ montrent qu'elle en vaut le
    cout.
    """
    if get_settings().reecriture_requetes:
        return await chercher_avec_reecriture(question)
    return await chercher(question)


async def generate_answer(question: str) -> Reponse:
    """Le pipeline RAG complet : chercher, puis repondre a partir du trouve."""
    passages = await _rechercher(question)

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
