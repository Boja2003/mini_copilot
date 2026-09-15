"""Reecriture de requete : traduire l'intention de l'etudiant dans la langue
et le vocabulaire du corpus.

Le probleme mesure : le corpus est en anglais, les questions en francais.
Quand le terme cle se ressemble dans les deux langues (Dijkstra, convexe /
convex), la recherche s'en sort. Quand il ne se ressemble pas (« parcours en
largeur » / « breadth-first search »), elle rate.

Le LLM sert ici de traducteur terminologique, pas de repondeur : il ne voit
aucun passage et ne produit que des requetes de recherche.
"""

import json
import logging

from .config import get_settings
from .mistral import LLMError, appeler_chat

logger = logging.getLogger(__name__)

NB_REFORMULATIONS = 2

# ATTENTION A LA FUITE : l'exemple de ce prompt ne doit JAMAIS venir du
# golden set. Ecrire « parcours en largeur -> breadth-first search » ici,
# ce serait souffler la reponse a la question d'evaluation qui echouait : le
# chiffre monterait, et il ne mesurerait plus rien. D'ou un exemple pris
# hors des cours.
PROMPT_REECRITURE = """Tu transformes une question d'etudiant en requetes de \
recherche pour un corpus de supports de cours redige en ANGLAIS \
(mathematiques, informatique, statistiques).

Produis exactement {n} requetes de recherche courtes, en anglais, avec la \
terminologie technique standard des cours anglophones.
- Traduis chaque notion par son nom consacre en anglais, pas mot a mot.
- Garde les noms propres tels quels.
- Pas de phrase interrogative : des mots-cles et expressions techniques.

Exemple pour « Qu'est-ce que l'apprentissage par renforcement ? » :
{{"requetes": ["reinforcement learning", "reinforcement learning reward policy"]}}

Reponds uniquement avec un objet JSON de la forme {{"requetes": [...]}}."""


def _extraire_requetes(brut: str) -> list[str]:
    """Lit la sortie JSON du modele ; leve ValueError si elle est inexploitable."""
    donnees = json.loads(brut)
    requetes = donnees.get("requetes") if isinstance(donnees, dict) else None
    if not isinstance(requetes, list):
        raise ValueError("cle « requetes » absente ou mal formee")
    return [r.strip() for r in requetes if isinstance(r, str) and r.strip()]


async def reformuler(question: str) -> list[str]:
    """Renvoie la question d'origine, suivie de ses reformulations.

    Ne leve jamais : si le LLM echoue ou repond n'importe quoi, on retombe
    sur la question seule, c'est-a-dire exactement la recherche d'avant. La
    reecriture est une amelioration, pas une nouvelle facon de tomber en
    panne.
    """
    settings = get_settings()
    try:
        brut = await appeler_chat(
            [
                {
                    "role": "system",
                    "content": PROMPT_REECRITURE.format(n=NB_REFORMULATIONS),
                },
                {"role": "user", "content": question},
            ],
            modele=settings.mistral_reecriture_model,
            # Temperature nulle : elle reduit la variabilite, sans la supprimer.
            # Mesure : trois appels identiques ne donnent presque jamais les
            # memes requetes, et le parametre random_seed de Mistral n'y change
            # rien. D'ou l'enregistrement des requetes dans les resultats
            # d'evaluation (voir eval/retrieval.py).
            temperature=0.0,
            format_json=True,
        )
        reformulations = _extraire_requetes(brut)
    except (LLMError, ValueError) as exc:
        logger.warning("Reecriture impossible, question seule : %s", exc)
        return [question]

    requetes = [question]
    for reformulation in reformulations[:NB_REFORMULATIONS]:
        if reformulation.casefold() not in {r.casefold() for r in requetes}:
            requetes.append(reformulation)
    return requetes
