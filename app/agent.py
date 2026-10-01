"""Couche agent : une question, plusieurs recherches, une synthese.

Mesure qui justifie ce module (eval/resultats/multisauts-*.json, 14
questions demandant chacune plusieurs aspects, memes etiquettes pour les
trois bras) :

                      couverture   questions completes   latence mediane
  production             0.821          9 / 14              873 ms
  question brute         0.857         10 / 14              239 ms
  agent                  0.964         13 / 14             1629 ms

Le motif que l'agent corrige : les cinq passages se concentrent sur
l'aspect le mieux apparie a la question, et l'autre disparait.

Trois decisions, tirees de cette mesure :

1. La decomposition produit directement des requetes en ANGLAIS
   technique. Le corpus est anglophone : decouper et traduire dans le
   meme appel evite de relancer la reecriture pour chaque sous-question.

2. Les resultats sont entrelaces, PAS fusionnes par RRF. C'est contre-
   intuitif par rapport au reste du projet, ou RRF sert partout : RRF
   recompense le consensus, donc un passage trouve par toutes les
   sous-questions. Ici on veut l'inverse — que chaque face garde sa
   place, meme trouvee par une seule sous-question. La ligne « question
   brute » du tableau mesure ce point a elle seule : sans aucune
   decomposition, elle bat deja la production, qui fusionne par RRF des
   requetes reecrites.

3. Le repli ne doit jamais etre silencieux. La premiere version de cette
   mesure attribuait a la decomposition un gain obtenu sans elle : le
   modele repondait une liste d'objets au lieu d'une liste de chaines,
   le filtre la vidait sans rien dire, et l'agent retombait 14 fois sur
   14 sur la question seule. D'ou, ici, une erreur sur toute sortie non
   conforme, et, dans l'evaluation, l'enregistrement des sous-questions
   reellement lancees plus un compteur de replis dans les agregats.

Aucun appel d'outil : les quatre modeles accessibles sur ce compte
ignorent les outils declares et repondent de memoire (teste). Le graphe
est donc explicite, et le modele n'est sollicite que pour une sortie
JSON etroite.
"""

import json
import logging

from .config import get_settings
from .mistral import LLMError, appeler_chat
from .observabilite import observer
from .retrieval import NB_PASSAGES, Passage, RechercheDetaillee, chercher

logger = logging.getLogger(__name__)

NB_SOUS_QUESTIONS_MAX = 3

# ATTENTION A LA FUITE : l'exemple ne doit porter sur aucune notion du corpus
# ni du golden set multi-sauts, sinon on souffle la reponse aux questions
# d'evaluation. L'apprentissage supervise n'est traite dans aucun cours ici.
PROMPT_DECOMPOSITION = """Tu prepares la recherche documentaire qui servira a \
repondre a la question d'un etudiant. Le corpus est redige en ANGLAIS \
(mathematiques, informatique, statistiques).

Decoupe la question en 2 ou 3 requetes de recherche courtes, en anglais, une \
par ASPECT DISTINCT de la question.
- Deux requetes ne doivent jamais chercher la meme chose : si la question \
compare deux notions, il faut une requete par notion.
- Terminologie technique standard des cours anglophones, pas de traduction \
mot a mot.
- Des mots-cles, jamais une phrase interrogative.
- Chaque requete est une CHAINE : un tableau plat, pas des objets.

Exemple pour « Quelle est la difference entre apprentissage supervise et non \
supervise ? » :
{"requetes": ["supervised learning labeled training data", "unsupervised \
learning clustering unlabeled data"]}

Reponds uniquement avec un objet JSON : {"requetes": ["...", "..."]}."""


def _extraire_requetes(brut: str) -> list[str]:
    donnees = json.loads(brut)
    requetes = donnees.get("requetes") if isinstance(donnees, dict) else None
    if not isinstance(requetes, list):
        raise ValueError("cle « requetes » absente ou mal formee")
    retenues = [r.strip() for r in requetes if isinstance(r, str) and r.strip()]
    # Ce garde-fou vient d'une mesure faussee : sur les 14 questions
    # multi-sauts, la decomposition est retombee 14 fois sur la question
    # seule sans que rien ne le signale. Le modele repond parfois une liste
    # d'OBJETS — {"title": ..., "keywords": [...]} — au lieu d'une liste de
    # chaines ; le filtre ci-dessus la vidait en silence. L'evaluation a donc
    # conclu a un gain de la decomposition alors qu'elle n'avait jamais
    # tourne. Une sortie non conforme doit laisser une trace.
    if requetes and not retenues:
        raise ValueError(f"aucune requete exploitable dans {brut[:120]!r}")
    return retenues


async def decomposer(question: str) -> list[str]:
    """Decoupe la question en requetes de recherche, une par aspect.

    Ne leve jamais : si le modele echoue ou repond n'importe quoi, on
    retombe sur la question seule, c'est-a-dire le comportement d'avant.
    L'agent est une amelioration, pas une nouvelle facon de tomber en panne.
    """
    settings = get_settings()
    with observer("decomposition", "chain", input={"question": question}) as etape:
        try:
            brut = await appeler_chat(
                [
                    {"role": "system", "content": PROMPT_DECOMPOSITION},
                    {"role": "user", "content": question},
                ],
                nom="llm-decomposition",
                modele=settings.mistral_reecriture_model,
                temperature=0.0,
                format_json=True,
            )
            requetes = _extraire_requetes(brut)[:NB_SOUS_QUESTIONS_MAX]
        except (LLMError, ValueError) as exc:
            logger.warning("Decomposition impossible, question seule : %s", exc)
            etape.update(
                output=[question],
                level="WARNING",
                status_message=f"repli sur la question seule : {exc}",
            )
            return [question]

        if not requetes:
            etape.update(
                output=[question],
                level="WARNING",
                status_message="aucune sous-question produite",
            )
            return [question]
        etape.update(output=requetes)
        return requetes


def entrelacer(listes: list[list[Passage]], nb: int) -> list[Passage]:
    """Un tour de table : le 1er de chaque liste, puis le 2e, etc.

    Volontairement PAS une fusion RRF, contrairement au reste du projet.
    RRF recompense le consensus : un passage trouve par plusieurs requetes
    remonte. C'est exactement l'inverse du besoin ici, ou chaque face doit
    garder sa place meme si une seule sous-question la trouve.
    """
    retenus: list[Passage] = []
    vus: set = set()
    for rang in range(max((len(liste) for liste in listes), default=0)):
        for liste in listes:
            if rang >= len(liste):
                continue
            passage = liste[rang]
            cle = (
                passage.id
                if passage.id is not None
                else (passage.source, passage.page, passage.contenu)
            )
            if cle in vus:
                continue
            vus.add(cle)
            retenus.append(passage)
            if len(retenus) == nb:
                return retenus
    return retenus


async def chercher_par_faces(
    question: str, nb: int = NB_PASSAGES
) -> RechercheDetaillee:
    """Une recherche par aspect de la question, puis entrelacement.

    Chaque sous-question recoit sa propre recherche hybride complete ; les
    places du contexte final sont ensuite reparties entre elles.
    """
    with observer("agent", "chain", input={"question": question}) as etape:
        requetes = await decomposer(question)
        listes = [await chercher(requete, nb) for requete in requetes]
        passages = entrelacer(listes, nb)
        etape.update(
            output={
                "sous_questions": requetes,
                "passages": [{"document": p.titre, "page": p.page} for p in passages],
            }
        )
    logger.info(
        "Agent : %s sous-questions -> %s passages", len(requetes), len(passages)
    )
    return RechercheDetaillee(passages=passages, requetes=requetes)
